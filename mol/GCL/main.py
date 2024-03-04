import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam

from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import argparse

import numpy as np
import random

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def accuracy(true_values,predictions):
    N = true_values.shape[0]
    accuracy = (true_values == predictions).sum() / N
    return accuracy


multicls_criterion = torch.nn.CrossEntropyLoss()
cls_criterion = torch.nn.BCEWithLogitsLoss()


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train_contrast(encoder_model, contrast_model, dataloader, optimizer,device):
    encoder_model.train()
    epoch_loss = 0
    for step,data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss/step


def train_linear_probing(encoder_model, linear_layer, dataloader, optimizer,device,finetune):
    encoder_model.train()
    if finetune==0:
        encoder_model.eval()
        for param in encoder_model.parameters():
            param.requires_grad = False
    linear_layer.train()
    epoch_loss = 0
    for step,data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        pred = linear_layer(g)
        #g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        #loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss = multicls_criterion(pred.to(torch.float32), data.y.view(-1,))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/step



def test(encoder_model,linear_layer, dataloader,device,evaluator):
    encoder_model.eval()

    linear_layer.eval()
    x = []
    y = []
    y_true = []
    y_pred = []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        with torch.no_grad():
            _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
            pred = linear_layer(g)
        #loss = multicls_criterion(pred.to(torch.float32), data.y.view(-1,))
        y_true.append(data.y.view(-1,1).detach().cpu())
        y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def main():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=2,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--training_ratio',type=float,default=0.9,
                        help="must be lower than or equal to 0.9")
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--random_seed',type=int,default=7)
    parser.add_argument('--finetune',type=int,default=1)
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()
    print(args)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    path = "/egr/research-dselab/liujin33/graph_scaling_law/models/mol/dataset"
    #dataset = TUDataset(path, name='COLLAB')
    dataset = PygGraphPropPredDataset(name = args.dataset,root=path)
    y_dic = {}
    for d in dataset:
        if str(d.y) not in y_dic:
            y_dic[str(d.y)] = 1
        else:
            y_dic[str(d.y)] += 1
        
    print(len(dataset))
    print(y_dic)
    perm_index = torch.randperm(len(dataset))
    train_index = perm_index[:int(args.training_ratio*len(dataset))]
    val_index = perm_index[int(0.8*len(dataset)):int(0.9*len(dataset))]
    test_index = perm_index[int(0.9*len(dataset)):]
    #idx = [i for i in range(len(dataset))]
    #split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    #evaluator = accuracy

    train_loader = DataLoader(dataset[train_index], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[val_index], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[test_index], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    #dataloader = DataLoader(dataset, batch_size=args.batch_size)
    input_dim = max(dataset.num_features, 1)
    
    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    linear_output_layer = LogReg(args.emb_dim*args.num_layer, args.emb_dim).to(device)
    gconv = GConv(input_dim=input_dim, hidden_dim=args.emb_dim, num_layers=args.num_layer).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr)
    optimizer2 = Adam([
                {'params': encoder_model.parameters()},
                {'params': linear_output_layer.parameters()}
            ], lr=0.001)
    evaluator = Evaluator(args.dataset)
    import wandb
    wandb.init(
    # set the wandb project where this run will be logged
    project=args.dataset+"-self-supervised",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": args.gnn,
    "epochs": args.epochs,
    "drop_ratio": args.drop_ratio,
    "num_layer": args.num_layer,
    "emb_dim": args.emb_dim,
    "batch_size": args.batch_size,
    "training_ratio": args.training_ratio,
    "random": args.random_seed
    }
    )
    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, args.epochs+1):
            loss = train_contrast(encoder_model, contrast_model, train_loader, optimizer,device)
            wandb.log({"train_contrast_loss":loss})
            #print("epoch: "+str(epoch),"loss: "+str(loss))
            pbar.set_postfix({'loss': loss})
            pbar.update()
    with tqdm(total=50, desc='(T)') as pbar:
        for epoch in range(1, int(args.epochs/2)+1):
            loss = train_linear_probing(encoder_model, linear_output_layer, valid_loader, optimizer2,device, args.finetune)
            wandb.log({"train_probing_loss":loss})
            #print("epoch: "+str(epoch),"loss: "+str(loss))
            pbar.set_postfix({'loss': loss})
            pbar.update()
    test_result = test(encoder_model, linear_output_layer,test_loader,device,evaluator)
    print(test_result)
    an = test_result['rocauc']
    print(f'AUC={an:.4f}')
    wandb.log({'AUC':test_result})


if __name__ == '__main__':
    main()

