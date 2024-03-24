import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
import sys
from tqdm import tqdm
import argparse
import time
import numpy as np
import datetime
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import random

multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0
    for step, batch in enumerate(loader):
        if(step%100==0):
            print("step",step)
            sys.stdout.flush()
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            loss_accum += loss.item()
            loss.backward()
            optimizer.step()
    print('Train Loss',loss_accum/(step+1))
    return loss_accum/(step+1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    loss_accum = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            loss_accum += loss.item()
            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict),loss_accum/(step+1)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--lr', type=int, default=0.001,
                        help='learning rate')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=2000,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='number of workers (default: 0)')
    parser.add_argument('--split_way', type=str, default=None,
                        help='split method(by node number, edge number of diameter) (default: None)')
    parser.add_argument('--split_upbound', type=int, default=5)
    parser.add_argument('--split_lowbound', type=int, default=4)
    parser.add_argument('--random_seed', type=int, default=3)
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--training_ratio',type=float,default=1,
                        help="must be lower than or equal to 0.9")
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()
    print(args)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
   
    ### automatic dataloading and splitting

    dataset = PygGraphPropPredDataset(name = args.dataset, transform = add_zeros,root='/egr/research-dselab/liujin33/graph_scaling_law/dataset')

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    if args.split_way == None:
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(args.training_ratio*len(split_idx["train"]))]
        print(len(subset_idx))
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    else: 
        raw_data = dataset
        select_index = []
        if args.split_way == "node":
            for i in range(len(raw_data)):
                data = raw_data[i]
                if data.num_nodes<=args.split_upbound and data.num_nodes>args.split_lowbound:
                    select_index.append(i)
        elif args.split_way == "edge":
            for i in range(len(raw_data)):
                data = raw_data[i]
                if data.edge_attr.shape[0]<=args.split_upbound and data.edge_attr.shape[0]>args.split_lowbound:
                    select_index.append(i)
        elif args.split_way == "diameter":
            dia = np.load('/egr/research-dselab/liujin33/graph_scaling_law/ppa_dias.npy')
            for i in range(len(raw_data)):
                if dia[i]<=args.split_upbound and dia[i]>args.split_lowbound:
                    select_index.append(i)
        select_data = raw_data[select_index]
        rand_index = torch.randperm(len(select_index))
        print("we choose that split by "+args.split_way)
        print("our upper bound is "+str(args.split_upbound))
        print("our lower bound is "+str(args.split_lowbound))
        print("there are "+str(len(select_index))+" total graphs we get")
        sys.stdout.flush()
        train_idx = rand_index[:int(0.8*len(select_index))]
        valid_idx = rand_index[int(0.8*len(select_index)):int(0.9*len(select_index))]
        test_idx = rand_index[int(0.8*len(select_index)):]
        train_loader = DataLoader(select_data[train_idx], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        valid_loader = DataLoader(select_data[valid_idx], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(select_data[test_idx], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
     
    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    
    print(f'#Params: {sum(p.numel() for p in model.parameters())}')
    params = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    val_loss_curve=[]
    test_loss_curve=[]
    #train_curve = []
    import wandb
    wandb.init(
    # set the wandb project where this run will be logged
    project='ppa_basic_scale_gcn',
    
    # track hyperparameters and run metadata
    config={
    "random seed":args.random_seed,
    "learning_rate": args.lr,
    "architecture": args.gnn,
    "dataset": args.dataset,
    "epochs": args.epochs,
    "drop_ratio": args.drop_ratio,
    "num_layer": args.num_layer,
    "emb_dim": args.emb_dim,
    "batch_size": args.batch_size,
    "training_ratio": args.training_ratio
    }
    )
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch),datetime.datetime.now())
        print('Training...')
        train_loss = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        #train_perf = eval(model, device, train_loader, evaluator)
        valid_perf, valloss = eval(model, device, valid_loader, evaluator)
        test_perf,testloss = eval(model, device, test_loader, evaluator)

        print({'Validation perf': valid_perf, 'Test pref': test_perf, 'Val loss':valloss, 'Test loss':testloss})

        #train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        val_loss_curve.append(valloss)
        test_loss_curve.append(testloss)
        wandb.log({"train loss":train_loss,"val acc": valid_perf, "val loss": valloss, "test acc": test_perf, "test loss":testloss})


    best_val_epoch = np.argmax(np.array(valid_curve))
    best_val_loss_epoch = np.argmin(np.array(val_loss_curve))
    #best_train = max(train_curve)
    
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]),'Epoch:',best_val_epoch)
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    print('Best validation Loss: {}'.format(val_loss_curve[best_val_loss_epoch]),'Epoch:',best_val_loss_epoch)
    print('Test loss: {}'.format(test_loss_curve[best_val_loss_epoch]))
    wandb.log({"para num": params, "best val score": valid_curve[best_val_epoch], "best test score":test_curve[best_val_epoch],
               "best val loss":val_loss_curve[best_val_loss_epoch],"best test loss":test_loss_curve[best_val_loss_epoch]})

    # if not args.filename == '':
    #     torch.save('Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch])


if __name__ == "__main__":
    main()
