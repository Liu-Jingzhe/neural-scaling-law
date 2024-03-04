import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

dataset = PygGraphPropPredDataset(name = "ogbg-molhiv")
print(dataset[12])
N_entropy = []
density = []
average_degree = []
degree_var = []
diameter = []
node_num = []
edge_num = []
g = nx.Graph(dataset[1000].edge_index.T.numpy().tolist())
for g in tqdm(dataset):
    g = nx.Graph(g.edge_index.T.numpy().tolist())
    e = g.number_of_edges()
    n = g.number_of_nodes()
    node_num.append(int(n))
    edge_num.append(int(e))
    degrees = np.array([val for (node, val) in g.degree()])
    average_degree.append(float(np.mean(degrees)))
    degree_var.append(float(np.var(degrees)))
    density.append(float(nx.density(g)))
    if not nx.is_connected(g):
        diameter.append(-1)
    else:
        diameter.append(int(nx.diameter(g)))
    N_entropy.append(float(np.sum((1/e)*degrees*np.log(degrees))))



import json
# 创建字典
info_dict = {'node_num':node_num,'edge_num':edge_num,'N_entropy': N_entropy, 'diameter': diameter, 'density':density,'average_degree':average_degree,'degree_var':degree_var}
# dumps 将数据转换成字符串
info_json = json.dumps(info_dict,sort_keys=False, indent=4, separators=(',', ': '))
# 显示数据类型
print(type(info_json))
f = open('ogbg-molhiv_data_info', 'w')
f.write(info_json)
