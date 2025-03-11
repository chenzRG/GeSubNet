import os
import pickle
import numpy as np
import pandas as pd
import math
from itertools import chain
import itertools
import copy
import random
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.nn.utils.prune as prune
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import scipy.sparse as ssp
from scipy import signal
from scipy.fft import fftshift

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
import logging
from functools import partial
from collections import OrderedDict
import seaborn as sns
import matplotlib as mpl
import time
from scipy.signal import savgol_filter
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

import torch_geometric.transforms as T
from torch_geometric.data import Data

import torch
import random
import tqdm
import pandas as pd
import os

import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse

from dataloader import Mydatasets
from model.SELM_NN import SELM
from model.GCN import GCN
from model.NeoGNN import NeoGNN
from model.GAT import GAT
from trainer import train_SELM, train_link_predictor, eval_link_predictor, minimize_MSE
from utils import decode_graph, convert_to_networkx, plot_graphs_in_subplots_with_similarity, plot_embedding

parser = argparse.ArgumentParser()
parser.add_argument('--GNN', default='GCN', type=str,
                    help='GCN  or NeoGNN')
parser.add_argument('--run_description', default='default', type=str,
                    help='Run Description')
parser.add_argument('--device', default='default', type=str,
                    help='cpu or cuda')
args = parser.parse_args()

if args.run_description == 'default':
    args.run_description = args.GNN
if args.device == 'default':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#log_dir = os.path.join("log", args.run_description)
log_dir = os.path.expanduser("log/"+args.run_description+"/")


data_dir = ("data/")
dat = pd.read_csv(os.path.join(data_dir,'BRCA_noded_mRNA.csv'))
dat = dat.drop(dat.columns[[0]], axis=1)
dat = dat.T
feature_num = dat.shape[1]
data_num = dat.shape[0]
dat = dat.values.reshape(-1,feature_num)
print(dat.shape)

index = pd.read_csv(os.path.join(data_dir, 'BRCA_label_num.csv'), header=0)
index = index[0 :data_num].astype(int)
print(index[0 :data_num].apply(pd.value_counts))
label = index.values.tolist()  #list
cls_num = len(set([i for i in index['Label']]))
print(cls_num,'Subtypes,',len(label),'Samples')

divided_datasets = []
for label_value in set(index['Label']):
    filtered_data = dat[index['Label'] == label_value]
    divided_datasets.append(filtered_data)

for i, dataset in enumerate(divided_datasets):
    print(f"Dataset {i+1} (Label {i}): Shape {dataset.shape}")

train_data, test_data, train_label, test_label = train_test_split(dat, label, test_size = 0.2,random_state = 1)
print('train data:',len(train_data))
print('test data:',len(test_data))
train_data_set = Mydatasets(data1 = train_data, label = train_label)
test_data_set = Mydatasets(data1 = test_data, label = test_label)
train_dataloader = torch.utils.data.DataLoader(train_data_set, batch_size = 16, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data_set, batch_size = 16, shuffle=False)

SELM_model = SELM(input_size = feature_num, 
            enc_dim = 64,
            enc_lay1 = 256,
            num_embeddings = 16,  
            embedding_dim = 32,   
            commitment_cost = 1,
            dropout = 0
           ).to(DEVICE)


SELM_acc = train_SELM(SELM_model, train_dataloader, test_dataloader, device, log_dir, epoch = 100)
SELM_model.load_state_dict(torch.load(os.path.join(log_dir,'SELM.pt')))

en_lat = []
en_labels = []
data_set = Mydatasets(data1 = dat, label = label)
data_set = torch.utils.data.DataLoader(data_set, batch_size = 256, shuffle=True)

for i in range(len(dat)):
    en_data = data_set.dataset[i][0]
    en_label = data_set.dataset[i][1]
    z,_,mlp = SELM_model(en_data.view(1, 1, feature_num).float().to(device))
    en_lat.append(z.cpu().detach().numpy())
    en_labels.append(en_label.cpu().detach().numpy())

encode_out = np.array(en_lat)
encode_out = encode_out.reshape(len(dat), -1)
encode_labels = np.array(en_labels)
DGM_z = encode_out

print('input:', dat.shape)
print('encode_out:', encode_out.shape)

expression_matrix = pd.read_csv(os.path.join(data_dir,'BRCA_noded_mRNA.csv'), index_col=0)
gene_gene_edges = pd.read_csv(os.path.join(data_dir,'gene_gene_edges.csv'))

x = torch.tensor(expression_matrix.values, dtype=torch.float)
edge_indices = torch.tensor(gene_gene_edges.values.T, dtype=torch.long)

edge_weight = torch.ones(edge_indices.size(1), dtype=float)

#adjacency matrix
A = ssp.csr_matrix((edge_weight, (edge_indices[0], edge_indices[1])), 
                    shape=(dat.shape[0], dat.shape[0]))


original_graph = Data(x=x, edge_index=edge_indices)

graphs = []
for i, dataset in enumerate(divided_datasets):
    x = torch.tensor(dataset.T, dtype=torch.float)
    edge_indices = torch.tensor(gene_gene_edges.values.T, dtype=torch.long)

    graph = Data(x=x, edge_index=edge_indices)
    graphs.append(graph)
    
    globals()[f"graph{i}"] = graph
    print(f"Graph {i+1} (Label {i}):")
    print(graph)
print(graphs)

parent_dir = log_dir
link_prediction = []

for part in range(cls_num):
    log_dir = os.path.join(parent_dir, "label"+str(part))
    #random sampling graph for first step GNN training
    devided_graph = graphs[part]
    sample_n = devided_graph.x.shape[1]

    original_shape = original_graph.x.shape[1]
    selected_indices = random.sample(range(original_shape), sample_n)
    new_x = original_graph.x[:,selected_indices]

    rs_graph = Data(x=new_x, edge_index=edge_indices).to(device)

    split = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.05,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
    )

    train_data, val_data, test_data = split(rs_graph)
    #train_data, val_data, test_data = split(devided_graph)
    feature_num = train_data.x.shape[1]
    print('node_feature_num:',feature_num)

    print('first_step_random_sampled_graph:',rs_graph)
    print('second_step_devided_graph:',devided_graph,'\n')
    print('RandomLinkSplit:')
    print(train_data)
    print(val_data)
    print(test_data)

    print(device)

    if args.GNN == 'GCN':
        GNN_model = GCN(feature_num, 64, 32).to(device)
    elif args.GNN == 'NeoGNN':
        GNN_model = NeoGNN(feature_num, 64, 32).to(device)
    elif args.GNN == 'GAT':
        GNN_model = GAT(feature_num, 64, 32).to(device)


    train_link_predictor(GNN_model, train_data.to(device), val_data.to(device), A, log_dir)
    GNN_model.load_state_dict(torch.load(os.path.join(log_dir, "saved_models",'new_test_disc_1')))
    test_auc = eval_link_predictor(GNN_model, test_data.to(device), A)
    print(f"Link Predictor Test: {test_auc:.3f}")
    link_prediction.append(test_auc)
    print(f"Link Predictor Test: {test_auc:.3f}")

data = {
    'Metric': ['SELM_auc'] + [f'link_acc_{i}' for i in range(len(link_prediction))],
    'Value': [SELM_acc] + link_prediction
}
df = pd.DataFrame(data)
df.to_excel(os.path.join(parent_dir, "result.xlsx"), index=False, engine='openpyxl')        
