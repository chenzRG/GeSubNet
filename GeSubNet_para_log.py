import argparse
import os
import torch
import random
import numpy as np

num_gpus = torch.cuda.device_count()

if num_gpus >= 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
elif num_gpus >= 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
parser = argparse.ArgumentParser(description='G4G-NET')

# Main
parser.add_argument('--GNN', default='GCN', type=str, help='GCN  or NeoGNN')
parser.add_argument('--run_description', default=args.GNN, type=str, help='Run Description')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')

# Add encoder dimension argument
parser.add_argument('--enc_dim', default=64, type=int, help='Encoder dimension')
# Add encoder layer 1 dimension argument
parser.add_argument('--enc_lay1', default=256, type=int, help='Encoder layer 1 dimension')
# Add number of embeddings argument
parser.add_argument('--num_embeddings', default=32, type=int, help='Number of embeddings')
# Add embedding dimension argument
parser.add_argument('--embedding_dim', default=32, type=int, help='Embedding dimension')
# Add commitment cost argument
parser.add_argument('--commitment_cost', default=1, type=float, help='Commitment cost')
# Add dropout argument
parser.add_argument('--dropout', default=0, type=float, help='Dropout rate')

# Add in_channels argument
parser.add_argument('--in_channels', default=feature_num, type=int, help='Input channels')
# Add hidden_channels argument
parser.add_argument('--hidden_channels', default=64, type=int, help='Hidden channels')
# Add out_channels argument
parser.add_argument('--out_channels', default=32, type=int, help='Output channels')
# Add num_layers argument
parser.add_argument('--num_layers', default=3, type=int, help='Number of layers')
# Add dropout argument
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')
# Add f_edge_dim argument
parser.add_argument('--f_edge_dim', default=8, type=int, help='Edge feature dimension')
# Add f_node_dim argument
parser.add_argument('--f_node_dim', default=128, type=int, help='Node feature dimension')
# Add g_phi_dim argument
parser.add_argument('--g_phi_dim', default=128, type=int, help='g_phi dimension')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# optimization
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu',type =bool, help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
