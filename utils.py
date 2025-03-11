import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

@torch.no_grad()
def decode_graph(model, data, A, threshold=0.8):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index, A).view(-1).sigmoid()
    out = (out >= threshold).to(torch.float32)  
    selected_edges = data.edge_label_index[:, out == 1]
    build_graph = Data(x=data.x, edge_index=selected_edges)
    return build_graph

def convert_to_networkx(graph, n_sample=None):
    g = to_networkx(graph, node_attrs=["x"])
    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
    return g

def plot_graphs_in_subplots_with_similarity(g1, g2, log_dir, type, min_distance=0.5):
    num_edges_g1 = g1.number_of_edges()
    num_edges_g2 = g2.number_of_edges()
    num_shared_edges = len(set(g1.edges) & set(g2.edges))
    
    jaccard_similarity = num_shared_edges / (num_edges_g1 + num_edges_g2 - num_shared_edges)
   
    layout1 = nx.spring_layout(g1, seed=42, k=min_distance)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    nx.draw(g1, layout1, node_size=10, with_labels=False, arrows=False, ax=axes[0])
    axes[0].set_title("Graph 1")

    nx.draw(g2, layout1, node_size=10, with_labels=False, arrows=False, ax=axes[1])
    axes[1].set_title("Graph 2")

    plt.figtext(0.5, 0.02, f"Jaccard Similarity: {jaccard_similarity:.2f}", ha="center", size=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, type +"_graph.png"))
    plt.close('all')
    
def plot_embedding(GNN_model, used_graph, input_parts_dict, DGM_parts_dict, part, step_num, log_dir):
    GNN_model.eval()  
    with torch.no_grad():
        z = GNN_model.encode(used_graph.x, used_graph.edge_index)
    GNN_z = z.cpu().numpy()
    print('GNN_z:', GNN_z.shape)

    result = np.dot(GNN_z, DGM_parts_dict[part])

    input_normalized = (input_parts_dict[part] - input_parts_dict[part].min()) / (input_parts_dict[part].max() - input_parts_dict[part].min())
    result_normalized = (result - result.min()) / (result.max() - result.min())

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    im1 = axes[0].imshow(input_normalized, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title(f"Part {part} (Input)")

    im2 = axes[1].imshow(result_normalized, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(f"Part {part} (MM Reconstruct)")

    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar2 = fig.colorbar(im2, ax=axes[1])

    file_name = f"matrix_step_{step_num}.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, file_name))
    plt.close('all')