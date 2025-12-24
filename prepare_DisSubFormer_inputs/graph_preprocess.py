# General Imports

import sys
import numpy as np
import pandas as pd

# PyTorch and PyTorch Geometric
import torch
from torch_geometric.data import Data

# Networkx
import networkx as nx

# Ours
sys.path.insert(0, '../')
import dissubformer_input_config as config



def create_dataset_objects(graph, node_embeddings, edge_weights=None, split=True ):
    
    """
    Create a PyTorch Geometric Data object from a PPI or GO_FS graph.

    Args:
        - graph (Graph object): A NetworkX graph representing PPI or GO_FS graph.
        - node_embeddings (Tensor): Tensor containing embeddings for each node.
        - edge_weights (Tensor, Optional): Tensor containing weights for each edge.
        - split (bool, Optional): Whether to split the edges into train, val, and test sets.

    Returns:
        - new_graph (Data object): A PyG Data object representing the input graph.
    """

    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    y = torch.ones(edge_index.shape[1]) # labels
    num_classes = len(torch.unique(y))

    if split:
        # Randomly split edges into train, validation, and test sets (80/10/10 split)
        split_idx = np.arange(len(y))
        np.random.shuffle(split_idx)
        train_idx = split_idx[: 8 * len(split_idx) // 10]
        val_idx = split_idx[8 * len(split_idx) // 10 : 9 * len(split_idx) // 10]
        test_idx = split_idx[9 * len(split_idx) // 10 :]

        # Create boolean masks for each set
        # Train set
        train_mask = torch.zeros(len(y), dtype=torch.bool)
        train_mask[train_idx] = 1

        # Val set
        val_mask = torch.zeros(len(y), dtype=torch.bool)
        val_mask[val_idx] = 1

        # Test set
        test_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask[test_idx] = 1 
    else:
        train_mask = val_mask = test_mask = None

    new_graph = Data(x=node_embeddings, y=y, num_classes=num_classes, edge_index=edge_index, 
                     edge_attr=edge_weights, train_mask=train_mask, val_mask=val_mask, 
                     test_mask=test_mask)

    return new_graph



def read_dataset(graph_edge_list, dataset_name, node_embeddings=None):
    
    """
    Load either the PPI or GO_FS graph and generate a PyTorch Geometric Data object.

    Args:
        - graph_edge_list (str): Path to the edge list file of the graph (PPI or GO_FS).
        - dataset_name (str): Dataset identifier (PPI or GO_FS).
        - node_embeddings (str, optional): Path to the node embeddings file (only for GO_FS).

    Returns:
        - all_data (Data object): A PyG Data object representing the input graph.
    """
    
    global PPI_nodes

    if dataset_name == "PPI":
        nx_Graph = nx.read_edgelist(graph_edge_list, nodetype = int)
        node_embeddings = torch.eye(len(nx_Graph.nodes), dtype=torch.float)
        all_data = create_dataset_objects(nx_Graph, node_embeddings)
        PPI_nodes = all_data.num_nodes


    elif dataset_name == "GO_FS":
        nx_Graph = nx.read_weighted_edgelist(graph_edge_list, nodetype=int)
        GO_FS_node_embeddings = pd.read_csv(node_embeddings, header=None, index_col=0)
        GO_FS_node_embeddings.index = GO_FS_node_embeddings.index.astype(int)

        # Ensure all protein nodes present in PPI are included in the GO_FS graph
        extra_protein_nodes = set(range(min(nx_Graph.nodes()), PPI_nodes)) - set(nx_Graph.nodes())
        for node in extra_protein_nodes:
            nx_Graph.add_node(node)

        # Assign node embeddings, defaulting to zero vector if missing
        for node in sorted(nx_Graph.nodes()):
            if node in GO_FS_node_embeddings.index:
                nx_Graph.nodes[node]['embeddings'] = np.array(
                    [float(x) for x in GO_FS_node_embeddings.loc[node].values[0].split(',')], dtype=float)
            else:
                nx_Graph.nodes[node]['embeddings'] = np.zeros(len(GO_FS_node_embeddings.iloc[0].values[0].split(',')), dtype=float)
                
        node_embeddings = torch.tensor(np.array([nx_Graph.nodes[node]['embeddings'] for node in sorted(nx_Graph.nodes())]), dtype=torch.float)

        edge_weights = torch.tensor([edge[2]['weight'] for edge in nx_Graph.edges(data=True)], dtype=torch.float).reshape(-1, 1)

        all_data = create_dataset_objects(nx_Graph, node_embeddings, edge_weights )

    return all_data



def set_batch_data(batch, all_data):
    
    """
    Assign node embeddings, edge attributes, and masks from the full graph to the given mini-batch.

    Args:
        - batch (Data object): The PyG mini-batch Data object.
        - all_data (Data object): The PyG full graph Data object containing node embeddings, edge attributes, and masks.

    Returns:
        - Data: The updated PyG mini-batch Data object with assigned embeddings, edge attributes, and labels.
    """
        
    batch.x = all_data.x[batch.n_id]
    batch.train_mask = all_data.train_mask[batch.e_id] 
    batch.val_mask = all_data.val_mask[batch.e_id]
    batch.y = torch.ones(len(batch.e_id))
    batch.edge_attr = all_data.edge_attr[batch.e_id] if all_data.edge_attr is not None else None

    assert torch.all((batch.train_mask & batch.val_mask) == 0) #Train and val masks overlap

    return batch



