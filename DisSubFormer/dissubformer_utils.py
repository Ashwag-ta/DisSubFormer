# General Imports
import sys
import numpy as np

# Sci-kit Learn 
from sklearn.metrics import f1_score, accuracy_score, average_precision_score

# PyTorch and PyTorch Geometric
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

# Networkx
import networkx as nx

# Ours
sys.path.insert(0, '..') # add config to path
import main_config as config



def process_edges(pos_edges, neg_edges, sample_fraction):
    
    """
    Sample a fraction of negative edges, concatenates them with positive edges, and generates corresponding binary edge labels.

    Args:
        - pos_edges (Tensor): Tensor of positive edges.
        - neg_edges (Tensor): Tensor of negative edges.
        - sample_fraction (float): Fraction of negative edges to sample.

    Returns:
        - combined_edges (Tensor): Tensor containing both positive and sampled negative edges.
        - edge_labels (Tensor): Tensor where 1 indicates a positive edge and 0 indicates a sampled negative edge.
    """
  
    # Sample a fraction of the negative edges
    num_neg_samples = int(neg_edges.size(0) * sample_fraction)
    sampled_neg_edges = neg_edges[torch.randperm(neg_edges.size(0))[:num_neg_samples]]
  

    # Concatenate positive and sampled negative edges
    combined_edges = torch.cat([pos_edges, sampled_neg_edges], dim=0).t()
    
    # Generate labels: 1 for positive, 0 for negative
    edge_labels = torch.cat([
        torch.ones(pos_edges.size(0), dtype=torch.long),  # Positive labels
        torch.zeros(sampled_neg_edges.size(0), dtype=torch.long)], dim=0)
    
    # Shuffle the edges and corresponding labels
    perm = torch.randperm(combined_edges.size(1))  
    combined_edges, edge_labels = combined_edges[:, perm], edge_labels[perm]
    
    return combined_edges, edge_labels
    


def read_DDI_graph(subgraphs_file, DDI_edge_list_file, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10):

    """
    Read a DDI graph, constructs a NetworkX and PyTorch Geometric representation, and splits the graph edges into train, validation, and test sets.

    Args:
        - subgraphs_file (str): Path to file containing subgraph node IDs.
        - DDI_edge_list_file (str): Path to file containing the DDI edge list.
        - train_ratio (float): Proportion of edges used for train.
        - val_ratio (float): Proportion of edges used for validation.
        - test_ratio (float): Proportion of edges used for testing.

    Returns:
        - train_DDI_edges / val_DDI_edges / test_DDI_edges (Tensor): Combined edges (positive + sampled negative) for each split.
        - train_DDI_edges_label / val_DDI_edges_label / test_DDI_edges_label (Tensor): Corresponding binary edge labels for each split (1 = positive, 0 = negative).
    """

    # Build a NetworkX graph 
    DDI_graph = nx.Graph()
    with open(subgraphs_file) as sub_f:
        for line in sub_f:
            subgraph_id = int(line.split("\t")[1].strip())
            DDI_graph.add_node(subgraph_id)
    DDI_graph.add_edges_from(nx.read_edgelist(DDI_edge_list_file, nodetype=int).edges())

    # Convert the NetworkX graph to a PyTorch Geometric Data object
    edge_index = torch.tensor(list(DDI_graph.edges()), dtype=torch.long).t().contiguous()
    num_nodes = DDI_graph.number_of_nodes()
    edge_label = torch.ones(edge_index.size(1), dtype=torch.long)  # Label all edges as positive
    data = Data(edge_index=edge_index, num_nodes=num_nodes, edge_label=edge_label)

    # Split into train, validation, and test using RandomLinkSplit
    transform = RandomLinkSplit(is_undirected=True, split_labels=True, num_val=val_ratio, num_test=test_ratio)
    train_data, val_data, test_data = transform(data)

    # Extract positive and negative edges from each split
    train_pos_edges = train_data.pos_edge_label_index.t()
    train_neg_edges = train_data.neg_edge_label_index.t()
    val_pos_edges = val_data.pos_edge_label_index.t()
    val_neg_edges = val_data.neg_edge_label_index.t()
    test_pos_edges = test_data.pos_edge_label_index.t()
    test_neg_edges = test_data.neg_edge_label_index.t()

    # Sample negative edges and generate labels
    train_DDI_edges, train_DDI_edges_label = process_edges(train_pos_edges, train_neg_edges, sample_fraction=0.25)
    val_DDI_edges, val_DDI_edges_label = process_edges(val_pos_edges, val_neg_edges, sample_fraction=0.25)
    test_DDI_edges, test_DDI_edges_label = process_edges(test_pos_edges, test_neg_edges, sample_fraction=0.25)
   
    return train_DDI_edges, train_DDI_edges_label, val_DDI_edges, val_DDI_edges_label, test_DDI_edges, test_DDI_edges_label



def assign_subgraphs_to_splits(subgraphs_file, train_edges, val_edges, test_edges):
    
    """
    Assign subgraphs to the train, validation, and test sets based on their associated splitting edges.

    Returns:
        - all_subgraphs (list): All parsed subgraphs from the file.
        - train_subgraphs / val_subgraphs / test_subgraphs (list): Subgraphs assigned to the train, validation, and test sets, respectively.
        - train_subgraphs_dict / val_subgraphs_dict / test_subgraphs_dict (dict): Mapping from subgraph ID to subgraphs for each split.
        - train_subgraphs_indices / val_subgraphs_indices / test_subgraphs_indices (list): Indices of subgraphs assigned to each split.

    """

    all_subgraphs = []
    train_subgraphs = []
    val_subgraphs = []
    test_subgraphs = []

    train_subgraphs_dict = {}
    val_subgraphs_dict = {}
    test_subgraphs_dict = {}

    train_subgraphs_indices = []
    val_subgraphs_indices = []
    test_subgraphs_indices = []

    # Read the subgraph file and assign subgraphs to the appropriate split
    with open(subgraphs_file) as sub_f:
        for idx, line in enumerate(sub_f):
            subgraph_nodes = line.split("\t")[0].split("-")
            subgraph_nodes = [int(node) for node in subgraph_nodes]
            subgraph_id = int(line.split("\t")[1].strip())
            all_subgraphs.append(subgraph_nodes)

            # Assign to train set
            train_subgraphs.append(subgraph_nodes)
            train_subgraphs_indices.append(idx)
            if subgraph_id not in train_subgraphs_dict:
                train_subgraphs_dict[subgraph_id] = []
            train_subgraphs_dict[subgraph_id].append(subgraph_nodes)

            # Assign to validation set
            val_subgraphs.append(subgraph_nodes)
            val_subgraphs_indices.append(idx)
            if subgraph_id not in val_subgraphs_dict:
                val_subgraphs_dict[subgraph_id] = []
            val_subgraphs_dict[subgraph_id].append(subgraph_nodes)

            # Assign to test set
            test_subgraphs.append(subgraph_nodes)
            test_subgraphs_indices.append(idx)
            if subgraph_id not in test_subgraphs_dict:
                test_subgraphs_dict[subgraph_id] = []
            test_subgraphs_dict[subgraph_id].append(subgraph_nodes)
            
    return (
        all_subgraphs, train_subgraphs, val_subgraphs, test_subgraphs, 
        train_subgraphs_dict, val_subgraphs_dict, test_subgraphs_dict, 
        train_subgraphs_indices, val_subgraphs_indices, test_subgraphs_indices
           )



def get_component_global_neighborhood_set(PPI_graph, component, radius, ego_graph_dict=None):

    """
    Extract the set of nodes in the k-hop neighborhood surrounding a given component, excluding the nodes within the component itself.
    
    Args:
        - PPI_graph (Graph object): A NetworkX graph representing PPI graph.
        - component (Tensor): 1D tensor of node IDs in the component. 
        - radius (int): Number of hops for the ego network (k-hop neighborhood).
        - ego_graph_dict (dict, optional): Precomputed dictionary mapping node IDs to their ego neighborhoods.
        
    Returns:
        - global_nodes (set): Set of nodes in the k-hop neighborhood that are not part of the component.
    """

    if type(component) is torch.Tensor: 
        component_inds_non_neg = (component!=config.PAD_VALUE).nonzero().view(-1)
        component_set = {int(n) for n in component[component_inds_non_neg]}
    else:
        component_set = set(component)

    # Collect the union of all k-hop neighborhoods for nodes in the component
    neighborhood = set()
    for node in component_set: 
        if ego_graph_dict == None: 
            # if it hasn't already been computed, compute ego graph (centered at node with given radius)
            ego_g = nx.ego_graph(PPI_graph, node, radius = radius).nodes()
        else:
            # Use precomputed ego graph (adjusting for 0-indexing)
            ego_g = ego_graph_dict[node - 1] 

        neighborhood = neighborhood.union(set(ego_g))

    # Exclude original component nodes to get the global nodes
    global_nodes = neighborhood.difference(component_set)

    return global_nodes



def get_global_nodes(PPI_graph, subgraph, all_subgraphs_dis_similarities):

    """
    Identify global candidate nodes based on similarity to subgraphs.

    Returns:
        - in_glob_nodes (list): Subgraph nodes that have at least one edge to a node outside the subgraph.
        - non_subgraph_nodes (array): Nodes in the global graph that are not part of the subgraph.
        - all_valid_nodes (list): Combined set of in_glob_nodes and non_subgraph_nodes — the full set of global candidate nodes. 
    """

    # Get all nodes in the PPI graph that are not part of the subgraph
    non_subgraph_nodes = np.array(list(set(PPI_graph.nodes()).difference(set(subgraph.nodes()))))
    subgraph_nodes = np.array(list(subgraph.nodes()))
    
    # Get adjacency matrix for the PPI graph
    adj_PPI_graph = nx.adjacency_matrix(PPI_graph).todense()

    # Subset adjacency matrix to extract edges between subgraph and non-subgraph nodes
    subgraph_to_global = adj_PPI_graph[np.ix_(subgraph_nodes - 1, non_subgraph_nodes - 1)]

    # Identify subgraph nodes that are connected to at least one non-subgraph node
    has_global_edge = (np.sum(subgraph_to_global, axis=1) > 0).flatten()
    in_glob_nodes = list(subgraph_nodes[has_global_edge])
    
    # Combine subgraph nodes connected to the global graph and non-subgraph nodes
    all_valid_nodes = list(set(in_glob_nodes).union(set(non_subgraph_nodes)))
    
    nodes_type = {
        "non_subgraph_nodes": non_subgraph_nodes,
        "in_glob_nodes": in_glob_nodes,
        "all_valid_nodes": all_valid_nodes,
                }

    # Sort each group of nodes based on their total similarity to all subgraphs
    sorted_nodes = {}
    for key, nodes in nodes_type.items():
        all_nodes_total_sim = np.zeros(len(nodes))
        for i, node in enumerate(nodes):
            # Compute the total similarity of this node with all subgraphs
            curr_node_sim = all_subgraphs_dis_similarities[:, node - 1].detach().cpu().numpy()  
            all_nodes_total_sim[i] = np.sum(curr_node_sim)

        # Sort nodes in descending order based on their total similarity with all subgraphs
        sorted_indices = np.argsort(-all_nodes_total_sim)  
        sorted_nodes[key] = [nodes[i] for i in sorted_indices]

    sorted_non_subgraph_nodes = sorted_nodes["non_subgraph_nodes"]
    sorted_in_glob_nodes = sorted_nodes["in_glob_nodes"]
    sorted_all_valid_nodes = sorted_nodes["all_valid_nodes"]
  
    return sorted_in_glob_nodes, sorted_non_subgraph_nodes, sorted_all_valid_nodes



def compute_accuracy(prob, DDI_edge_labels):
    
    """
    Compute the accuracy for edge prediction.
    """
    
    preds = (prob >= 0.5).float()  # Threshold at 0.5
    
    accuracy = accuracy_score(DDI_edge_labels.cpu().numpy(), preds.cpu().numpy())
    accuracy = torch.tensor([accuracy])
    
    return accuracy



def compute_f1_ap_metrics(prob, DDI_edge_labels):
    
    """
    Compute the F1 score and average precision (AP) for edge prediction.
    """
    preds = (prob >= 0.5).float()  # Threshold at 0.5
    preds = preds.cpu().numpy()
    
    DDI_edge_labels = DDI_edge_labels.cpu().numpy()
    
    # Compute F1 score
    f1 = f1_score(DDI_edge_labels, preds, average='binary', zero_division=0)
    f1 = torch.tensor([f1])
    
    # Compute average precision (AP)
    ap = average_precision_score(DDI_edge_labels, prob.cpu().numpy()) if DDI_edge_labels.sum() > 0 else 0.0
    ap = torch.tensor([ap])
    
    return f1, ap

