# General Imports
import os
import sys
import json
import numpy as np
import multiprocessing

# Networkx
import networkx as nx

# PyTorch
import torch

# Ours
import dissubformer_input_config as config



def get_pairwise_Euc_distances(node_id, node_embeddings):
    
    """
    Precompute the pairwise Euclidean distances between a specific node and all nodes in the graph based on their embeddings.
    
    Args:
        - node_id (int): The ID of the node to compute distances for.
        - node_embeddings (Tensor): Tensor containing embeddings of all nodes. 

    Returns:
        - node_Euc_distances: Distances from the given node to all other nodes.
    """
    
    curr_node_embedding = node_embeddings[node_id].unsqueeze(0)
    node_Euc_distances = torch.cdist(curr_node_embedding, node_embeddings, p=2).squeeze(0).cpu().detach().numpy()

    return node_Euc_distances


    
def precompute_Euc_distances_for_node(args):
    
    node_id, node_embeddings = args
    
    return get_pairwise_Euc_distances(node_id, node_embeddings)



def precompute_graph_metrics(node_embeddings, graph, node_ids, num_processes):
    
    AP_sampling_sim_dir = config.DATA_RESULTS_DIR / 'Data' / 'AP_sampling_similarities'
    if not os.path.exists(AP_sampling_sim_dir):
        os.makedirs(AP_sampling_sim_dir)

    Head_attention_sim_dir = config.DATA_RESULTS_DIR / 'Data' / 'Head_attention_similarities'
    if not os.path.exists(Head_attention_sim_dir):
        os.makedirs(Head_attention_sim_dir)
        
    # Process for precomputing ego graphs
    if config.PRECOMPUTE_EGO_GRAPHS:
        if not (config.DATA_RESULTS_DIR / 'Data' / 'Ego_graphs.txt').exists() or config.OVERRIDE:
            print("Precomputing ego graph for each node...")
            ego_graph_dict = {}
            for node in graph.nodes():
                ego_graph = nx.ego_graph(graph, node, radius=1)
                neighbors = sorted([int(neighbor) for neighbor in ego_graph.nodes()])
                ego_graph_dict[node] = neighbors
            ordered_ego_graph_dict = {node: ego_graph_dict.get(node, []) for node in node_ids}
            with open(str(config.DATA_RESULTS_DIR / 'Data' / 'Ego_graphs.txt'), 'w') as ego_file:
                json.dump(ordered_ego_graph_dict, ego_file)
            print("Ego graphs dictionary saved.")
        else:
            print("Ego graphs dictionary already exist. Loading from file.")
            
    # Process for precomputing Euclidean distances
    if config.PRECOMPUTE_EUCLIDEAN_DISTANCES:  
        if not (config.DATA_RESULTS_DIR / 'Data' / 'Euclidean_distances_matrix.npy').exists() or config.OVERRIDE:
            print("Precomputing Euclidean distances...")
            args_list = [(node_id, node_embeddings) for node_id in node_ids]
            with multiprocessing.Pool(processes=num_processes) as pool:
                all_node_Euc_distances = pool.map(precompute_Euc_distances_for_node, args_list)
            Euc_distances_matrix = np.stack(all_node_Euc_distances)  
            np.save(str(config.DATA_RESULTS_DIR / 'Data' / 'Euclidean_distances_matrix.npy'), Euc_distances_matrix)
            print("Euclidean distances matrix saved.")
        else:
            print("Euclidean distances matrix already exists. Loading from file.")

    # Process for precomputing shortest paths and intermediate nodes between all node pairs in the graph
    if config.PRECOMPUTE_SHORTEST_PATHS or config.PRECOMPUTE_INTERMEDIATE_NODES:
        if (not (config.DATA_RESULTS_DIR / 'Data' / 'Shortest_paths_matrix.npy').exists()\
            or not (config.DATA_RESULTS_DIR / 'Data' / 'Intermediate_nodes_matrix.npy').exists() or config.OVERRIDE):
                print("Precomputing pairwise shortest path lengths and the number of intermediate nodes between all node pairs in the graph...")
                num_nodes = len(node_ids)
                shortest_paths_matrix = np.full((num_nodes, num_nodes), np.inf) 
                np.fill_diagonal(shortest_paths_matrix, 0)
                intermediate_nodes_matrix = np.full((num_nodes, num_nodes), np.inf)
                np.fill_diagonal(intermediate_nodes_matrix, -1)
                shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
                
                for node_a, lengths_dict in shortest_path_lengths.items():
                    for node_b, shortest_path_length in lengths_dict.items():
                        i, j = node_ids.index(node_a), node_ids.index(node_b)
                        shortest_paths_matrix[i, j] = shortest_path_length
                        intermediate_nodes_matrix[i, j] = shortest_path_length - 1
                        
                if config.PRECOMPUTE_SHORTEST_PATHS or config.OVERRIDE:
                    np.save(str(config.DATA_RESULTS_DIR / 'Data' / 'Shortest_paths_matrix.npy'), shortest_paths_matrix)
                    print("Shortest paths matrix saved.")
                    
                if config.PRECOMPUTE_INTERMEDIATE_NODES or config.OVERRIDE:
                    np.save(str(config.DATA_RESULTS_DIR / 'Data' / 'Intermediate_nodes_matrix.npy'), intermediate_nodes_matrix)
                    print("Matrix of the number of intermediate nodes saved.")
        else:
            print("Matrices for shortest paths and the number of intermediate nodes already exist. Loading from file.")

    # Process for precomputing the adjacency matrix for the graph
    if config.PRECOMPUTE_ADJ_MATRIX:
        if not (config.DATA_RESULTS_DIR / 'Data' / 'ADJ_matrix.npy').exists() or config.OVERRIDE:
            print("Precomputing adjacency matrix ...")
            adjacency_matrix = nx.to_numpy_array(graph, nodelist=node_ids, dtype=int)
            np.fill_diagonal(adjacency_matrix, 1)
            np.save(str(config.DATA_RESULTS_DIR / 'Data' / 'ADJ_matrix.npy'), adjacency_matrix)
            print("Adjacency matrix saved.")    
        else:
            print("Adjacency matrix already exists. Loading from file.")

     # Process for precomputing node degrees
    if config.PRECOMPUTE_NODE_DEGREES:
        if not (config.DATA_RESULTS_DIR / 'Data' / 'Node_degrees.txt').exists() or config.OVERRIDE:
            print("Precomputing node degrees...")
            node_degrees = dict(graph.degree())
            ordered_node_degrees = {node: node_degrees.get(node, 0) for node in node_ids}
            with open(str(config.DATA_RESULTS_DIR / 'Data' / 'Node_degrees.txt'), 'w') as deg_file:
                json.dump(ordered_node_degrees, deg_file)
            print("Node degrees dictionary saved.")
        else:
            print("Node degrees file already exists. Loading from file.")


            
def main():
    graph = nx.read_edgelist(str(config.DATA_RESULTS_DIR / 'Data' / 'PPI.txt'))
    relabel_graph = nx.relabel_nodes(graph, lambda x: int(x))
    node_ids = sorted(relabel_graph.nodes())
    node_embeddings = torch.load(str(config.DATA_RESULTS_DIR / 'Data' / 'combined_PPI_GO_FS_embeddings.pth'))
    precompute_graph_metrics(node_embeddings, relabel_graph, node_ids, config.N_PROCESSSES)


     
if __name__ == '__main__':
    main()
