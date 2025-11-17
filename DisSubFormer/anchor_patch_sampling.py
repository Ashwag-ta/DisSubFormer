# General Imports
import sys
import numpy as np
from collections import defaultdict

# PyTorch 
import torch

# Networkx
import networkx as nx

# Ours
sys.path.insert(0, '..')
import main_config as config
import dissubformer_utils



def hawkes_process_sampling_seq(PPI_graph, start_node, curr_node_anchor_patch, in_global_nodes,
                                non_subgraph_nodes, all_valid_nodes, inside, hyperparameters, eucli_dis):
    
    """
    Generate node sequences for structure anchor patches using a Hawkes process.
    Sequences are sampled either locally within the anchor patch or from global nodes.

    Returns:
        - curr_patch_sequences (List[Tensor]): List of node sequences.
    """
    
    curr_patch_sequences = []

    # Local structure anchor patches
    if inside:
        prev_node = start_node
        prev_node_neighbors = [n for n in curr_node_anchor_patch if n != prev_node]
        
        if len(prev_node_neighbors) == 0:
             current_sequence = [prev_node] 
             curr_patch_sequences.append(torch.cat([
                torch.LongTensor(current_sequence),
                torch.LongTensor([config.PAD_VALUE] * (hyperparameters['hawkes_process_sequence_len'] - len(current_sequence)))
                ]))
        else:
             for initial_neighbor in prev_node_neighbors:
                if len(curr_patch_sequences) >= hyperparameters['n_hawkes_process_sequences']:
                    break
                
                current_node_neighbors = [n for n in prev_node_neighbors if n != initial_neighbor]
                current_sequence = [start_node, initial_neighbor]
                historical_nodes = [start_node]
                visited_nodes = set(current_sequence)
                
                while len(current_sequence) < hyperparameters['hawkes_process_sequence_len']:
                    current_node = current_sequence[-1]
                    if len(current_node_neighbors) == 0:
                        break
                    
                    next_node = select_next_node(current_node, current_node_neighbors, historical_nodes, eucli_dis, hyperparameters, PPI_graph)
                    current_sequence.append(next_node)
                    historical_nodes.append(current_node)
                    visited_nodes.add(next_node)
                    current_node_neighbors = [n for n in current_node_neighbors if n != next_node]  
                    
                padded_current_sequence = torch.cat([
                    torch.LongTensor(current_sequence),
                    torch.LongTensor([config.PAD_VALUE] * (hyperparameters['hawkes_process_sequence_len'] - len(current_sequence)))
                    ])
                curr_patch_sequences.append(padded_current_sequence)

    # Global structure anchor patches        
    else:
        for global_node in in_global_nodes:
            if len(curr_patch_sequences) >= hyperparameters['n_hawkes_process_sequences']:
                break
            
            current_sequence = [global_node]
            historical_nodes = [global_node]
            visited_nodes = set(current_sequence)

            while len(current_sequence) < hyperparameters['hawkes_process_sequence_len']:
                current_node = current_sequence[-1]
                current_node_neighbors = [n for n in list(PPI_graph.neighbors(current_node)) if n in all_valid_nodes and n not in visited_nodes]

                if len(current_node_neighbors) == 0:
                    break

                next_node = select_next_node(current_node, current_node_neighbors, historical_nodes, eucli_dis, hyperparameters, PPI_graph)
                current_sequence.append(next_node)
                if current_node not in historical_nodes:
                    historical_nodes.append(current_node)
                visited_nodes.add(next_node)

            padded_current_sequence = torch.cat([
                torch.LongTensor(current_sequence),
                torch.LongTensor([config.PAD_VALUE] * (hyperparameters['hawkes_process_sequence_len'] - len(current_sequence)))
                ])
            curr_patch_sequences.append(padded_current_sequence)

    return curr_patch_sequences



def select_next_node(current_node, neighbors, historical_nodes, eucli_dis, hyperparameters, PPI_graph):
    
    """
    Select the next node using a Hawkes process: combines base rate (distance) and decayed historical influence.

    Returns:
        - next_node (int): Selected node ID from the neighbors.
    """
    
    base_rates = []
    historical_influences = []

    for neighbor in neighbors:
        base_rate = -eucli_dis[current_node-1, neighbor-1] # NOTE: Subtract 1 because the node indices are 1-based
        base_rates.append(base_rate)

        dis_for_each_h_node = np.array([np.exp(-eucli_dis[h-1, neighbor-1]) for h in historical_nodes]) 
        dis_sum_across_all_h_nodes = np.sum(dis_for_each_h_node)
        
        if dis_sum_across_all_h_nodes == 0:
            weights = np.zeros(len(historical_nodes))
        else:
            weights = dis_for_each_h_node / dis_sum_across_all_h_nodes

        influence_sum = 0           
        for idx_h, h_node in enumerate(historical_nodes):
            alpha_h_nei = -eucli_dis[h_node - 1, neighbor - 1] * weights[idx_h] 
            time_diff = 0 if h_node == current_node else len(historical_nodes) - historical_nodes.index(h_node)
            decay = np.exp(-hyperparameters['hawkes_decay_rate'] * time_diff)
            influence_sum += alpha_h_nei * decay

        historical_influences.append(influence_sum)

    intensities = np.exp(np.array(base_rates) + np.array(historical_influences))
    probabilities = intensities / np.sum(intensities)

    next_node = neighbors[np.argmax(probabilities)]

    return next_node



def perform_hawkes_process(hyperparameters, PPI_graph, anchor_patch_ids, eucli_dis, all_subgraphs_dis_similarities, inside):
    
    """
    Perform the Hawkes process to all structure anchor patches, either locally or globally.
    
    Returns:
        - all_patch_hawkes_process_seq (Tensor): All sampled sequences of shape (n_ap, n_seq_per_ap, seq_len).
    """
    
    all_patches = []
    start_nodes = []
    all_patch_hawkes_process_seq = []

    for start_node, patch in anchor_patch_ids.items():
        all_patches.append(patch)
        start_nodes.append(start_node)

    all_anchor_patches = torch.stack(all_patches)  
    n_sampled_patches, max_patch_len = all_anchor_patches.shape

    for patch_idx, patch in enumerate(all_anchor_patches):
        curr_node_anchor_patch = np.array(patch[patch != config.PAD_VALUE])
        curr_node_anchor_patch_subgraph = PPI_graph.subgraph(curr_node_anchor_patch)

        if inside:
            in_global_nodes, non_subgraph_nodes, all_valid_nodes = None, None, None
        else:
            in_global_nodes, non_subgraph_nodes, all_valid_nodes = dissubformer_utils.get_global_nodes(
                PPI_graph, curr_node_anchor_patch_subgraph, all_subgraphs_dis_similarities)

        patch_sequences = hawkes_process_sampling_seq(
            PPI_graph, start_nodes[patch_idx], curr_node_anchor_patch,
            in_global_nodes, non_subgraph_nodes, all_valid_nodes, inside, hyperparameters, eucli_dis)

        while len(patch_sequences) < hyperparameters['n_hawkes_process_sequences']:
            patch_sequences.append(torch.LongTensor([config.PAD_VALUE] * hyperparameters['hawkes_process_sequence_len']))
                
        padded_curr_patch_sequences = torch.stack(patch_sequences)
        all_patch_hawkes_process_seq.append(padded_curr_patch_sequences)

    all_patch_hawkes_process_seq = torch.stack(all_patch_hawkes_process_seq).view(
        n_sampled_patches, hyperparameters['n_hawkes_process_sequences'], hyperparameters['hawkes_process_sequence_len'])

    return all_patch_hawkes_process_seq

    

def sample_position_anchor_patches(hyperparameters,  dis_similarities_subgraph , dis_similarities_cc , cc_ids , subgraphs ):
    
    """
    Sample local or global position-based anchor patches using node similarity scores.

    If 'subgraphs' is provided, samples top-weighted nodes from each subgraph (local patches).
    Otherwise, selects top nodes across the entire graph (global patches).

    Returns:
        - List of local patches: shape (n_subgraphs, n_patches)
        - Or list of global patches: shape (n_patches)
    """
    
    local_patches = []
    global_patches = []
    required_size = hyperparameters['n_anchor_patches_pos_in'] * (hyperparameters['n_layers'])

    # Sampling local anchor patches
    if subgraphs :
        for s, components in enumerate(cc_ids):  
            cc_weights = dis_similarities_cc[s] 
            subgraph_nodes = np.unique(np.array(components).flatten())  
            total_weights = np.sum(cc_weights.cpu().numpy()[:, subgraph_nodes - 1], axis=0)
            
            loc_highest_weight_node = np.argsort(-total_weights)[:hyperparameters['n_layers'] * hyperparameters['n_anchor_patches_pos_in']]
            if len(loc_highest_weight_node) < required_size:
                loc_highest_weight_node = np.pad(
                    loc_highest_weight_node, (0, required_size - len(loc_highest_weight_node)), constant_values=config.PAD_VALUE)

            local_patches.append([subgraph_nodes[i] + 1 if i != config.PAD_VALUE else config.PAD_VALUE for i in loc_highest_weight_node])

        return local_patches
    
    # Sampling global anchor patches                 
    else:
        subgraph_total_weights = np.zeros(dis_similarities_subgraph.shape[1])
        for subgraph in range(dis_similarities_subgraph.shape[0]):  
            subgraph_weights = dis_similarities_subgraph[subgraph, :].detach().cpu().numpy()
            subgraph_total_weights +=  subgraph_weights
        glob_highest_weight_nodes = np.argsort(-subgraph_total_weights)[:hyperparameters['n_layers'] * hyperparameters['n_anchor_patches_pos_out']]
        global_patches.extend(glob_highest_weight_nodes + 1)
        
        return global_patches


        
def sample_neighborhood_anchor_patch(hyperparameters, cc_ids, global_set, dis_similarities_cc, sample_inside=True):
            
    """
    Sample local or global neighborhood-based anchor patches using node similarity scores.

    For each connected component, selects top-scoring nodes either from:
        - within the component ('sample_inside=True'), or
        - global neighbors of the component ('sample_inside=False').

    Returns:
        - anchor_patches (Tensor): 
            shape (batch_size, max_n_cc, n_anchor_patches_N_in) if local.
            shape (batch_size, max_n_cc, n_anchor_patches_N_out) if global.
    """
    
    batch_sz, max_n_cc, _ = cc_ids.shape
    all_samples = []
    
    # Sampling local anchor patches
    if sample_inside:
        for subgraph_idx, components in enumerate(cc_ids): 
            subgraph_samples = []
            for component in range(max_n_cc):
                component_nodes = cc_ids[subgraph_idx, component]  
                non_padded_component_nodes = component_nodes[component_nodes != 0]
                
                if len(non_padded_component_nodes) > 0:
                    node_weights = dis_similarities_cc[subgraph_idx, component, non_padded_component_nodes - 1]  
                    sorted_indices = np.argsort(-node_weights.cpu().numpy())[:hyperparameters['n_anchor_patches_N_in']]
                    loc_highest_weight_node = non_padded_component_nodes[sorted_indices]

                    if len(loc_highest_weight_node) < hyperparameters['n_anchor_patches_N_in']:
                        loc_highest_weight_node = torch.cat([
                        loc_highest_weight_node,torch.full((hyperparameters['n_anchor_patches_N_in'] - len(loc_highest_weight_node),),
                        fill_value=config.PAD_VALUE,dtype=torch.long)
                        ])
                        
                    subgraph_samples.append(loc_highest_weight_node.tolist())
                else:
                    subgraph_samples.append([0] * hyperparameters['n_anchor_patches_N_in'])

            all_samples.append(subgraph_samples)
        
        anchor_patches = torch.tensor(all_samples)
        

    # Sampling global anchor patches
    else:
        for subgraph_idx, components in enumerate(cc_ids): 
            subgraph_samples = []
            for component in range(max_n_cc):
                global_nodes = global_set[subgraph_idx, component]  
                non_padded_global_nodes = global_nodes[global_nodes != 0]  

                if len(non_padded_global_nodes) > 0:
                    global_weights = dis_similarities_cc[subgraph_idx, component, non_padded_global_nodes - 1]
                    sorted_global_indices = np.argsort(-global_weights.cpu().numpy())[:hyperparameters['n_anchor_patches_N_out']]
                    global_highest_weight_node = non_padded_global_nodes[sorted_global_indices]

                    if len(global_highest_weight_node) < hyperparameters['n_anchor_patches_N_out']:
                        global_highest_weight_node = torch.cat([
                        global_highest_weight_node,torch.full((hyperparameters['n_anchor_patches_N_out'] - len(global_highest_weight_node),),
                        fill_value=config.PAD_VALUE,dtype=torch.long)
                        ])
                    subgraph_samples.append(global_highest_weight_node.tolist())
                else:
                    subgraph_samples.append([0] * hyperparameters['n_anchor_patches_N_out'])
            
            all_samples.append(subgraph_samples)
            
        anchor_patches = torch.tensor(all_samples)
        
    return anchor_patches



def sample_structure_anchor_patches(hyperparameters, PPI_graph, all_subgraphs_dis_similarities, device):

    """
    Sample structure-based anchor patches using ego-graphs around top-ranked nodes, which can be used for sampling local and global sequences.

    Returns:
        - structure_anchors_dic (dict): Mapping from center node to anchor patch (as tensor)
        - structure_anchors (Tensor): Shape (n_sampled_patches, max_patch_length)
    """

    n_samples = hyperparameters['max_sample_anchor_patches_structure']  
    all_patches = []
    structure_anchors_dic = defaultdict(dict)

    # Compute cumulative similarity score across all subgraphs for each node
    all_subgraph_total_weights = np.zeros(all_subgraphs_dis_similarities.shape[1]) 
    for subgraph in range(all_subgraphs_dis_similarities.shape[0]): 
                curr_subgraph_weights = all_subgraphs_dis_similarities[subgraph, :].detach().cpu().numpy() 
                all_subgraph_total_weights +=  curr_subgraph_weights

    start_nodes = np.argsort(-all_subgraph_total_weights)[:n_samples] + 1 

    for i, node in enumerate(start_nodes):
        subgraph = [n for n in list(nx.ego_graph(PPI_graph, node, radius=hyperparameters['structure_anchor_patch_radius']).nodes)] 
        all_patches.append(subgraph)
    
    max_anchor_len = max([len(s) for s in all_patches])
    padded_all_patches = []
  
    for patch in all_patches:
        fill_len = max_anchor_len - len(patch)
        padded_patch = torch.cat([torch.LongTensor(patch),torch.LongTensor((fill_len)).fill_(config.PAD_VALUE)])
        padded_all_patches.append(padded_patch)
        
    structure_anchors = torch.stack(padded_all_patches).long()

    # Create dictionary mapping start node to its corresponding anchor patch
    for idx, patch in enumerate(structure_anchors):
            start_node = start_nodes[idx]
            structure_anchors_dic[start_node] = patch
    
    return structure_anchors_dic, structure_anchors



def init_anchors_pos_loc(split, hyperparameters, train_ccs_neigh_pos_similarities, val_ccs_neigh_pos_similarities, test_ccs_neigh_pos_similarities,\
                                device, train_cc_ids, val_cc_ids , test_cc_ids, train_subgraphs, val_subgraphs, test_subgraphs ):
    
    """
    Initialize local positional anchor patches for each dataset and layer.
    
    Returns:
        - anchors_pos_loc (dict): Nested dictionary:
              dataset name -> layer index -> tensor of shape (n_subgraphs, n_anchor_patches_pos_in)
    """
   
    if split == 'all':
        dataset_names = ['train', 'val', 'test']
        datasets = [train_subgraphs, val_subgraphs, test_subgraphs]
        cc_ids = [train_cc_ids, val_cc_ids, test_cc_ids]
        dis_similarities_ccs =  [train_ccs_neigh_pos_similarities, val_ccs_neigh_pos_similarities, test_ccs_neigh_pos_similarities]
        
    elif split == 'train_val':
        dataset_names = ['train', 'val']
        datasets = [train_subgraphs, val_subgraphs]
        cc_ids = [train_cc_ids, val_cc_ids]
        dis_similarities_ccs =  [train_ccs_neigh_pos_similarities, val_ccs_neigh_pos_similarities]
        
    elif split == 'test':
        dataset_names = ['test']
        datasets = [test_subgraphs]
        cc_ids = [test_cc_ids]
        dis_similarities_ccs =  [test_ccs_neigh_pos_similarities]
        
    anchors_pos_loc = defaultdict(dict)
    
    for dataset_name, dataset, dis_similarities_cc, cc_id in zip(dataset_names, datasets, dis_similarities_ccs, cc_ids):
        anchors = sample_position_anchor_patches (hyperparameters=hyperparameters,  dis_similarities_subgraph=None,
                                                  dis_similarities_cc=dis_similarities_cc, cc_ids=cc_id, subgraphs=dataset)
        for layer in range(hyperparameters['n_layers']):
            anchors_pos_loc[dataset_name][layer] = [subgraph[layer * hyperparameters['n_anchor_patches_pos_in']:(layer + 1) * hyperparameters['n_anchor_patches_pos_in']]
            for subgraph in anchors if len(subgraph) >= (layer + 1) * hyperparameters['n_anchor_patches_pos_in']]
            
            anchors_pos_loc[dataset_name][layer] = (
                torch.stack([torch.tensor(patch) for patch in anchors_pos_loc[dataset_name][layer]])
                if anchors_pos_loc[dataset_name][layer]
                else torch.tensor([])
                )

    return anchors_pos_loc

   

def init_anchors_pos_glob(hyperparameters, dis_similarities_subgraph, device):
    
    """
    Initialize global positional anchor patches for each layer.

    Returns:
        - anchors_pos_glob (dict): Dictionary:
              layer index -> tensor of shape (n_anchor_patches_pos_out)
    """
    
    anchors_pos_glob = {}
    anchors = torch.tensor(sample_position_anchor_patches(
        hyperparameters=hyperparameters, dis_similarities_subgraph=dis_similarities_subgraph, dis_similarities_cc=None, cc_ids=None, subgraphs=None))
  
    for layer in range(hyperparameters['n_layers']):
            start_idx = layer *  hyperparameters['n_anchor_patches_pos_out']
            end_idx = start_idx + hyperparameters['n_anchor_patches_pos_out']
            patches = anchors[start_idx:end_idx]
            anchors_pos_glob[layer] = patches
            
    return anchors_pos_glob



def init_anchors_neighborhood(split, hyperparameters, device, train_cc_ids, val_cc_ids, test_cc_ids, train_N_border, val_N_border, test_N_border,\
                              train_ccs_neigh_pos_similarities, val_ccs_neigh_pos_similarities, test_ccs_neigh_pos_similarities):
    
    """
    Initialize local and global neighborhood-based anchor patches per dataset and layer.

    Returns:
        - anchors_neigh_loc (dict): Local neighborhood patches (dataset -> layer -> tensor)
        - anchors_neigh_glob (dict): Global neighborhood patches (dataset -> layer -> tensor)
    """

    if split == 'all':
        dataset_names = ['train', 'val', 'test']
        datasets = [train_cc_ids, val_cc_ids, test_cc_ids]
        global_sets = [train_N_border, val_N_border, test_N_border]
        dis_similarities_ccs =  [train_ccs_neigh_pos_similarities, val_ccs_neigh_pos_similarities, test_ccs_neigh_pos_similarities]
        
    elif split == 'train_val':
        dataset_names = ['train', 'val']
        datasets = [train_cc_ids, val_cc_ids]
        global_sets = [train_N_border, val_N_border]
        dis_similarities_ccs =  [train_ccs_neigh_pos_similarities, val_ccs_neigh_pos_similarities]  
        
    elif split == 'test':
        dataset_names = ['test']
        datasets = [test_cc_ids]
        global_sets = [test_N_border]
        dis_similarities_ccs =  [test_ccs_neigh_pos_similarities]
        
    anchors_neigh_loc = defaultdict(dict)
    anchors_neigh_glob = defaultdict(dict)

    for dataset_name, dataset, global_set, dis_similarities_cc in zip(dataset_names, datasets, global_sets, dis_similarities_ccs):
        for layer in range(hyperparameters['n_layers']):
            anchors_neigh_loc[dataset_name][layer] = sample_neighborhood_anchor_patch(hyperparameters, dataset, global_set, dis_similarities_cc, sample_inside=True)
            anchors_neigh_glob[dataset_name][layer] = sample_neighborhood_anchor_patch(hyperparameters, dataset, global_set, dis_similarities_cc, sample_inside=False)
            
    return anchors_neigh_loc, anchors_neigh_glob



def init_anchors_structure(hyperparameters, structure_anchors, loc_structure_anchor_hawkes_process, glob_structure_anchor_hawkes_process):
   
    """
    Initialize structure-based anchor patches and precomputed local & global Hawkes sequences for each layer.

    Returns:
        - anchors_struc (dict): Dictionary:
              layer index -> (anchor_patch_tensor, patch_indices, local_hw_seq, global_hw_seq)
    """
    
    anchors_struc = {}
    
    for layer in range(hyperparameters['n_layers']):
        start_idx = layer * hyperparameters['n_anchor_patches_structure']
        end_idx = (layer + 1) * hyperparameters['n_anchor_patches_structure']
        if end_idx > structure_anchors.shape[0]:
            end_idx = structure_anchors.shape[0]
        indices = list(range(start_idx, end_idx))
        
        anchors_struc[layer] = (structure_anchors[indices,:], indices, loc_structure_anchor_hawkes_process[indices,:,:], glob_structure_anchor_hawkes_process[indices,:,:] )
    
    return anchors_struc



def retrieve_anchor_patche_embeds(hyperparameters, subgraph_idx, cc_ids, cc_embed_mask, node_embeds,\
    anchors_pos_loc, anchors_pos_glob, anchors_neigh_loc, anchors_neigh_global, anchors_structure,\
    loc_struct_anchor_embeds, glob_struct_anchor_embeds, layer_num, head_type, dataset_type, inside, device=None):

    """
    Retrieve anchor patches and their embeddings for a given head type and layer.

    Returns:
        - anchor_patches (Tensor): Anchor patch IDs of shape (batch_size, max_n_cc, n_anchor_patches, max_len).
        - anchor_patche_mask (Tensor): Boolean mask of shape (batch_size, max_n_cc, n_anchor_patches, max_len), where True indicates valid entries.
        - anchor_patche_embeds (Tensor): Anchor patch embeddings of shape (batch_size, max_n_cc, n_anchor_patches, embed_dim).
    
    """
    
    batch_size, max_n_cc, max_size_cc = cc_ids.shape
   
    if head_type == 'position':
        if inside:
            anchor_patches_tensor = anchors_pos_loc[dataset_type][layer_num][subgraph_idx.to('cpu')].squeeze(1)
            anchor_patches = anchor_patches_tensor.unsqueeze(1).repeat(1,max_n_cc,1) # repeat anchor patches for each CC
            anchor_patches[~cc_embed_mask] = config.PAD_VALUE # mask padding CCs
        else:
            anchor_patches = anchors_pos_glob[layer_num].unsqueeze(0).unsqueeze(0).repeat(batch_size,max_n_cc,1)
            anchor_patches[~cc_embed_mask] = config.PAD_VALUE 

        anchor_patches = anchor_patches.to(device) 
        anchor_patche_embeds = node_embeds(anchor_patches.to(node_embeds.weight.device).long())
        anchor_patche_mask = (anchor_patches != config.PAD_VALUE).bool().unsqueeze(-1)
        anchor_patches = anchor_patches.unsqueeze(-1)

    elif head_type == 'neighborhood':
        if inside:
            anchor_patches = anchors_neigh_loc[dataset_type][layer_num][subgraph_idx.to('cpu')].squeeze(1).to(cc_ids.device)
        else:
            anchor_patches = anchors_neigh_global[dataset_type][layer_num][subgraph_idx.to('cpu')].squeeze(1).to(cc_ids.device)
            
        anchor_patches = anchor_patches.to(device)  
        anchor_patche_embeds = node_embeds(anchor_patches.to(node_embeds.weight.device).long())
        anchor_patche_mask = (anchor_patches != config.PAD_VALUE).bool().unsqueeze(-1)
        anchor_patches = anchor_patches.unsqueeze(-1)

    elif head_type == 'structure':
        anchor_patches,anchor_patch_indx, _, _ = anchors_structure[layer_num]
        precomputed_struct_embeddings = loc_struct_anchor_embeds if inside else glob_struct_anchor_embeds 
        anchor_patche_embeds = precomputed_struct_embeddings[anchor_patch_indx]  
            
        anchor_patches = anchor_patches.unsqueeze(0).unsqueeze(0).expand(batch_size, max_n_cc, -1, -1)
        anchor_patches[~cc_embed_mask] = config.PAD_VALUE  

        anchor_patche_mask = (anchor_patches != config.PAD_VALUE).bool()
        anchor_patche_embeds = anchor_patche_embeds.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_n_cc, 1, 1)
        anchor_patche_embeds[~cc_embed_mask] = config.PAD_VALUE

    return anchor_patches, anchor_patche_mask, anchor_patche_embeds





