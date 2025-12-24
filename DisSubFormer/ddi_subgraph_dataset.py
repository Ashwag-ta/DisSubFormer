# General Imports
from typing import Dict

# PyTorch
import torch
from torch.utils.data import Dataset



class DDISubgraphDataset(Dataset):
    
    """
    Custom PyTorch Dataset for Disease-Disease Interaction (DDI) link prediction.

    Include subgraph metadata and DDI labels, along with precomputed similarity matrices for positional,
    neighborhood, and structural attention heads, both locally and globally.
    """

    def __init__(self, DDI_edges, DDI_edges_label, subgraphs_dict: Dict,
                 subgraphs_idx, cc_ids,loc_pos_head_sim: Dict, glob_pos_head_sim: Dict, loc_neigh_head_sim: Dict,
                 glob_neigh_head_sim: Dict, loc_struct_head_sim, glob_struct_head_sim):
        

       # Subgraph IDs, DDI edges, and associated labels
        self.subgraphs_idx = subgraphs_idx
        self.subgraphs_dict = subgraphs_dict
        self.cc_ids = cc_ids
        self.DDI_edges = DDI_edges  
        self.DDI_edges_label = DDI_edges_label

        # Precomputed similarity matrices per attention head  
        self.loc_pos_head_sim = loc_pos_head_sim  
        self.glob_pos_head_sim = glob_pos_head_sim  
        self.loc_neigh_head_sim = loc_neigh_head_sim 
        self.glob_neigh_head_sim = glob_neigh_head_sim 
        self.loc_struct_head_sim = loc_struct_head_sim 
        self.glob_struct_head_sim = glob_struct_head_sim 
        
        # Map original subgraph indices to a continuous range
        self.subgraphs_idx_mapping = {original_idx: map_idx for map_idx, original_idx in enumerate(self.subgraphs_idx)}
    
  

    def __len__(self):
        
        """
        Return the number of DDI edges in the dataset.
        """
        
        return len(self.DDI_edges_label)


   
    def __getitem__(self, idx):
        
        """
        Retrieve a single DDI sample and its associated subgraph data.
        
        Return:
            The edge, label, subgraph metadata, and similarity tensors for both subgraphs involved in the interaction.
        """
       
        # Get edge and label
        DDI_edge = self.DDI_edges[:, idx] 
        DDI_edge_label = self.DDI_edges_label[idx]
        subgraph1_idx, subgraph2_idx = torch.tensor(DDI_edge[0].item()), torch.tensor(DDI_edge[1].item())
        
        # Get node IDs and connected component IDs for each subgraph
        subgraph1_node_ids = torch.tensor(self.subgraphs_dict.get(int(subgraph1_idx.item()), [])).squeeze()
        subgraph2_node_ids = torch.tensor(self.subgraphs_dict.get(int(subgraph2_idx.item()), [])).squeeze()
        subgraph1_cc_ids = self.cc_ids[subgraph1_idx]
        subgraph2_cc_ids = self.cc_ids[subgraph2_idx]

        ###################################################
        # Extract similarities for all heads
        ###################################################

        # Extract positional similarities
        subgraph1_loc_pos_head_sim = self.extract_subgraph_similarities(self.loc_pos_head_sim, subgraph1_idx) if self.loc_pos_head_sim != None else None
        subgraph2_loc_pos_head_sim = self.extract_subgraph_similarities(self.loc_pos_head_sim, subgraph2_idx) if self.loc_pos_head_sim != None else None
        subgraph1_glob_pos_head_sim = self.extract_subgraph_similarities(self.glob_pos_head_sim, subgraph1_idx) if self.glob_pos_head_sim != None else None
        subgraph2_glob_pos_head_sim = self.extract_subgraph_similarities(self.glob_pos_head_sim, subgraph2_idx) if self.glob_pos_head_sim != None else None

        # Extract neighborhood similarities
        subgraph1_loc_neigh_head_sim = self.extract_subgraph_similarities(self.loc_neigh_head_sim, subgraph1_idx) if self.loc_neigh_head_sim != None else None
        subgraph2_loc_neigh_head_sim = self.extract_subgraph_similarities(self.loc_neigh_head_sim, subgraph2_idx) if self.loc_neigh_head_sim != None else None
        subgraph1_glob_neigh_head_sim = self.extract_subgraph_similarities(self.glob_neigh_head_sim, subgraph1_idx) if self.glob_neigh_head_sim != None else None
        subgraph2_glob_neigh_head_sim = self.extract_subgraph_similarities(self.glob_neigh_head_sim, subgraph2_idx) if self.glob_neigh_head_sim != None else None

        # Extract structural similarities
        subgraph1_loc_struct_head_sim = self.loc_struct_head_sim[subgraph1_idx] if self.loc_struct_head_sim != None else None
        subgraph2_loc_struct_head_sim = self.loc_struct_head_sim[subgraph2_idx] if self.loc_struct_head_sim != None else None
        subgraph1_glob_struct_head_sim = self.glob_struct_head_sim[subgraph1_idx] if self.glob_struct_head_sim != None else None
        subgraph2_glob_struct_head_sim = self.glob_struct_head_sim[subgraph2_idx] if self.glob_struct_head_sim != None else None
       
        return ( DDI_edge, DDI_edge_label, subgraph1_idx, subgraph1_node_ids, subgraph1_cc_ids,subgraph1_loc_pos_head_sim, subgraph1_glob_pos_head_sim,
                 subgraph1_loc_neigh_head_sim, subgraph1_glob_neigh_head_sim, subgraph1_loc_struct_head_sim, subgraph1_glob_struct_head_sim,
                 subgraph2_idx, subgraph2_node_ids, subgraph2_cc_ids, subgraph2_loc_pos_head_sim, subgraph2_glob_pos_head_sim,
                 subgraph2_loc_neigh_head_sim, subgraph2_glob_neigh_head_sim,   
                 subgraph2_loc_struct_head_sim, subgraph2_glob_struct_head_sim)
    

    
    def extract_subgraph_similarities(self, sim_dict, subgraph_idx):
        
            """
            Extract similarity tensors for a given subgraph across all layers from a similarity dictionary.
            """
            
            subgraph_similarities = {}
            for layer, sim in sim_dict.items():
                subgraph_similarities[layer] = sim[subgraph_idx]
                        
            return subgraph_similarities
        
