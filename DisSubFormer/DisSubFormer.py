# General Imports
import sys
import numpy as np
from pathlib import Path
import json
from typing import Dict
import multiprocessing

# Sci-kit Learn 
from sklearn.metrics import  roc_curve, roc_auc_score, precision_recall_curve

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parameter import Parameter

# Pytorch lightning
#import pytorch_lightning as pl
import lightning.pytorch as pl

# Networkx
import networkx as nx

# Ours
sys.path.insert(0, '..') 
import main_config as config
import dissubformer_utils
from anchor_patch_sampling import *
from ddi_subgraph_dataset import DDISubgraphDataset
from heads_attention_dissubformer import MHADisSubFormer
from hawkes_embedding_model import *



class DisSubFormer(pl.LightningModule):
    
    """
    PyTorch Lightning module for the DisSubFormer model.
    
    This model performs link prediction using subgraph embeddings to predict disease comorbidity.
    """

    def __init__(self, hyperparameters: Dict, PPI_graph_path: str, DDI_RR0_graph_path: str, Subgraphs_path: str,
                 Embedding_path: str,  AP_sampling_similarities_path: str, Head_attention_similarities_path: str, Ego_graph_path: str,
                 Euclidean_distances_path:str, Shortest_paths_path:str, Intermediate_nodes_path:str, ADJ_matrix_path: str, Node_degrees_dic_path: str ):
        super(DisSubFormer, self).__init__()
        
        # Dictionary of model hyperparameters
        self.hyperparameters = hyperparameters
        
        # Paths to input data and precomputed matrices
        self.PPI_graph_path = PPI_graph_path
        self.DDI_RR0_graph_path = DDI_RR0_graph_path
        self.Subgraphs_path = Subgraphs_path
        self.Embedding_path = Embedding_path
        self.AP_sampling_similarities_path = AP_sampling_similarities_path 
        self.Head_attention_similarities_path = Head_attention_similarities_path
        self.Ego_graph_path = Ego_graph_path
        self.Euclidean_distances_path = Euclidean_distances_path
        self.Shortest_paths_path = Shortest_paths_path
        self.Intermediate_nodes_path = Intermediate_nodes_path
        self.ADJ_matrix_path = ADJ_matrix_path
        self.Node_degrees_dic_path = Node_degrees_dic_path
        
        # Read in data
        self.read_data()

        # Model configuration hyperparameters
        self.node_embed_size = self.hyperparameters['node_embed_size']
        self.dropout_rate = self.hyperparameters['dropout_rate']
        self.attention_dropout = nn.Dropout(self.dropout_rate)
        self.ffn_dropout = nn.Dropout(self.dropout_rate)
        self.learning_rate = hyperparameters['learning_rate']
        
        # Dynamically Compute the number of active heads
        self.active_heads = sum([
            self.hyperparameters.get('neighborhood_head', False),
            self.hyperparameters.get('structure_head', False),
            self.hyperparameters.get('position_head', False)
            ])
        
        # Initial linear transformation 
        self.initial_transformation = nn.Linear(self.node_embed_size, self.node_embed_size)
        
        # Projection after concatenation of multi-head outputs
        self.all_head_projection = nn.Sequential(
            nn.Linear(self.active_heads * self.node_embed_size, self.node_embed_size),
            nn.ReLU())

        # Layer normalizations
        self.layer_norm1 = nn.LayerNorm(self.node_embed_size) 
        self.layer_norm2 = nn.LayerNorm(self.node_embed_size) 

        # Feed-forward block
        self.feed_forward = nn.Sequential(
            nn.Linear(self.node_embed_size, self.node_embed_size * 2),
            nn.ReLU(),
            self.ffn_dropout,  
            nn.Linear(self.node_embed_size * 2, self.node_embed_size),
            self.ffn_dropout)
        
        # MLP for final link prediction
        self.mlp = nn.Sequential(
            nn.Linear(self.node_embed_size * 2 * self.hyperparameters['n_layers'], self.node_embed_size),
            nn.ReLU(),
            nn.Linear(self.node_embed_size, 1),
            nn.Sigmoid()
            )

        # Binary cross-entropy loss function for link prediction
        self.loss = nn.BCELoss() 

        # Track validation metrics
        self.val_metric_scores = []

        # To store Hawkes process-based embeddings for structural anchor patches encoding 
        self.hawkes_node_embeddings = None

        # Initialize the DisSubFormer attention heads
        self.initialize_heads()


   
    def initialize_heads(self):
        
            """
            Initialize attention heads for position, neighborhood, and structure.
            """

            if self.hyperparameters['neighborhood_head']:
                self.neighborhood_head = self.initialize_single_head(self.hyperparameters['n_layers'], "neighborhood")
                print(" >>> Neighborhood head initialization complete <<< ")

            if self.hyperparameters['position_head']:
                self.position_head = self.initialize_single_head(self.hyperparameters['n_layers'], "position")
                print(" >>> Position head initialization complete <<< ")

            if self.hyperparameters['structure_head']:
                self.structure_head = self.initialize_single_head(self.hyperparameters['n_layers'], "structure")
                print(" >>> Structure head initialization complete <<< ")



    def initialize_single_head(self, n_layers, head_type):
        
            """
            Initialize a single head (position/neighborhood/structure) with local and global attention.
            
            Args:
                - n_layers (int): Number of layers for DisSubFormer.
                - head_type (str): The type of attention head to initialize ('position', 'neighborhood', or 'structure').
                
            Returns:
                - head (ModuleList object): A list of attention layers for the specified head, where each layer includes local and global components.   
            """
            
            head = nn.ModuleList()
            for l in range(n_layers):
                layer = nn.ModuleDict({
                    'query_local': nn.Linear(self.node_embed_size, self.node_embed_size, bias=False),
                    'key_local': nn.Linear(self.node_embed_size, self.node_embed_size, bias=False),
                    'value_local': nn.Linear(self.node_embed_size, self.node_embed_size, bias=False),
                    'query_global': nn.Linear(self.node_embed_size, self.node_embed_size, bias=False),
                    'key_global': nn.Linear(self.node_embed_size, self.node_embed_size, bias=False),
                    'value_global': nn.Linear(self.node_embed_size, self.node_embed_size, bias=False),
                    'local': MHADisSubFormer(self.hyperparameters),
                    'global': MHADisSubFormer(self.hyperparameters)
                })
                head.append(layer)
                
            return head

        

    def MHADisSubFormer_layer(self, dataset_type, attention_head, head_Q, head_K, head_V, subgraph_idx, subgraph_ids, cc_ids, 
            cc_embeds, cc_embed_mask, sims, layer_num, head_type, inside=True):
        
        """
        Perform a single DisSubFormer local or global head attention layer for a specific head.

        Returns:
            - cc_embed_matrix (Tensor): Updated embedding matrix for the connected components.
        """

        # Retrieve anchor patches and their embeddings
        if head_type == 'structure':
            anchor_patches, anchor_mask, anchor_embeds = retrieve_anchor_patche_embeds(self.hyperparameters, subgraph_idx, cc_ids, cc_embed_mask, None,
                            self.anchors_pos_loc, self.anchors_pos_glob, self.anchors_neigh_loc, self.anchors_neigh_glob, self.anchors_structure,
                            self.loc_structure_anchor_hawkes_process_embeds, self.glob_structure_anchor_hawkes_process_embeds, layer_num, head_type,
                                                                                       dataset_type, inside, device=None)
            anchor_mask = anchor_mask[..., :1]
            
            # Extract similarity indices of current structure APs from the full pre-sampled similarity matrix
            struct_anchors_sim_idx = self.anchors_structure[layer_num][1]
            
        else:
            anchor_patches, anchor_mask, anchor_embeds = retrieve_anchor_patche_embeds(self.hyperparameters, subgraph_idx, cc_ids, cc_embed_mask, self.node_embeddings,
            self.anchors_pos_loc, self.anchors_pos_glob, self.anchors_neigh_loc, self.anchors_neigh_glob, self.anchors_structure, None, None, layer_num, head_type,
                                                                                       dataset_type, inside, device=None)
            struct_anchors_sim_idx = None

        # Apply initial linear transformation to anchor patch embeddings
        anchor_embeds = anchor_embeds.to(self.initial_transformation.weight.device)
        anchor_embeds = self.initial_transformation(anchor_embeds)
        anchor_mask = anchor_mask.to(anchor_embeds.device)
        anchor_embeds = anchor_embeds * anchor_mask.to(anchor_embeds.device)

        # Perform attention computation to update CC embeddings
        cc_embed_matrix = attention_head(head_Q, head_K, head_V, sims, struct_anchors_sim_idx, cc_ids, 
            cc_embeds, cc_embed_mask, anchor_patches, anchor_embeds, anchor_mask, head_type)
                
        return cc_embed_matrix


    
# =============================
# Forward pass
# =============================

    def forward(self, batch_idx, dataset_type, DDI_edges, DDI_edge_labels, subgraph_idx, subgraph_ids, cc_ids, loc_pos_head_sim, glob_pos_head_sim,
                loc_neigh_head_sim, glob_neigh_head_sim, loc_struct_head_sim, glob_struct_head_sim):
        
        """
        Forward pass for DisSubFormer model.

        Computes subgraph embeddings by applying multi-head attention from positional, neighborhood, and structural heads across all layers.

        Returns:
            - subgraph_embedding (Tensor): Final subgraph representation for disease comorbidity prediction.
        """
        
        print(f">>> Start forward pass of the model for batch {batch_idx + 1} in {dataset_type} data <<<")
        
        # Initialize connected component (CC) embeddings
        init_cc_embeds = self.init_cc_embeddings(cc_ids, self.hyperparameters['cc_aggregator'])
        batch_sz, max_n_cc, _ = init_cc_embeds.shape
        cc_embed_mask = (cc_ids != config.PAD_VALUE)[:,:,0]

        # Apply initial linear transformation to CC embeddings
        init_cc_embeds = self.initial_transformation(init_cc_embeds)
        init_cc_embeds = init_cc_embeds * cc_embed_mask.unsqueeze(-1)
    
        neigh_loc_cc_embeds = init_cc_embeds.clone()  
        neigh_glob_cc_embeds = init_cc_embeds.clone()    
        pos_loc_cc_embeds = init_cc_embeds.clone()    
        pos_glob_cc_embeds = init_cc_embeds.clone()      
        struct_loc_cc_embeds = init_cc_embeds.clone() 
        struct_glob_cc_embeds = init_cc_embeds.clone()    
        
        prev_layer_output = init_cc_embeds.clone()
        cc_embeds_all_layers_heads_output = []
        
        for l in range(self.hyperparameters['n_layers']):
            
            layer_head_outputs = []

            
             # Neighborhood head
            if self.hyperparameters['neighborhood_head']:
                print(" >>> Neighborhood head attention computation to update CC embeddings <<< ")

                # Local 
                cc_embed_loc_NH = self.MHADisSubFormer_layer(
                    dataset_type, self.neighborhood_head[l]['local'], self.neighborhood_head[l]['query_local'],
                    self.neighborhood_head[l]['key_local'], self.neighborhood_head[l]['value_local'], 
                    subgraph_idx, subgraph_ids, cc_ids, neigh_loc_cc_embeds, cc_embed_mask, sims=loc_neigh_head_sim[l], 
                    layer_num=l, head_type='neighborhood', inside=True
                    )
                # Global   
                cc_embed_glob_NH = self.MHADisSubFormer_layer(
                    dataset_type, self.neighborhood_head[l]['global'], self.neighborhood_head[l]['query_global'], 
                    self.neighborhood_head[l]['key_global'], self.neighborhood_head[l]['value_global'], 
                    subgraph_idx, subgraph_ids, cc_ids, neigh_glob_cc_embeds, cc_embed_mask, sims=glob_neigh_head_sim[l], 
                    layer_num=l, head_type='neighborhood', inside=False
                    )
                
                # Combine local and global neighborhood embeddings
                NH_output = cc_embed_loc_NH.squeeze(2) + cc_embed_glob_NH.squeeze(2)

                # Apply dropout 
                NH_output = self.attention_dropout(NH_output)
                layer_head_outputs.append(NH_output)


            # Position head
            if self.hyperparameters['position_head']:
                print(" >>> Position head attention computation to update CC embeddings <<< ")

                # Local    
                cc_embed_loc_PH = self.MHADisSubFormer_layer(
                    dataset_type, self.position_head[l]['local'], self.position_head[l]['query_local'], 
                    self.position_head[l]['key_local'], self.position_head[l]['value_local'], 
                    subgraph_idx, subgraph_ids, cc_ids, pos_loc_cc_embeds, cc_embed_mask, 
                    sims=loc_pos_head_sim[l], layer_num=l, head_type='position', inside=True
                    )
                # Global
                cc_embed_glob_PH = self.MHADisSubFormer_layer(
                    dataset_type, self.position_head[l]['global'], self.position_head[l]['query_global'], 
                    self.position_head[l]['key_global'], self.position_head[l]['value_global'], 
                    subgraph_idx, subgraph_ids, cc_ids, pos_glob_cc_embeds, cc_embed_mask, 
                    sims=glob_pos_head_sim[l], layer_num=l, head_type='position', inside=False
                    )
                
                 # Combine local and global position embeddings
                PH_output = cc_embed_loc_PH.squeeze(2) + cc_embed_glob_PH.squeeze(2)
                
                # Apply dropout 
                PH_output = self.attention_dropout(PH_output)
                layer_head_outputs.append(PH_output)


            # Structure head
            if self.hyperparameters['structure_head']:
                print(" >>> Structure head attention computation to update CC embeddings <<< ")

                # Local      
                cc_embed_loc_SH = self.MHADisSubFormer_layer(
                    dataset_type, self.structure_head[l]['local'], self.structure_head[l]['query_local'], 
                    self.structure_head[l]['key_local'], self.structure_head[l]['value_local'], 
                    subgraph_idx, subgraph_ids, cc_ids, struct_loc_cc_embeds, cc_embed_mask, 
                    sims=loc_struct_head_sim, layer_num=l, head_type='structure', inside=True
                    )
                # Global  
                cc_embed_glob_SH = self.MHADisSubFormer_layer(
                    dataset_type, self.structure_head[l]['global'], self.structure_head[l]['query_global'], 
                    self.structure_head[l]['key_global'], self.structure_head[l]['value_global'], 
                    subgraph_idx, subgraph_ids, cc_ids, struct_glob_cc_embeds, cc_embed_mask, 
                    sims=glob_struct_head_sim, layer_num=l, head_type='structure', inside=False)
          
                # Combine local and global structure embeddings
                SH_output = cc_embed_loc_SH.squeeze(2) + cc_embed_glob_SH.squeeze(2)

                # Apply dropout 
                SH_output = self.attention_dropout(SH_output)
                layer_head_outputs.append(SH_output)
    
                
            # Concatenate all heads' outputs
            all_head_outputs = torch.cat(layer_head_outputs, dim=-1)

            # Project concatenated multi-head output
            projected_all_head_outputs = self.all_head_projection(all_head_outputs)

            # Residual connection and normalization
            residual_input = prev_layer_output
            projected_all_head_outputs_normalized = self.layer_norm1(projected_all_head_outputs + residual_input)

            # Feed-Forward Layer followed by residual connection and normalization
            ffn_output = self.feed_forward(projected_all_head_outputs_normalized)
            ffn_residual_input = projected_all_head_outputs_normalized
            fnn_normalized = self.layer_norm2(ffn_output + ffn_residual_input) 
            prev_layer_output = fnn_normalized
            cc_embeds_all_layers_heads_output.append(fnn_normalized)
            
         # Concatenate embeddings from all layers
        concatenated_all_layers = torch.cat(cc_embeds_all_layers_heads_output, dim=-1)

        # Aggregate CC embeddings to get final subgraph embedding
        subgraph_embedding = (concatenated_all_layers * cc_embed_mask.unsqueeze(-1)).sum(dim=1)
        
        print(f">>> Final subgraph embeddings successfully computed for batch {batch_idx + 1} <<<")

        return subgraph_embedding



# =============================
# Hawkes-Based Embeddings for Structure Anchor Patches
# =============================

    def setup(self, stage=None):

        """
        Setup method for preparing Hawkes process-based embeddings depending on the stage.

        Args:
            - stage (str): One of 'fit', 'validate', or 'test'.
        """
        
        if self.hyperparameters['structure_head']:
            hawkes_node_embeddings_path = config.PROJECT_ROOT /'Data/hawkes_node_embeddings.pth'
            hawkes_loc_struc_patch_embeddings_path = config.PROJECT_ROOT /'Data/hawkes_loc_struc_patch_embeddings.pth'
            hawkes_glob_struc_patch_embeddings_path = config.PROJECT_ROOT /'Data/hawkes_glob_struc_patch_embeddings.pth'
            
            if stage == "fit":
                if hawkes_node_embeddings_path.exists():
                    print(" >>> Hawkes node embeddings already exist. Loading from file <<< ")
                    self.hawkes_node_embeddings = torch.load(hawkes_node_embeddings_path)
                else:
                    print(" >>> Training Hawkes process for node embeddings <<< ")
                    hawkes_trainer = HawkesSeqEmbedding(
                        self.PPI_graph,
                        self.node_embeddings,
                        self.loc_structure_anchor_hawkes_process,
                        self.glob_structure_anchor_hawkes_process,
                        self.hyperparameters,
                        hawkes_node_embeddings_path
                        )
                     
                    # Train Hawkes embeddings
                    hawkes_trainer.train_hawkes_node_embeddings()
                    print(" >>> Hawkes node embeddings computed and saved <<< ")

                    # Retrieve trained embeddings
                    self.hawkes_node_embeddings = hawkes_trainer.retrieve_hawkes_embeddings()
                    
                #Aggregate embeddings for structure-based anchor patches.
                if hawkes_loc_struc_patch_embeddings_path.exists()and hawkes_glob_struc_patch_embeddings_path.exists():
                     print(" >>> Loading precomputed Hawkes local and global structure anchor patch embeddings for Train and Val Datasets <<< ")
                     self.loc_structure_anchor_hawkes_process_embeds = torch.load(hawkes_loc_struc_patch_embeddings_path)
                     self.glob_structure_anchor_hawkes_process_embeds = torch.load(hawkes_glob_struc_patch_embeddings_path)
                else:
                     print(" >>> Aggregating Hawkes local and global structure anchor patch embeddings for Train and Val Datasets <<< ")
                     self.loc_structure_anchor_hawkes_process_embeds = self.aggregate_structure_anchor_patch_embeddings(self.loc_structure_anchor_hawkes_process)
                     torch.save(self.loc_structure_anchor_hawkes_process_embeds, hawkes_loc_struc_patch_embeddings_path)
                     
                     self.glob_structure_anchor_hawkes_process_embeds = self.aggregate_structure_anchor_patch_embeddings(self.glob_structure_anchor_hawkes_process)
                     torch.save(self.glob_structure_anchor_hawkes_process_embeds, hawkes_glob_struc_patch_embeddings_path)
                     
            if stage == "test":
                print(" >>> Preparing Hawkes structure anchor patch embeddings for Test Dataset <<< ")
              
                if hawkes_node_embeddings_path.exists() and hawkes_loc_struc_patch_embeddings_path.exists()and hawkes_glob_struc_patch_embeddings_path.exists():
                    print(" >>> Loading precomputed Hawkes node embeddings <<< ")
                    self.hawkes_node_embeddings = torch.load(hawkes_node_embeddings_path)
                    print(" >>> Loading precomputed Hawkes local and global structure anchor patch embeddings for Test Dataset <<< ")
                    self.loc_structure_anchor_hawkes_process_embeds = torch.load(hawkes_loc_struc_patch_embeddings_path)
                    self.glob_structure_anchor_hawkes_process_embeds = torch.load(hawkes_glob_struc_patch_embeddings_path)
                    
                else:
                    raise FileNotFoundError(" >>> Hawkes embeddings not found. Please generate them during training <<< ")


                
    def aggregate_structure_anchor_patch_embeddings(self, structure_anchor_hawkes_process):
        
        """
        Aggregates node embeddings on structure anchor patch sequences generated by the Hawkes process to produce one embedding for each structure anchor patch.

        Args:
            - structure_anchor_hawkes_process(Tensor): local or global sampled structure anchor patches genrated by Hawkes process of shape
            (n_ap, n_seq_per_ap, seq_len).

        Returns:
            - struc_anchor_patch_embeds (Tensor): Aggregated embeddings for each anchor patch (n_ap, embed_dim).

        """
        
        # Retrieve embeddings for all nodes in the anchor patch sequences
        seq_embeds = self.hawkes_node_embeddings[structure_anchor_hawkes_process.to(self.device).long()]  # Shape: (n_ap, n_seq_per_ap, seq_len, embed_dim)
        structure_anchor_mask = (structure_anchor_hawkes_process != config.PAD_VALUE).unsqueeze(-1)

        # Aggregate embeddings across each sequence in the patch
        if self.hyperparameters['hawkes_seq_embedding_aggregation'] == "sum":
            seq_agg = torch.sum(seq_embeds, dim=2)  # Sum across sequence nodes
            
        elif self.hyperparameters['hawkes_seq_embedding_aggregation'] == "max":
            seq_agg, _ = torch.max(seq_embeds, dim=2)  # Max pooling across sequence nodes
            
        # Aggregate across all sequences in the anchor patch 
        struc_anchor_patch_embeds = seq_agg.sum(dim=1) / structure_anchor_mask.sum(dim=(1, 2))  # Normalize by total valid nodes 
        
        return struc_anchor_patch_embeds


                
# =============================
# Training, validation, and test steps
# =============================

    def on_train_epoch_start(self):
        
        """
        Reset storage for training outputs at the start of the epoch.
        """

        print("\n\n>>> Start training epoch <<<", flush=True)
        
        self.train_step_outputs = []  

        
 
    def training_step(self, train_batch, batch_idx):
        
        """
        Runs a single training step over a batch.
        """
                
        print(f"\n\n>>> Start training step for batch {batch_idx + 1} <<<", flush=True)
        
        # Extract batch data
        DDI_edges = train_batch['DDI_edges']
        DDI_edge_labels = train_batch['DDI_edge_labels'].squeeze(-1)
        subgraph_idx = train_batch['subgraph_idx']
        subgraph_ids = train_batch['subgraph_ids']
        cc_ids = train_batch['cc_ids']
        
        # Extract similarities
        loc_pos_head_sim = train_batch['loc_pos_head_sim']
        glob_pos_head_sim = train_batch['glob_pos_head_sim']
        loc_neigh_head_sim = train_batch['loc_neigh_head_sim']
        glob_neigh_head_sim = train_batch['glob_neigh_head_sim']
        loc_struct_head_sim = train_batch['loc_struct_head_sim']
        glob_struct_head_sim = train_batch['glob_struct_head_sim']
        
        # Forward pass
        subgraph_embeddings = self.forward(batch_idx, 'train', DDI_edges, DDI_edge_labels, subgraph_idx, subgraph_ids, cc_ids, loc_pos_head_sim,
                                           glob_pos_head_sim, loc_neigh_head_sim, glob_neigh_head_sim, loc_struct_head_sim, glob_struct_head_sim)

        # Get embedding for each pair of subgraphs in DDI_edges
        edge_pos_in_subgraph = torch.searchsorted(subgraph_idx, DDI_edges)
        subgraph_emb1 = subgraph_embeddings[edge_pos_in_subgraph[:, 0]]
        subgraph_emb2 = subgraph_embeddings[edge_pos_in_subgraph[:, 1]]

        # Concatenate and pass through MLP
        combined_emb = torch.cat([subgraph_emb1, subgraph_emb2], dim=1)  
        logits = self.mlp(combined_emb).squeeze()  

        #  Compute loss and metrics
        loss = self.loss(logits, DDI_edge_labels.float())  
        accuracy = dissubformer_utils.compute_accuracy(logits, DDI_edge_labels)
        f1, average_precision = dissubformer_utils.compute_f1_ap_metrics(logits, DDI_edge_labels)

        results = {
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1': f1,
            'train_average_precision': average_precision,
            'train_logits': logits,
            'train_labels': DDI_edge_labels
            }

        # Step-level metrics (only show in progress bar)
        self.log_dict({
            'train_loss_step': loss,
            'train_accuracy_step': accuracy,
            'train_f1_step': f1,
            'train_average_precision_step': average_precision}, batch_size=self.hyperparameters['batch_size'], on_step=True, on_epoch=False, prog_bar=True)
        
        # Epoch-level (do not show in progress bar to avoid clutter)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1': f1,
            'train_average_precision': average_precision}, batch_size=self.hyperparameters['batch_size'], on_step=False, on_epoch=True, prog_bar=False)
            
        self.train_step_outputs.append(results)
        
        return {
            'loss': loss,
            'train_accuracy': accuracy,
            'train_f1': f1,
            'train_average_precision': average_precision,
            'train_logits': logits,
            'train_labels': DDI_edge_labels
            }


           
    def on_train_epoch_end(self):
        
        """
        Aggregate metrics at the end of the training epoch.
        """

        avg_loss = self.trainer.callback_metrics['train_loss']
        avg_accuracy = self.trainer.callback_metrics['train_accuracy']
        avg_f1 = self.trainer.callback_metrics['train_f1']
        avg_average_precision = self.trainer.callback_metrics['train_average_precision']

        print("\n\n>>> Train epoch results <<<", flush=True)
        print(f"Train epoch loss: {avg_loss:.4f}")
        print(f"Train epoch accuracy: {avg_accuracy:.4f}")
        print(f"Train epoch F1: {avg_f1:.4f}")
        print(f"Train epoch average precision: {avg_average_precision:.4f}")

        # Compute AUROC for the entire epoch
        logits = torch.cat([x['train_logits'] for x in self.train_step_outputs], dim=0)
        labels = torch.cat([x['train_labels'] for x in self.train_step_outputs], dim=0)
        auroc = roc_auc_score(labels.cpu().detach().numpy(), logits.cpu().detach().numpy())
        print(f"Train epoch AUROC: {auroc:.4f}")
        tensorboard_logs = {'train_auroc': auroc}
        self.log_dict(tensorboard_logs, prog_bar=False)
        
        self.train_step_outputs.clear()

        print(">>> End training epoch <<< \n")
        
        return tensorboard_logs



    def on_validation_epoch_start(self):
        
        """
        Reset storage for validation outputs at the start of the epoch.
        """
        print("\n\n>>> Start validation epoch <<<", flush=True)
        
        self.val_step_outputs = [] 



    def validation_step(self, val_batch, batch_idx):
        
        """
        Runs a single validation step over a batch.
        """
        
        print(f"\n>>> Start validation step for batch {batch_idx + 1} <<<")
        
        # Extract validation data
        DDI_edges = val_batch['DDI_edges']
        DDI_edge_labels = val_batch['DDI_edge_labels'].squeeze(-1)
        subgraph_idx = val_batch['subgraph_idx']
        subgraph_ids = val_batch['subgraph_ids']
        cc_ids = val_batch['cc_ids']

        # Extract similarities
        loc_pos_head_sim = val_batch['loc_pos_head_sim']
        glob_pos_head_sim = val_batch['glob_pos_head_sim']
        loc_neigh_head_sim = val_batch['loc_neigh_head_sim']
        glob_neigh_head_sim = val_batch['glob_neigh_head_sim']
        loc_struct_head_sim = val_batch['loc_struct_head_sim']
        glob_struct_head_sim = val_batch['glob_struct_head_sim']

        # Forward pass
        subgraph_embeddings = self.forward(batch_idx, 'val', DDI_edges, DDI_edge_labels, subgraph_idx, subgraph_ids, cc_ids, loc_pos_head_sim,
                                           glob_pos_head_sim, loc_neigh_head_sim, glob_neigh_head_sim, loc_struct_head_sim, glob_struct_head_sim) 

        # Get embedding for each pair of subgraphs in DDI_edges
        edge_pos_in_subgraph = torch.searchsorted(subgraph_idx, DDI_edges)
        subgraph_emb1 = subgraph_embeddings[edge_pos_in_subgraph[:, 0]]
        subgraph_emb2 = subgraph_embeddings[edge_pos_in_subgraph[:, 1]]

        # Concatenate and pass through MLP
        combined_emb = torch.cat([subgraph_emb1, subgraph_emb2], dim=1)
        logits = self.mlp(combined_emb).squeeze()

        #  Compute loss and metrics
        loss = self.loss(logits, DDI_edge_labels.float()) 
        accuracy = dissubformer_utils.compute_accuracy(logits, DDI_edge_labels)
        f1, average_precision = dissubformer_utils.compute_f1_ap_metrics(logits, DDI_edge_labels)
        
        results = {
                'val_logits': logits,
                'val_labels': DDI_edge_labels
                }
        
        # Log metrics
        self.log_dict({
            'val_loss': loss, 
            'val_accuracy': accuracy,
            'val_f1': f1,
            'val_average_precision': average_precision}, batch_size=self.hyperparameters['batch_size'], on_step=True, on_epoch=True, prog_bar=False)
        
        self.val_step_outputs.append(results)
        
        return results


       
    def on_test_epoch_start(self):

        """
        Reset storage for test outputs at the start of the epoch.
        """

        print("\n\n>>> Start testing epoch <<<", flush=True)
        
        self.test_step_outputs = []  


    
    def test_step(self, test_batch, batch_idx):

        """
        Runs a single test step over a batch.

        """
        
        print(f"\n\n>>> Start testing step for batch {batch_idx + 1} <<<", flush=True)
        
        # Extract test data
        DDI_edges = test_batch['DDI_edges']
        DDI_edge_labels = test_batch['DDI_edge_labels'].squeeze(-1)
        subgraph_idx = test_batch['subgraph_idx']
        subgraph_ids = test_batch['subgraph_ids']
        cc_ids = test_batch['cc_ids']

        # Extract similarities
        loc_pos_head_sim = test_batch['loc_pos_head_sim']
        glob_pos_head_sim = test_batch['glob_pos_head_sim']
        loc_neigh_head_sim = test_batch['loc_neigh_head_sim']
        glob_neigh_head_sim = test_batch['glob_neigh_head_sim']
        loc_struct_head_sim = test_batch['loc_struct_head_sim']
        glob_struct_head_sim = test_batch['glob_struct_head_sim']

        # Forward pass
        subgraph_embeddings = self.forward(batch_idx, 'test', DDI_edges, DDI_edge_labels, subgraph_idx, subgraph_ids, cc_ids, loc_pos_head_sim,
                                           glob_pos_head_sim, loc_neigh_head_sim, glob_neigh_head_sim, loc_struct_head_sim, glob_struct_head_sim)

        # Get embedding for each pair of subgraphs in DDI_edges
        edge_pos_in_subgraph = torch.searchsorted(subgraph_idx, DDI_edges) # bec edge list id is not not cont
        subgraph_emb1 = subgraph_embeddings[edge_pos_in_subgraph[:, 0]]
        subgraph_emb2 = subgraph_embeddings[edge_pos_in_subgraph[:, 1]]

        # Concatenate and pass through MLP
        combined_emb = torch.cat([subgraph_emb1, subgraph_emb2], dim=1)  
        logits = self.mlp(combined_emb).squeeze()  
        
         # Compute loss and metrics
        loss = self.loss(logits, DDI_edge_labels.float()) 
        accuracy = dissubformer_utils.compute_accuracy(logits, DDI_edge_labels)
        f1, average_precision = dissubformer_utils.compute_f1_ap_metrics(logits, DDI_edge_labels)
        
        results = {
                'test_logits': logits,
                'test_labels': DDI_edge_labels
                }

        # Log metrics
        self.log_dict({
            'test_loss': loss, 
            'test_accuracy': accuracy,
            'test_f1': f1,
            'test_average_precision': average_precision}, batch_size=self.hyperparameters['batch_size'], on_step=True, on_epoch=True, prog_bar=False)

        self.test_step_outputs.append(results)
        
        return results



# =============================
# Validation, and test epoch end
# =============================

    def on_validation_epoch_end(self):
        
        """
        Called at the end of the validation epoch to aggregate metrics and log them.
        """
        
        avg_loss = self.trainer.callback_metrics['val_loss']
        avg_accuracy = self.trainer.callback_metrics['val_accuracy']
        avg_f1 = self.trainer.callback_metrics['val_f1']
        avg_average_precision = self.trainer.callback_metrics['val_average_precision']

        print("\n\n>>> Validation epoch results <<<", flush=True)
        print(f"Validation epoch loss: {avg_loss:.4f}")
        print(f"Validation epoch accuracy: {avg_accuracy:.4f}")
        print(f"Validation epoch F1: {avg_f1:.4f}")
        print(f"Validation epoch average precision: {avg_average_precision:.4f}")

        # Compute AUROC for the entire epoch
        logits = torch.cat([x['val_logits'] for x in self.val_step_outputs], dim=0)
        labels = torch.cat([x['val_labels'] for x in self.val_step_outputs], dim=0)
        auroc = roc_auc_score(labels.cpu(), logits.cpu())  
        print(f"Validation epoch AUROC: {auroc:.4f}")
        self.log_dict({'val_auroc': auroc}, prog_bar=False)
        
        val_epoch_metrics = {
            'epoch': self.current_epoch, 
            'val_loss': avg_loss.item(),
            'val_accuracy': avg_accuracy.item(),
            'val_f1': avg_f1.item(),
            'val_average_precision': avg_average_precision.item(),
            'val_auroc': auroc,
            }
        
        self.val_metric_scores.append(val_epoch_metrics)
        
        self.val_step_outputs.clear()

        print(">>> End validation epoch <<< \n")


        
    def on_test_epoch_end(self):

        """
        Called at the end of the test epoch to aggregate metrics and log them.
        """
        
        avg_loss = self.trainer.callback_metrics['test_loss']
        avg_accuracy = self.trainer.callback_metrics['test_accuracy']
        avg_f1 = self.trainer.callback_metrics['test_f1']
        avg_average_precision = self.trainer.callback_metrics['test_average_precision']

        # Compute AUROC for the entire epoch
        logits = torch.cat([x['test_logits'] for x in self.test_step_outputs], dim=0)
        labels = torch.cat([x['test_labels'] for x in self.test_step_outputs], dim=0)
        auroc = roc_auc_score(labels.cpu(), logits.cpu())
        self.log_dict({'test_auroc_epoch': auroc}, prog_bar=False)

        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(labels.cpu().numpy(), logits.cpu().numpy())
       
        # Precision-Recall Curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels.cpu().numpy(), logits.cpu().numpy())

        self.test_epoch_metrics = {
            'epoch': self.current_epoch, 
            'test_loss': avg_loss.item(),
            'test_accuracy': avg_accuracy.item(),
            'test_f1': avg_f1.item(),
            'test_average_precision': avg_average_precision.item(),
            'test_auroc': auroc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_thresholds': roc_thresholds.tolist(),
            'precision_curve': precision_curve.tolist(),
            'recall_curve': recall_curve.tolist(),
            'pr_thresholds': pr_thresholds.tolist(),
            'logits': logits,
           'labels': labels,
            }

        self.test_step_outputs.clear()

        return self.test_epoch_metrics


    
# =============================
# Read data 
# =============================

    def reindex_data(self, data):
        
        """
        Relabel node indices in the data to be 1-indexed instead of 0-indexed.
        
        Handles both lists and dictionaries.
            - For lists: Each element is assumed to be a list of node indices.
            - For dictionaries: Each value is assumed to be a list of node indices.
        """
        
        if isinstance(data, dict):
            new_subgraph = {}
            for key, subgraph_nodes in data.items():
                new_subgraph[key] = [[node + 1 for node in subgraph] for subgraph in subgraph_nodes]
                
            return new_subgraph
        
        elif isinstance(data, list):
            new_subgraph = []
            for subgraph in data:
                new_subgraph.append([node + 1 for node in subgraph])
                
            return new_subgraph
        


    def read_data(self):
        
        """
        Read in the PPI and DDI graphs, subgraphs, and initialize embeddings.
        """

        # Read the PPI graph from the edge list
        self.PPI_graph = nx.read_edgelist(config.PROJECT_ROOT / self.PPI_graph_path)

        # Read and split the DDI graph into train, validation, and test sets
        self.train_DDI_edges, self.train_DDI_edges_label, self.val_DDI_edges, self.val_DDI_edges_label, self.test_DDI_edges, self.test_DDI_edges_label = dissubformer_utils.read_DDI_graph(
                                                                                                    config.PROJECT_ROOT /self.Subgraphs_path, config.PROJECT_ROOT / self.DDI_RR0_graph_path)
        
        self.all_subgraphs, self.train_subgraphs, self.val_subgraphs, self.test_subgraphs, self.train_subgraphs_dict, self.val_subgraphs_dict, self.test_subgraphs_dict,\
                    self.train_subgraphs_indices, self.val_subgraphs_indices, self.test_subgraphs_indices = dissubformer_utils.assign_subgraphs_to_splits(
                       config.PROJECT_ROOT /self.Subgraphs_path, self.train_DDI_edges, self.val_DDI_edges, self.test_DDI_edges)
           
        # Renumber nodes to start at index 1 instead of 0
        mapping = {n:int(n)+1 for n in self.PPI_graph.nodes()}
        self.PPI_graph = nx.relabel_nodes(self.PPI_graph, mapping)

        # Reindex all subgraphs accordingly
        self.train_subgraphs = self.reindex_data(self.train_subgraphs)
        self.val_subgraphs = self.reindex_data(self.val_subgraphs)
        self.test_subgraphs = self.reindex_data(self.test_subgraphs)
        self.all_subgraphs = self.reindex_data(self.all_subgraphs)
        self.train_subgraphs_dict = self.reindex_data(self.train_subgraphs_dict)
        self.val_subgraphs_dict = self.reindex_data(self.val_subgraphs_dict)
        self.test_subgraphs_dict = self.reindex_data(self.test_subgraphs_dict)

        # Load pretrained node embeddings learned from PPI and GO_FS training
        pretrained_node_embeds = torch.load(config.PROJECT_ROOT / self.Embedding_path, map_location=torch.device('cpu'), weights_only=True)
        zeros = torch.zeros(1, pretrained_node_embeds.shape[1])
        embeds = torch.cat((zeros, pretrained_node_embeds), 0)
        
        # optionally freeze the node embeddings
        self.node_embeddings = nn.Embedding.from_pretrained(embeds, freeze=self.hyperparameters['freeze_node_embeds'], padding_idx=config.PAD_VALUE).to(self.device)
        
        print(" >>> Finished reading data <<< ")


        
# =============================
# Initialize connected components & associated embeddings
# =============================

    def initialize_cc_ids(self, subgraph_ids):
        
        """
        Initialize the 3D matrix of shape (n_subgraphs, max_n_cc, max_len_cc).

        Args:
            - subgraph_ids (list): list of subgraphs where each subgraph is a list of node ids.

        Returns:
            - reshaped_cc_ids_pad (Tensor): padded tensor of shape (n_subgraphs, max_n_cc, max_len_cc)
        """

        n_subgraphs = len(subgraph_ids) 
        cc_id_list = [] 
        
        for curr_subgraph_ids in subgraph_ids:
            subgraph = nx.subgraph(self.PPI_graph, curr_subgraph_ids) # networkx version of subgraph
            con_components = list(nx.connected_components(subgraph)) # get connected components in subgraph
            cc_id_list.append([torch.LongTensor(list(cc_ids)) for cc_ids in con_components])

        # pad number of connected components
        max_n_cc = max([len(cc) for cc in cc_id_list]) # max number of cc across all subgraphs
        for cc_list in cc_id_list:
            while True:
                if len(cc_list) == max_n_cc: break
                cc_list.append(torch.LongTensor([config.PAD_VALUE]))

        # pad number of nodes in connected components
        all_pad_cc_ids = [cc for cc_list in cc_id_list for cc in cc_list]
        assert len(all_pad_cc_ids) % max_n_cc == 0
        con_component_ids_pad = pad_sequence(all_pad_cc_ids, batch_first=True, padding_value=config.PAD_VALUE) 
        reshaped_cc_ids_pad = con_component_ids_pad.view(n_subgraphs, max_n_cc, -1)
        
        return reshaped_cc_ids_pad # (n_subgraphs, max_n_cc, max_len_cc)


    
    def init_cc_embeddings(self, cc_id_list, aggregator):

        """
        Initialize connected component embeddings as either the sum or max of node embeddings in the connected component.

        Args:
            - cc_id_list (Tensor): Tensor of shape (n_subgraphs, max_n_cc, max_len_cc)
            - aggregator (str): Aggregation method: 'sum' or 'max'

        Returns:
            - Tensor of shape (n_subgraphs, max_n_cc, node_embedding_dim)
        """
        
        if aggregator == 'sum':
            return torch.sum(self.node_embeddings(cc_id_list.to(self.device)), dim=2)
        elif aggregator == 'max':
            return torch.max(self.node_embeddings(cc_id_list.to(self.device)), dim=2)[0]


        
# =============================
# Initialize global sets around CCs for each subgraph
# =============================

    def initialize_global_sets(self, file_name, cc_ids, radius, ego_graph_dict=None):
        
        """
        Create and store a matrix of k-hop global node IDs for each connected component (CC) in each subgraph.
        These global sets are used for sampling neighborhood-based anchor patches used with neighborhood head.
        
        Returns:
            - global_set_matrix (Tensor): Padded tensor of shape (n_subgraphs, max_n_cc, max_global_set_sz).

        """
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        all_global_sets = []
        
        # Compute k-hop global sets for each component in each subgraph
        for s, subgraph in enumerate(cc_ids):
            global_sets = []
            for c, component in enumerate(subgraph):
                component_global = dissubformer_utils.get_component_global_neighborhood_set(self.PPI_graph, component, radius, ego_graph_dict)
                global_sets.append(component_global)
            all_global_sets.append(global_sets)

        # Pad global sets 
        max_global_set_len = max([len(s) for l in all_global_sets for s in l])
        global_set_matrix = torch.zeros((n_subgraphs, max_n_cc, max_global_set_len), dtype=torch.long).fill_(config.PAD_VALUE)
        for s, subgraph in enumerate(all_global_sets):
            for c,component in enumerate(subgraph):
                fill_len = max_global_set_len - len(component)
                global_set_matrix[s,c,:] = torch.cat([torch.LongTensor(list(component)),torch.LongTensor((fill_len)).fill_(config.PAD_VALUE)])
        
        # Save matrix to file 
        np.save(file_name, global_set_matrix.cpu().numpy())
        return global_set_matrix # (n_subgraphs, max_n_cc, max_glob_set_sz)

    

    def get_global_sets(self, split):
        
        """ 
        Retrieve or compute k-hop global sets for connected components in subgraphs, used for sampling neighborhood-based anchor patches.
        """

        AP_sampling_similarities_path = config.PROJECT_ROOT / self.AP_sampling_similarities_path 
        
        if self.hyperparameters['neighborhood_head']:
            # Load or prepare ego-graph dictionary
            Ego_graph_path = config.PROJECT_ROOT / self.Ego_graph_path 
            if Ego_graph_path.exists():
                with open(str(Ego_graph_path), 'r') as f:
                    ego_graph_dict = json.load(f)
                ego_graph_dict = {int(key): value for key, value in ego_graph_dict.items()}
            else: ego_graph_dict = None

            # Define paths to global set files
            train_neigh_glob_set_path = AP_sampling_similarities_path / (str(self.hyperparameters["neigh_sample_global_size"])+ '_train_glob_set.npy') 
            val_neigh_glob_set_path = AP_sampling_similarities_path / (str(self.hyperparameters["neigh_sample_global_size"]) + '_val_glob_set.npy') 
            test_neigh_glob_set_path = AP_sampling_similarities_path / (str(self.hyperparameters["neigh_sample_global_size"]) + '_test_glob_set.npy')
            
            if split == 'train_val':
                if train_neigh_glob_set_path.exists(): 
                    self.train_neigh_glob_set = torch.tensor(np.load(train_neigh_glob_set_path, allow_pickle=True))
                else:
                    self.train_neigh_glob_set = self.initialize_global_sets(train_neigh_glob_set_path, self.train_cc_ids, self.hyperparameters["neigh_sample_global_size"], ego_graph_dict)

                if val_neigh_glob_set_path.exists(): 
                    self.val_neigh_glob_set = torch.tensor(np.load(val_neigh_glob_set_path, allow_pickle=True))
                else:
                    self.val_neigh_glob_set = self.initialize_global_sets(val_neigh_glob_set_path, self.val_cc_ids, self.hyperparameters
                                                                          ["neigh_sample_global_size"], ego_graph_dict)

            elif split == 'test':
                if test_neigh_glob_set_path.exists(): 
                    self.test_neigh_glob_set = torch.tensor(np.load(test_neigh_glob_set_path, allow_pickle=True))
                else:
                    self.test_neigh_glob_set = self.initialize_global_sets(test_neigh_glob_set_path, self.test_cc_ids, self.hyperparameters["neigh_sample_global_size"], ego_graph_dict)  

        # If the neighborhood head is not used, set global sets to None
        else: 
            self.train_neigh_glob_set = None
            self.val_neigh_glob_set = None
            self.test_neigh_glob_set = None



# =============================
# Precompute sampling similarities for APs
# =============================

    def precompute_euclidean_dist_similarities (self, file_name_cc, file_name_supgraph, eucli_dis, cc_ids):
        
        """
        Precompute and save Euclidean distance-based similarity matrices.

        Returns:
            - dis_similarities_cc (Tensor): Tensor of shape (n_subgraphs, max_n_cc, n_nodes_in_graph) representing the similarity of each CC to all nodes in the graph.
            - dis_similarities_supgraph (Tensor): Tensor of shape (n_subgraphs, n_nodes_in_graph) representing the similarity of each subgraph to all nodes in the graph.
        """
        
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        n_nodes_in_graph = len(eucli_dis)
        
        self.dis_similarities_cc = torch.zeros((n_subgraphs, max_n_cc, n_nodes_in_graph)).fill_(config.PAD_VALUE)
        self.dis_similarities_supgraph = torch.zeros((n_subgraphs, n_nodes_in_graph)).fill_(config.PAD_VALUE)
        
        # For each subgraph, compute the similarity between each connected component and the subgraph with all nodes in the graph
        for s, subgraph in enumerate(cc_ids):
            total_subgraph_distances = np.zeros(n_nodes_in_graph)  
            count_cc = 0  
            for c, component in enumerate(subgraph):
                non_padded_component = component[component != config.PAD_VALUE].cpu().numpy()  # Remove padding
                if len(non_padded_component) > 0:
                    # Retrieve distances from each node in the connected component to all other nodes using the Euclidean distance matrix 
                    distances_from_cc = eucli_dis[non_padded_component - 1]  
                   
                    # Compute the mean distance for the connected component to all nodes
                    mean_distances = np.mean(distances_from_cc, axis=0)  
                    
                    # Compute similarity scores as normalized inverse distances
                    weights = 1 / (mean_distances + 1e-10) 
                    weights /= weights.sum()  
                    
                    self.dis_similarities_cc[s, c, :] = torch.tensor(weights)
                    # Aggregate similarity scores from all connected components to compute the subgraph-level similarity to all nodes
                    total_subgraph_distances += weights
                    count_cc += 1
                else:
                    self.dis_similarities_cc[s, c, :] = torch.zeros(n_nodes_in_graph)
                    
            if count_cc > 0:
                all_cc_weights = total_subgraph_distances / count_cc  
                self.dis_similarities_supgraph[s, :] = torch.tensor(all_cc_weights)

        # Save to files 
        if not file_name_cc.parent.exists():
            file_name_cc.parent.mkdir(parents=True)

        if not file_name_supgraph.parent.exists():
            file_name_supgraph.parent.mkdir(parents=True)

        np.save(file_name_cc, self.dis_similarities_cc.cpu().numpy())
        np.save(file_name_supgraph, self.dis_similarities_supgraph.cpu().numpy())

        return self.dis_similarities_cc, self.dis_similarities_supgraph


    
    def compute_ap_sampling_similarities(self, split):

        """
        Compute all necessary similarities for anchor patch sampling.
        """
        
        AP_sampling_similarities_path = config.PROJECT_ROOT / self.AP_sampling_similarities_path
        pairwise_eucli_dis_path = np.load((config.PROJECT_ROOT /self.Euclidean_distances_path), allow_pickle=True)
        
        cc_to_graph_sim_matrix_path = AP_sampling_similarities_path / 'cc_to_graph_similarities.npy'
        subgraph_to_graph_sim_matrix_path = AP_sampling_similarities_path / 'subgraph_to_graph_similarities.npy'
        
        # Precompute similarities between all CCs/subgraphs and all graph nodes
        self.all_cc_ids_dis_similarities, self.all_subgraphs_dis_similarities = self.precompute_euclidean_dist_similarities(cc_to_graph_sim_matrix_path,
                                                                            subgraph_to_graph_sim_matrix_path, pairwise_eucli_dis_path, self.all_cc_ids)
        
        # ==================== Sampling neighborhood or position anchor patches ==================== #
        if self.hyperparameters['position_head'] or self.hyperparameters['neighborhood_head']:
        
            # Load precomputed similarities for CCs and subgraphs to all nodes, or compute them if not found
            # CCs with all nodes similarities
            ccs_train_similarity_path = AP_sampling_similarities_path / 'train_cc_to_graph_similarities.npy'
            ccs_val_similarity_path = AP_sampling_similarities_path / 'val_cc_to_graph_similarities.npy' 
            ccs_test_similarity_path = AP_sampling_similarities_path / 'test_cc_to_graph_similarities.npy'
            
            # Subgraphs with all nodes similarities
            subgraphs_train_similarity_path = AP_sampling_similarities_path / 'train_subgraph_to_graph_similarities.npy'
            subgraphs_val_similarity_path = AP_sampling_similarities_path / 'val_subgraph_to_graph_similarities.npy' 
            subgraphs_test_similarity_path = AP_sampling_similarities_path / 'test_subgraph_to_graph_similarities.npy'
            
            if split == 'train_val':
                if ccs_train_similarity_path.exists() and subgraphs_train_similarity_path.exists():
                    print(" >>> Loading Euclidean distance-based similarities for sampling neighborhood/position APs for train data <<< ")
                    self.train_ccs_neigh_pos_similarities = torch.tensor(np.load(ccs_train_similarity_path, allow_pickle=True))
                    self.train_subgraphs_neigh_pos_similarities = torch.tensor(np.load(subgraphs_train_similarity_path, allow_pickle=True))
                else:
                    print(" >>> Computing Euclidean distance-based similarities for sampling neighborhood/position APs for train data <<< ")
                    self.train_ccs_neigh_pos_similarities, self.train_subgraphs_neigh_pos_similarities = self.precompute_euclidean_dist_similarities(ccs_train_similarity_path,
                                                                                                   subgraphs_train_similarity_path, pairwise_eucli_dis_path, self.train_cc_ids)

                if ccs_val_similarity_path.exists() and subgraphs_val_similarity_path.exists(): 
                    print(" >>> Loading Euclidean distance-based similarities for sampling neighborhood/position APs for val data <<< ")
                    self.val_ccs_neigh_pos_similarities = torch.tensor(np.load(ccs_val_similarity_path, allow_pickle=True))
                    self.val_subgraphs_neigh_pos_similarities = torch.tensor(np.load(subgraphs_val_similarity_path, allow_pickle=True))
                else:
                    print(" >>> Computing Euclidean distance-based similarities for sampling neighborhood/position APs for val data <<< ")
                    self.val_ccs_neigh_pos_similarities, self.val_subgraphs_neigh_pos_similarities = self.precompute_euclidean_dist_similarities(ccs_val_similarity_path,
                                                                                                 subgraphs_val_similarity_path, pairwise_eucli_dis_path, self.val_cc_ids)

            elif split == 'test':
                if ccs_test_similarity_path.exists() and subgraphs_test_similarity_path.exists(): 
                    print(" >>> Loading Euclidean distance-based similarities for sampling neighborhood/position APs for test data <<< ")
                    self.test_ccs_neigh_pos_similarities = torch.tensor(np.load(ccs_test_similarity_path, allow_pickle=True))
                    self.test_subgraphs_neigh_pos_similarities = torch.tensor(np.load(subgraphs_test_similarity_path, allow_pickle=True))
                else:
                    print(" >>> Computing Euclidean distance-based similarities for sampling neighborhood/position APs for test data <<< ")
                    self.test_ccs_neigh_pos_similarities, self.test_subgraphs_neigh_pos_similarities = self.precompute_euclidean_dist_similarities(ccs_test_similarity_path,
                                                                                                  subgraphs_test_similarity_path, pairwise_eucli_dis_path, self.test_cc_ids)

        # If the structure head is only used, set these to None
        else: 
            self.train_ccs_neigh_pos_similarities = None
            self.train_subgraphs_neigh_pos_similarities = None
            self.val_ccs_neigh_pos_similarities = None
            self.val_subgraphs_neigh_pos_similarities = None
            self.test_ccs_neigh_pos_similarities = None
            self.test_subgraphs_neigh_pos_similarities = None
            
        # ==================== Sampling structure anchor patches ==================== #
        if self.hyperparameters['structure_head']:
            
            # (1) Sample structure anchor patches
            
            struc_anchor_patches_dic_path = AP_sampling_similarities_path / ('struc_patches_dic_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npz')                                      
            struc_anchor_patches_path = AP_sampling_similarities_path / ('struc_patches_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')
                            
            if struc_anchor_patches_path.exists() and struc_anchor_patches_dic_path.exists():
                print(" >>> Loading sampled structure anchor patches <<< ")
                loaded_structure_anchors_dic = np.load(struc_anchor_patches_dic_path, allow_pickle=True)
                self.structure_anchors_dic = {int(start_node): torch.tensor(patch) for start_node, patch in loaded_structure_anchors_dic.items()}
                self.structure_anchors = torch.tensor(np.load(struc_anchor_patches_path, allow_pickle=True))
               
            else: 
                print(" >>> Sampling structure anchor patches <<< ")
                self.structure_anchors_dic , self.structure_anchors = sample_structure_anchor_patches(self.hyperparameters, self.PPI_graph,
                                                                                          self.all_subgraphs_dis_similarities, self.device)
                structure_anchors_dic_to_save = {str(start_node): patch.cpu().numpy() for start_node, patch in self.structure_anchors_dic.items()}
                np.savez(struc_anchor_patches_dic_path, **structure_anchors_dic_to_save)
                np.save(struc_anchor_patches_path, self.structure_anchors.cpu().numpy())
             
            # (2) Perform local and global Hawkes process on sampled structure anchor patches
            
            # local Hawkes process
            loc_struc_patch_hawkes_process_path = AP_sampling_similarities_path / ('loc_struc_patch_hawkes_process_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')                  
            if loc_struc_patch_hawkes_process_path.exists():
                print(" >>> Loading sampled local structure anchor patches <<< ")
                self.loc_structure_anchor_hawkes_process = torch.tensor(np.load(loc_struc_patch_hawkes_process_path, allow_pickle=True))
            else:
                print(" >>> Sampling local structure anchor patches <<< ")
                self.loc_structure_anchor_hawkes_process = perform_hawkes_process(self.hyperparameters, self.PPI_graph, self.structure_anchors_dic,
                                                                       pairwise_eucli_dis_path, self.all_subgraphs_dis_similarities, inside = True)
                np.save(loc_struc_patch_hawkes_process_path, self.loc_structure_anchor_hawkes_process.cpu().numpy())
 
            # global Hawkes process
            glob_struc_patch_hawkes_process_path = AP_sampling_similarities_path / ('glob_struc_patch_hawkes_proces_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) +  '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')                                          
            if glob_struc_patch_hawkes_process_path.exists():
                print(" >>> Loading sampled global structure anchor patches <<<")
                self.glob_structure_anchor_hawkes_process = torch.tensor(np.load(glob_struc_patch_hawkes_process_path, allow_pickle=True))
            else:
                print(" >>> Sampling global structure anchor patches <<< ")
                self.glob_structure_anchor_hawkes_process = perform_hawkes_process(self.hyperparameters, self.PPI_graph, self.structure_anchors_dic,
                                                                       pairwise_eucli_dis_path, self.all_subgraphs_dis_similarities, inside = False)
                np.save(glob_struc_patch_hawkes_process_path, self.glob_structure_anchor_hawkes_process.cpu().numpy())


                
# =============================
# Precompute head attention relational term similarities
# =============================

    def precompute_shortest_path_similarities(self, file_name, node_pairwise_shortest_paths, cc_ids):
        
        """
        Precompute the shortest path similarities between connected components (in each subgraph) and all nodes in the graph to use in position head attention.
        
        Returns:
            - shortest_path_similarities (Tensor):  Shortest path similarities tensor of shape (n_subgraphs, max_n_cc, n_nodes_in_graph).
        """

        n_subgraphs, max_n_cc, _ = cc_ids.shape
        n_nodes_in_graph = len(self.PPI_graph.nodes()) 

        cc_id_mask = (cc_ids[:,:,0] != config.PAD_VALUE)
        shortest_path_similarities = torch.zeros((n_subgraphs, max_n_cc, n_nodes_in_graph)).fill_(config.PAD_VALUE)

        # For each subgraph, compute the shortest path similarity between each connected component with all nodes in the graph
        for subgraph_idx, subgraph in enumerate(cc_ids):
            for component_idx, component in enumerate(subgraph):
                non_padded_component = component[component != config.PAD_VALUE].cpu().numpy() # remove padding
                if len(non_padded_component) > 0:
                    shortest_path_similarities[subgraph_idx,component_idx,:] = torch.tensor(np.min(node_pairwise_shortest_paths[non_padded_component - 1,:], axis=0))

        # Save to files 
        if not file_name.parent.exists(): file_name.parent.mkdir(parents=True)
        np.save(file_name, shortest_path_similarities.cpu().numpy())
        
        return shortest_path_similarities
    

        
    def precompute_local_position_head_similarities(self, shortest_path_similarities, cc_ids, loc_position_anchor_patches):
        
        """
        Precompute the local position-based similarity between connected components and local position anchor patches.
        
        Returns:
            - loc_pos_head_similarities (Tensor): Dictionary mapping each layer to a tensor of shape (n_subgraphs, max_n_cc, n_loc_pos_anchors) containing similarity scores.
        """
        
        loc_pos_head_similarities = defaultdict(torch.Tensor)
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        
        # Process each layer in local postion anchor patches
        for layer, loc_pos_anchor in loc_position_anchor_patches.items():
            
            # Repeat local anchors for each CC in each subgraph
            loc_pos_anchor = loc_pos_anchor.unsqueeze(1).repeat(1, max_n_cc, 1)
            loc_pos_n_anchors = loc_pos_anchor.size(-1)
            loc_pos_head_rel_values = torch.zeros(n_subgraphs, max_n_cc, loc_pos_n_anchors)
            
            # Compute positional relational term similarities between CCs and local anchor patches
            valid_loc_pos_anchor_mask = loc_pos_anchor != config.PAD_VALUE
            for i in range(n_subgraphs):
                for j in range(max_n_cc):
                    component = cc_ids[i, j]
                    if torch.all(component == config.PAD_VALUE):
                        continue
                    valid_loc_pos_anchor_nodes = loc_pos_anchor[i, j, valid_loc_pos_anchor_mask[i, j]] - 1
                    cc_ap_shortest_path = shortest_path_similarities[i, j, valid_loc_pos_anchor_nodes]
                    cc_ap_relat_value = (self.hyperparameters["pos_head_scaling_factor"]
                        * torch.exp(-self.hyperparameters["pos_head_decay_rate"] * cc_ap_shortest_path))
                    loc_pos_head_rel_values[i, j, : len(cc_ap_relat_value)] = cc_ap_relat_value

            # Add the layer results to the dictionary
            loc_pos_head_similarities[layer] = loc_pos_head_rel_values

        return loc_pos_head_similarities


    
    def precompute_global_position_head_similarities(self, shortest_path_similarities, cc_ids, glob_position_anchor_patches):
        
        """
        Precompute the global position-based similarity between connected components and global position anchor patches.
        
        Returns:
            - glob_pos_head_similarities (Tensor): Dictionary mapping each layer to a tensor of shape (n_subgraphs, max_n_cc, n_glob_pos_anchors) containing similarity scores.
        """
        
        glob_pos_head_similarities = defaultdict(torch.Tensor)
        n_subgraphs, max_n_cc, _ = cc_ids.shape

        # Process each layer in global position anchor patches
        for layer, glob_pos_anchor in glob_position_anchor_patches.items():

            # Repeat global anchors for each CC in each subgraph
            glob_pos_anchor = glob_pos_anchor.unsqueeze(0).unsqueeze(0).repeat(n_subgraphs, max_n_cc, 1)
            glob_pos_n_anchors = glob_pos_anchor.size(-1)
            glob_pos_head_rel_values = torch.zeros(n_subgraphs, max_n_cc, glob_pos_n_anchors)

            # Compute positional relational term similarities between CCs and global anchor patches
            for i in range(n_subgraphs): 
                for j in range(max_n_cc):  
                    component = cc_ids[i, j]
                    if torch.all(component == config.PAD_VALUE):
                        continue
                    glob_anchor_nodes = glob_pos_anchor[i, j] - 1
                    cc_ap_shortest_path = shortest_path_similarities[i, j, glob_anchor_nodes]
                    cc_ap_relat_value = (self.hyperparameters["pos_head_scaling_factor"]
                        * torch.exp(-self.hyperparameters["pos_head_decay_rate"] * cc_ap_shortest_path))
                    glob_pos_head_rel_values[i, j, : len(cc_ap_relat_value)] = cc_ap_relat_value

            # Add the layer results to the dictionary
            glob_pos_head_similarities[layer] = glob_pos_head_rel_values

        return glob_pos_head_similarities
    

    
    def precompute_intermediate_node_similarities(self, file_name, pairwise_intermediate_nodes, cc_ids):
        
        """
        Precompute the number of intermediate nodes between connected components (in each subgraph) and all nodes in the graph to used in neighborhood head attention.

        Returns:
            - intermediate_nodes_similarities (Tensor): Tensor of shape (n_subgraphs, max_n_cc, n_nodes_in_graph) containing the similarity values.
        """

        n_subgraphs, max_n_cc, _ = cc_ids.shape
        n_nodes_in_graph = len(self.PPI_graph.nodes())

        cc_id_mask = (cc_ids[:,:,0] != config.PAD_VALUE)
        intermediate_nodes_similarities = torch.zeros((n_subgraphs, max_n_cc, n_nodes_in_graph)).fill_(config.PAD_VALUE)

        # For each subgraph, compute the minimum number of intermediate nodes between each connected component and all graph nodes
        for subgraph_idx, subgraph in enumerate(cc_ids):
            for component_idx, component in enumerate(subgraph):
                non_padded_component = component[component != config.PAD_VALUE].cpu().numpy() #remove padding
                if len(non_padded_component) > 0:
                    intermediate_nodes_similarities[subgraph_idx,component_idx,:] = torch.tensor(np.min(pairwise_intermediate_nodes[non_padded_component - 1,:], axis=0))

        # Save to files 
        if not file_name.parent.exists(): file_name.parent.mkdir(parents=True)
        np.save(file_name, intermediate_nodes_similarities.cpu().numpy())
        
        return intermediate_nodes_similarities


        
    def precompute_neighborhood_head_similarities(self, intermediate_nodes_similarities, cc_ids, neighborhood_anchor_patches):
     
        """
        Precompute the neighborhood-based similarity between connected components and neighborhood anchor patches.

        Returns:
            - nei_head_similarities (Tensor): Dictionary mapping each layer to a tensor of shape (n_subgraphs, max_n_cc, n_neig_anchors) containing similarity scores.
        """
        
        nei_head_similarities = defaultdict(lambda: torch.Tensor())
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        
        # Process each layer in neighborhood position anchor patches        
        for layer, neig_anchor in neighborhood_anchor_patches.items():
            nei_n_anchors = neig_anchor.shape[2]
            nei_head_rel_values = torch.zeros((n_subgraphs, max_n_cc, nei_n_anchors))
            valid_nei_anchor_mask = neig_anchor != config.PAD_VALUE

            # Compute neighborhood-based relational term similarities between CCs and neighborhood anchor patches
            for i in range(n_subgraphs):
                for j in range(max_n_cc):
                    component = cc_ids[i, j]
                    if torch.all(component == config.PAD_VALUE):
                        continue
                    valid_nei_anchor_nodes = neig_anchor[i, j, valid_nei_anchor_mask[i, j]] - 1
                    cc_ap_intermediate_nodes = intermediate_nodes_similarities[i, j, valid_nei_anchor_nodes]
                    cc_ap_relat_value = torch.exp(-cc_ap_intermediate_nodes / self.hyperparameters['neigh_head_decay_rate'])
                    nei_head_rel_values[i, j, : cc_ap_relat_value.size(0)] = cc_ap_relat_value

            # Add the layer results to the dictionary
            nei_head_similarities[layer] = nei_head_rel_values
    
        return nei_head_similarities



    def precompute_structure_head_similarities(self, file_name, adj_matrix, node_degrees_dic, structure_ap_ids, cc_ids):
        
        """
        Precompute the structure-based similarity between connected components and structure anchor patches.

        Returns:
            - aggregated_structure_similarities (Tensor): Tensor of shape (n_subgraphs, max_n_cc, n_struct_aps) containing similarity scores.
        """
        
        node_degrees_tensor = torch.zeros(adj_matrix.shape[0])
        for node, degree in node_degrees_dic.items():
            node_degrees_tensor[node] = degree

        # Prepare intermediate structure anchor patch (AP) nodes
        n_ap, n_ap_sequences, ap_seq_len = structure_ap_ids.shape
        max_ap_nodes = n_ap_sequences * ap_seq_len
        all_intermediate_ap = torch.zeros((n_ap, max_ap_nodes)).fill_(config.PAD_VALUE)

        for ap_idx, ap_sequences in enumerate(structure_ap_ids):
            ap_nodes = ap_sequences.flatten().unique()
            all_intermediate_ap[ap_idx, :len(ap_nodes)] = ap_nodes

        # Speed up the process of similarities between each subgraph and all intermediate structure anchor patches using multiprocessing operation
        # Prepare input for parallel processing of subgraphs 
        parallel_subgraphs = []
        for subgraph_idx, subgraph in enumerate(cc_ids):
            parallel_subgraphs.append((subgraph_idx, subgraph, structure_ap_ids, all_intermediate_ap, adj_matrix, node_degrees_tensor))
        structure_similarities = torch.zeros(cc_ids.shape[:2] + structure_ap_ids.shape[:2]).fill_(0)
        group_size = 5  # Print progress after processing every 'group_size' subgraphs
        processed_count = 0  # Track number of subgraphs processed
        print(" >>> Starting multiprocessing computation for head structure-based similarities <<< ")
        with multiprocessing.Pool(processes=self.hyperparameters['n_processes']) as pool:
            for sim_result in pool.imap(DisSubFormer.compute_subgraph_similarity, parallel_subgraphs):
                subgraph_idx, similarities = sim_result
                structure_similarities[subgraph_idx] = similarities
                processed_count += 1

                if processed_count % group_size == 0:
                    print(f" >>> Done processing {processed_count} subgraphs out of {len(parallel_subgraphs)} <<< ")

        print(" >>> Finished multiprocessing computation for structure-based similarities <<< ")

        # Average similarity scores over all valid sequences for each anchor patch
        valid_sequences = (structure_similarities != 0).sum(dim=3)
        aggregated_structure_similarities = structure_similarities.sum(dim=3) / valid_sequences.clamp(min=1)
        aggregated_structure_similarities[valid_sequences == 0] = 0
              
        # Save to file
        if not file_name.parent.exists(): file_name.parent.mkdir(parents=True)
        np.save(file_name, aggregated_structure_similarities.cpu().numpy())
        
        return aggregated_structure_similarities
    


    @staticmethod
    def compute_subgraph_similarity(subgraph_data):
        
        """
        Compute similarities between connected components and anchor patches for a single subgraph.

        Args:
            subgraph_data (tuple): Contains subgraph_idx, subgraph, structure_ap_ids, all_intermediate_ap, adj_matrix, and node degrees.

        Returns:
            tuple: (subgraph_idx, subgraph_similarities)
        """
        
        subgraph_idx, subgraph, structure_ap_ids, all_intermediate_ap, adj_matrix, node_degrees_tensor = subgraph_data
        max_n_cc, _ = subgraph.shape
        n_ap, n_ap_sequences, _ = structure_ap_ids.shape
        subgraph_similarities = torch.zeros((max_n_cc, n_ap, n_ap_sequences)).fill_(0)

        # For each connected component, compute the similarity with direct and intermediate anchor patche sequences
        for cc_idx, cc in enumerate(subgraph):
            if (cc == config.PAD_VALUE).all():
                continue
            cc_nodes = (cc[cc != config.PAD_VALUE] - 1).clamp(min=0).cpu().numpy().astype(int)
            for ap_idx, ap_sequences in enumerate(structure_ap_ids):
                for ap_seq_idx, ap_sequence in enumerate(ap_sequences):
                    if (ap_sequence == config.PAD_VALUE).all():
                        continue
                    current_ap_seq = (ap_sequence[ap_sequence != config.PAD_VALUE] - 1).clamp(min=0).cpu().numpy().astype(int)

                    # Compute direct connection with current anchor patch sequences
                    direct_sim = np.sum(adj_matrix[np.ix_(cc_nodes, current_ap_seq)])

                    # Compute intermediate connections with all other anchor patch sequences
                    intermediate_sim = 0
                    for intermediate_ap in all_intermediate_ap:
                        intermediate_ap = (intermediate_ap[intermediate_ap != config.PAD_VALUE] - 1).cpu().numpy().astype(int)
                        if len(intermediate_ap) == 0:
                            continue
                        edge_cc_to_intermediate = np.sum(adj_matrix[np.ix_(cc_nodes, intermediate_ap)])
                        edge_intermediate_to_current = np.sum(adj_matrix[np.ix_(intermediate_ap, current_ap_seq)])
                        intermediate_sim += np.sqrt(edge_cc_to_intermediate * edge_intermediate_to_current)

                    # Compute structure relational term similarities between CCs and structure anchor patch sequences
                    total_similarities = direct_sim + intermediate_sim
                    cc_degree = node_degrees_tensor[cc_nodes].sum().item()
                    ap_degree = node_degrees_tensor[current_ap_seq].sum().item()
                    normalization_factor = np.sqrt(cc_degree * ap_degree)
                    normalized_sim = total_similarities / normalization_factor if normalization_factor > 0 else 0
                    subgraph_similarities[cc_idx, ap_idx, ap_seq_idx] = normalized_sim


        return subgraph_idx, subgraph_similarities



    def compute_head_attention_similarities(self, split):

        """
        Compute or load precomputed head-specific similarities (position, neighborhood, and structure) for attention mechanisms.
        """

        Head_attention_similarities_path = config.PROJECT_ROOT / self.Head_attention_similarities_path

        
        # ==================== POSITION HEAD ==================== #
        if self.hyperparameters['position_head']:

            print (" >>> Preparing position head similarities <<< ")
            
            # Read in precomputed shortest paths between all nodes in the graph
            node_pairwise_shortest_paths_path = config.PROJECT_ROOT / self.Shortest_paths_path
            node_pairwise_shortest_paths = np.load(config.PROJECT_ROOT / self.Shortest_paths_path, allow_pickle=True)

            # Read in precomputed position head similarities if they exist. If they don't, compute them
            train_shortest_path_similarities_path = Head_attention_similarities_path / 'train_shortest_path_similarities.npy'
            val_shortest_path_similarities_path = Head_attention_similarities_path / 'val_shortest_path_similarities.npy'
            test_shortest_path_similarities_path = Head_attention_similarities_path / 'test_shortest_path_similarities.npy'

            if split == 'train_val':
                
                    if train_shortest_path_similarities_path.exists():
                        self.train_shortest_path_similarities = torch.tensor(np.load(train_shortest_path_similarities_path, allow_pickle=True))
                    else:
                        self.train_shortest_path_similarities = self.precompute_shortest_path_similarities(train_shortest_path_similarities_path, 
                                                                                                           node_pairwise_shortest_paths, self.train_cc_ids)
                        
                    print(" >>> Precomputing local position-based head similarities for train data <<< ")
                    self.train_local_position_head_similarities = self.precompute_local_position_head_similarities(self.train_shortest_path_similarities,
                                                                                                           self.train_cc_ids, self.anchors_pos_loc['train'])

                    print(" >>> Precomputing global position-based head similarities for train data <<< ")
                    self.global_position_head_similarities = self.precompute_global_position_head_similarities(self.train_shortest_path_similarities,
                                                                                                           self.all_cc_ids, self.anchors_pos_glob)
                    self.train_global_position_head_similarities = self.global_position_head_similarities


                    if val_shortest_path_similarities_path.exists():
                        self.val_shortest_path_similarities = torch.tensor(np.load(val_shortest_path_similarities_path, allow_pickle=True))
                    else:
                        self.val_shortest_path_similarities = self.precompute_shortest_path_similarities(val_shortest_path_similarities_path, 
                                                                                                           node_pairwise_shortest_paths, self.val_cc_ids)
                        
                    print(" >>> Precomputing local position-based head similarities for val data <<< ")
                    self.val_local_position_head_similarities = self.precompute_local_position_head_similarities(self.val_shortest_path_similarities,
                                                                                                            self.val_cc_ids, self.anchors_pos_loc['val'])

                    print(" >>> Precomputing global position-based head similarities for val data <<< ")
                    self.val_global_position_head_similarities = self.global_position_head_similarities
                    
            
            elif split == 'test':
                if test_shortest_path_similarities_path.exists():
                    self.test_shortest_path_similarities = torch.tensor(np.load(test_shortest_path_similarities_path, allow_pickle=True))
                else:
                    self.test_shortest_path_similarities = self.precompute_shortest_path_similarities(test_shortest_path_similarities_path, 
                                                                                                            node_pairwise_shortest_paths, self.test_cc_ids)

                print(" >>> Precomputing local position-based head similarities for test data <<< ")
                self.test_local_position_head_similarities = self.precompute_local_position_head_similarities(self.test_shortest_path_similarities,
                                                                                                            self.test_cc_ids, self.anchors_pos_loc['test'])

                print(" >>> Precomputing global position-based head similarities for test data <<< ")
                self.test_global_position_head_similarities = self.global_position_head_similarities 
                                                                                                                     
        # If the position head is not used, set all position-related similarities to None
        else:
            self.train_local_position_head_similarities = None     
            self.train_global_position_head_similarities = None
            self.val_local_position_head_similarities = None
            self.val_global_position_head_similarities = None
            self.test_local_position_head_similarities = None
            self.test_global_position_head_similarities = None


        # ==================== NEIGHBORHOOD HEAD ==================== #
        if self.hyperparameters['neighborhood_head']:

            print (" >>> Preparing neighborhood head similarities <<< ")
            
            # Read in precomputed number of Intermediate nodes  between all nodes in the graph
            pairwise_intermediate_nodes_path = config.PROJECT_ROOT / self.Intermediate_nodes_path
            pairwise_intermediate_nodes = np.load(pairwise_intermediate_nodes_path, allow_pickle=True)
            
            # Read in precomputed similarities if they exist. If they don't, compute them
            train_intermediate_node_similarities_path = Head_attention_similarities_path / 'train_intermediate_node_similarities.npy'
            val_intermediate_node_similarities_path = Head_attention_similarities_path / 'val_intermediate_node_similarities.npy'
            test_intermediate_node_similarities_path = Head_attention_similarities_path / 'test_intermediate_node_similarities.npy'
            
            if split == 'train_val':
                
                if train_intermediate_node_similarities_path.exists():
                    self.train_intermediate_node_similarities = torch.tensor(np.load(train_intermediate_node_similarities_path, allow_pickle=True))
                else:
                    self.train_intermediate_node_similarities = self.precompute_intermediate_node_similarities(train_intermediate_node_similarities_path,
                                                                                                               pairwise_intermediate_nodes, self.train_cc_ids)
                    
                print(" >>> Precomputing local neighborhood-based head similarities for train data <<< ")
                self.train_local_neighborhood_head_similarities = self.precompute_neighborhood_head_similarities(self.train_intermediate_node_similarities,
                                                                                                                self.train_cc_ids, self.anchors_neigh_loc['train'])
                
                print(" >>> Precomputing global neighborhood-based head similarities for train data <<< ")
                self.train_global_neighborhood_head_similarities = self.precompute_neighborhood_head_similarities(self.train_intermediate_node_similarities,
                                                                                                                self.train_cc_ids, self.anchors_neigh_glob['train'])
                
                if val_intermediate_node_similarities_path.exists():
                    self.val_intermediate_node_similarities = torch.tensor(np.load(val_intermediate_node_similarities_path, allow_pickle=True))
                else:
                    self.val_intermediate_node_similarities = self.precompute_intermediate_node_similarities(val_intermediate_node_similarities_path,
                                                                                                             pairwise_intermediate_nodes, self.val_cc_ids)
                    
                print(" >>> Precomputing local neighborhood-based head similarities for val data <<< ")
                self.val_local_neighborhood_head_similarities = self.precompute_neighborhood_head_similarities(self.val_intermediate_node_similarities,
                                                                                                            self.val_cc_ids, self.anchors_neigh_loc['val'])

                print(" >>> Precomputing global neighborhood-based head similarities for val data <<< ")
                self.val_global_neighborhood_head_similarities = self.precompute_neighborhood_head_similarities(self.val_intermediate_node_similarities,
                                                                                                            self.val_cc_ids, self.anchors_neigh_glob['val'])

                
            elif split == 'test':
                if test_intermediate_node_similarities_path.exists():
                    self.test_intermediate_node_similarities = torch.tensor(np.load(test_intermediate_node_similarities_path, allow_pickle=True))
                else:
                    self.test_intermediate_node_similarities = self.precompute_intermediate_node_similarities(test_intermediate_node_similarities_path,
                                                                                                            pairwise_intermediate_nodes, self.test_cc_ids)         

                print(" >>> Precomputing local neighborhood-based head similarities for test data <<< ")
                self.test_local_neighborhood_head_similarities = self.precompute_neighborhood_head_similarities(self.test_intermediate_node_similarities,
                                                                                                            self.test_cc_ids, self.anchors_neigh_loc['test'])

                print(" >>> Precomputing global neighborhood-based head similarities for test data <<< ")
                self.test_global_neighborhood_head_similarities = self.precompute_neighborhood_head_similarities(self.test_intermediate_node_similarities,
                                                                                                            self.test_cc_ids, self.anchors_neigh_glob['test'])

        # If the neighborhood head is not used, set all neighborhood-related similarities to None 
        else: 
            self.train_local_neighborhood_head_similarities = None
            self.train_global_neighborhood_head_similarities = None
            self.val_local_neighborhood_head_similarities = None
            self.val_global_neighborhood_head_similarities = None
            self.test_local_neighborhood_head_similarities = None
            self.test_global_neighborhood_head_similarities = None

            
        # ==================== STRUCTURE HEAD ==================== #
        if self.hyperparameters['structure_head']:

            print (" >>> Preparing structure head similarities <<< ")
            
            # Load adjacency matrix and degree dictionary
            adj_matrix_path = config.PROJECT_ROOT / self.ADJ_matrix_path
            node_degrees_dic_path = config.PROJECT_ROOT / self.Node_degrees_dic_path
            adj_matrix = np.load(adj_matrix_path, allow_pickle=True)
            with open(str(node_degrees_dic_path), 'r') as degrees_file:
                node_degrees_dic = json.load(degrees_file)
            node_degrees_dic = {int(key): value for key, value in node_degrees_dic.items()}
            
            # Paths to local and global structure head similarities for train, val, and test datasets
            all_local_structure_head_similarities_path = Head_attention_similarities_path / ('all_local_structure_head_similarities_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')
            all_global_structure_head_similarities_path = Head_attention_similarities_path / ('all_global_structure_head_similarities_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')
            
            train_local_structure_head_similarities_path = Head_attention_similarities_path / ('train_local_structure_head_similarities_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')
            train_global_structure_head_similarities_path = Head_attention_similarities_path / ('train_global_structure_head_similarities_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')

            val_local_structure_head_similarities_path = Head_attention_similarities_path / ('val_local_structure_head_similarities_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')
            val_global_structure_head_similarities_path = Head_attention_similarities_path / ('val_global_structure_head_similarities_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')

            test_local_structure_head_similarities_path = Head_attention_similarities_path / ('test_local_structure_head_similarities_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')
            test_global_structure_head_similarities_path = Head_attention_similarities_path /('test_global_structure_head_similarities_' + str(self.hyperparameters['n_hawkes_process_sequences']) + '_' + str(self.hyperparameters['hawkes_process_sequence_len']) + '_' + str(self.hyperparameters['max_sample_anchor_patches_structure']) + '.npy')

            # Read in precomputed structure head similarities if they exist. If they don't, compute them
            if all_local_structure_head_similarities_path.exists() and all_global_structure_head_similarities_path.exists():
                self.all_local_structure_head_similarities = torch.tensor(np.load(all_local_structure_head_similarities_path, allow_pickle=True))
                self.all_global_structure_head_similarities = torch.tensor(np.load(all_global_structure_head_similarities_path, allow_pickle=True))

                if split == "train_val":
                    if train_local_structure_head_similarities_path.exists() and train_global_structure_head_similarities_path.exists():
                        print(" >>> Loading local and global structure head similarities for train data <<< ")
                        self.train_local_structure_head_similarities = torch.tensor(np.load(train_local_structure_head_similarities_path, allow_pickle=True))
                        self.train_global_structure_head_similarities = torch.tensor(np.load(train_global_structure_head_similarities_path, allow_pickle=True))

                    if val_local_structure_head_similarities_path.exists() and val_global_structure_head_similarities_path.exists():
                        print(" >>> Loading local and global structure head similarities for val data <<< ")
                        self.val_local_structure_head_similarities = torch.tensor(np.load(val_local_structure_head_similarities_path, allow_pickle=True))
                        self.val_global_structure_head_similarities = torch.tensor(np.load(val_global_structure_head_similarities_path, allow_pickle=True))


                elif split == "test":
                    if test_local_structure_head_similarities_path.exists() and test_global_structure_head_similarities_path.exists():
                        print(" >>> Loading local and global structure head similarities for test data <<< ")
                        self.test_local_structure_head_similarities = torch.tensor(np.load(test_local_structure_head_similarities_path, allow_pickle=True))
                        self.test_global_structure_head_similarities = torch.tensor(np.load(test_global_structure_head_similarities_path, allow_pickle=True))
                    else:
                        print(" >>> Assigning precomputed structure head similarities for test data <<< ")
                        self.test_local_structure_head_similarities = self.all_local_structure_head_similarities
                        np.save(test_local_structure_head_similarities_path, self.all_local_structure_head_similarities)
                        self.test_global_structure_head_similarities = self.all_global_structure_head_similarities
                        np.save(test_global_structure_head_similarities_path, self.all_global_structure_head_similarities)
                        print(" >>> local and global structure head similarities for test data saved <<< ")
                        
            else:
                print(" >>> Precomputing local structure head similarities <<< ")
                self.all_local_structure_head_similarities = self.precompute_structure_head_similarities(all_local_structure_head_similarities_path,
                                                            adj_matrix, node_degrees_dic, self.loc_structure_anchor_hawkes_process, self.all_cc_ids)
                print(" >>> Precomputing global structure head similarities <<< ")
                self.all_global_structure_head_similarities = self.precompute_structure_head_similarities(all_global_structure_head_similarities_path,
                                                            adj_matrix, node_degrees_dic, self.glob_structure_anchor_hawkes_process, self.all_cc_ids)
                
                if split == "train_val":
                    self.train_local_structure_head_similarities = self.all_local_structure_head_similarities
                    np.save(train_local_structure_head_similarities_path, self.all_local_structure_head_similarities)
                    self.train_global_structure_head_similarities = self.all_global_structure_head_similarities
                    np.save(train_global_structure_head_similarities_path, self.all_global_structure_head_similarities)

                    self.val_local_structure_head_similarities = self.all_local_structure_head_similarities
                    np.save(val_local_structure_head_similarities_path, self.all_local_structure_head_similarities)
                    self.val_global_structure_head_similarities = self.all_global_structure_head_similarities
                    np.save(val_global_structure_head_similarities_path, self.all_global_structure_head_similarities)
                    print(" >>> local and global structure head similarities saved <<< ")

        # If the structure head is not used, set all structure-related similarities to None 
        else:
            self.train_local_structure_head_similarities = None
            self.train_global_structure_head_similarities = None
            self.val_local_structure_head_similarities = None
            self.val_global_structure_head_similarities = None
            self.test_local_structure_head_similarities = None
            self.test_global_structure_head_similarities = None
        
        

# =============================
# Prepare data
# =============================

    def prepare_test_data(self):

        """
        Prepare the dataset required for the test phase. This includes:
            - Initializing connected components for test subgraphs.
            - Getting global neighborhood sets.
            - Computing anchor patch sampling similarity matrices.
            - Initializing position and neighborhood anchor patches.
            - Computing attention-based similarity matrices.
            
        Note:
            Global position and structure anchor patches are shared across splits,
            so they are not re-initialized here.
        """
        
        print(" >>> Preparing test dataset <<< ")
        self.test_cc_ids = self.initialize_cc_ids(self.test_subgraphs)

        print(" >>> Getting global sets for test dataset <<< ")
        self.get_global_sets(split='test')
        print(" >>> Finished getting global sets for test dataset <<< ")

        print(" >>> Computing anchor patch sampling similarities for test dataset <<< ")
        self.compute_ap_sampling_similarities(split='test')
        print(" >>> Finished Computing anchor patch sampling similarities for test dataset <<< ")


        if self.hyperparameters['neighborhood_head']:
            print(" >>> Initializing local and global neighborhood anchor patches for test dataset <<< ") 
            self.anchors_neigh_loc, self.anchors_neigh_glob = init_anchors_neighborhood('test', self.hyperparameters, self.device, None, None, self.test_cc_ids,
                                                         None, None, self.test_neigh_glob_set, None, None, self.test_ccs_neigh_pos_similarities)
            print(" >>> Finished initializing local and global neighborhood anchor patches for test dataset <<< ")
        else:
            self.anchors_neigh_loc, self.anchors_neigh_glob = None, None


        if self.hyperparameters['position_head']:
            print(" >>> Initializing local position anchor patches for test dataset <<< ")
            self.anchors_pos_loc = init_anchors_pos_loc('test', self.hyperparameters, None, None, self.test_ccs_neigh_pos_similarities,
                                                        self.device, None, None, self.test_cc_ids, None, None, self.test_subgraphs)
            print(" >>> Finished initializing local position anchor patches for test dataset <<< ")
        else:
            self.anchors_pos_loc = None


        print(" >>> Computing head attention similarities for test dataset <<< ")
        self.compute_head_attention_similarities(split='test')
        print(" >>> Finished computing head attention similarities for test dataset <<< ")

        print(" >>>  Preparation for test data completed <<< ")

        
        
    def prepare_data(self):

        """
        Prepare the data required for the training and validation phases. This includes:
            - Initializing connected components for train, val, and all subgraphs.
            - Getting global neighborhood sets.
            - Computing anchor patch sampling similarity matrices.
            - Initializing position, neighborhood, and structure anchor patches.
            - Computing attention-based similarity matrices for all heads.
        """
        
        print(" >>> Preparing train and val datasets <<< ", flush=True)
        # Initialize connected component matrices for train, validation, and combined sets (n_subgraphs, max_n_cc, max_len_cc)
        self.train_cc_ids = self.initialize_cc_ids(self.train_subgraphs)
        self.val_cc_ids = self.initialize_cc_ids(self.val_subgraphs)
        self.all_cc_ids = self.initialize_cc_ids(self.all_subgraphs)

        # Get global neighborhood sets for train and val CCs
        print(" >>> Getting global sets for train and val datasets <<< ", flush=True)
        self.get_global_sets(split='train_val')
        print(" >>> Finished getting global sets for train and val datasets <<< ", flush=True)

        # Compute anchor patch sampling similarities
        print(" >>> Computing anchor patch sampling similarities for train and val datasets <<< ", flush=True)
        self.compute_ap_sampling_similarities(split='train_val')
        print(" >>> Finished computing anchor patch sampling similarities for train and val datasets <<< ", flush=True)

    
        # Initialize neighborhood anchor patches
        if self.hyperparameters['neighborhood_head']:
            print(" >>> Initializing local and global neighborhood anchor patches for train and val datasets <<< ", flush=True)
            self.anchors_neigh_loc, self.anchors_neigh_glob = init_anchors_neighborhood('train_val', self.hyperparameters, self.device, self.train_cc_ids,
                                                      self.val_cc_ids, None, self.train_neigh_glob_set, self.val_neigh_glob_set, None,
                                                      self.train_ccs_neigh_pos_similarities, self.val_ccs_neigh_pos_similarities, None) 
            print(" >>> Finished initializing local and global neighborhood anchor patches for train and val datasets  <<< ", flush=True)
            
        else:
            self.anchors_neigh_loc, self.anchors_neigh_glob = None, None

        # Initialize position anchor patches
        if self.hyperparameters['position_head']:
            print(" >>> Initializing local and global position anchor patches for train and val datasets <<< ", flush=True)
            self.anchors_pos_loc = init_anchors_pos_loc('train_val', self.hyperparameters, self.train_ccs_neigh_pos_similarities, self.val_ccs_neigh_pos_similarities,
                                                        None, self.device, self.train_cc_ids, self.val_cc_ids, None, self.train_subgraphs, self.val_subgraphs, None)
        
            self.anchors_pos_glob = init_anchors_pos_glob(self.hyperparameters, self.all_subgraphs_dis_similarities, self.device)
            print(" >>> Finished initializing local and global position anchor patches for train and val datasets <<< ", flush=True)
       
        else:
            self.anchors_pos_loc, self.anchors_pos_glob = None, None

        
        # Initialize structure anchor patches
        if self.hyperparameters['structure_head']:
            print(" >>> Initializing local and global structure anchor patches <<< ", flush=True)
            self.anchors_structure = init_anchors_structure(self.hyperparameters, self.structure_anchors, self.loc_structure_anchor_hawkes_process,
                                                        self.glob_structure_anchor_hawkes_process) 
            print(" >>> Finished initializing local and global structure anchor patches <<< ", flush=True)
            
        else:
            self.anchors_structure = None


        # Compute attention-based similarities
        print(" >>> Computing head attention similarities for train and val datasets <<< ", flush=True)
        self.compute_head_attention_similarities(split='train_val')
        print(" >>> Finished computing head attention similarities for train and val datasets <<< ", flush=True)
        
        print(" >>> Preparation for train and val datasets completed <<< ", flush=True)


        
# =============================
# Load and collate Batch Data
# =============================

    def _pad_collate(self, batch):

        """
        Collate function for batching subgraph-level data used in DDI prediction.

        This function:
            - Batch subgraph indices, node IDs, and connected component (CC) IDs.
            - Gather similarity scores from all attention heads (positional, neighborhood, structural) for each subgraph in the batch.
            - Trim global padding from CC tensors and apply padding to node ID sequences based on batch length.
            - Stack similarity tensors per attention head and per layer across the batch.
        """

        # Containers for subgraph-specific data across the batch
        batch_subgraphs_data = {}
        batch_subgraph_idx = []
        batch_subgraph_node_ids = []
        batch_cc_ids = []

        # Per-head attention similarity containers (local & global)
        batch_loc_pos_head_sim = []
        batch_glob_pos_head_sim = []
        batch_loc_neigh_head_sim = []
        batch_glob_neigh_head_sim = []
        batch_loc_struct_head_sim = []
        batch_glob_struct_head_sim = []

        # Unpack each item in the batch
        DDI_edge, DDI_edge_label, \
        subgraph1_idx, subgraph1_node_ids, subgraph1_cc_ids, \
        subgraph1_loc_pos_head_sim, subgraph1_glob_pos_head_sim, subgraph1_loc_neigh_head_sim, \
        subgraph1_glob_neigh_head_sim, subgraph1_loc_struct_head_sim, subgraph1_glob_struct_head_sim, \
        subgraph2_idx, subgraph2_node_ids, subgraph2_cc_ids, \
        subgraph2_loc_pos_head_sim, subgraph2_glob_pos_head_sim, subgraph2_loc_neigh_head_sim, \
        subgraph2_glob_neigh_head_sim, subgraph2_loc_struct_head_sim, subgraph2_glob_struct_head_sim = zip(*batch)

        # Collect unique subgraph data from each DDI edge
        for e, edge in enumerate(DDI_edge):
            sub1_id = edge[0].item()
            sub2_id = edge[1].item()
            
            if sub1_id not in batch_subgraphs_data:
                    batch_subgraphs_data[sub1_id] = {
                    'subgraph_idx': subgraph1_idx[e],
                    'subgraph_node_ids': subgraph1_node_ids[e],
                    'cc_ids': subgraph1_cc_ids[e],
                    'loc_pos_head_sim': subgraph1_loc_pos_head_sim[e] if subgraph1_loc_pos_head_sim[e] is not None else None,
                    'glob_pos_head_sim': subgraph1_glob_pos_head_sim[e] if subgraph1_glob_pos_head_sim[e] is not None else None,
                    'loc_neigh_head_sim': subgraph1_loc_neigh_head_sim[e] if subgraph1_loc_neigh_head_sim[e] is not None else None,
                    'glob_neigh_head_sim': subgraph1_glob_neigh_head_sim[e] if subgraph1_glob_neigh_head_sim[e] is not None else None,
                    'loc_struct_head_sim': subgraph1_loc_struct_head_sim[e] if subgraph1_loc_struct_head_sim[e] is not None else None,
                    'glob_struct_head_sim': subgraph1_glob_struct_head_sim[e] if subgraph1_glob_struct_head_sim[e] is not None else None,
                    }

            if sub2_id not in batch_subgraphs_data:
                    batch_subgraphs_data[sub2_id] = {
                    'subgraph_idx': subgraph2_idx[e],
                    'subgraph_node_ids': subgraph2_node_ids[e],
                    'cc_ids': subgraph2_cc_ids[e],
                    'loc_pos_head_sim': subgraph2_loc_pos_head_sim[e] if subgraph2_loc_pos_head_sim[e] is not None else None,
                    'glob_pos_head_sim': subgraph2_glob_pos_head_sim[e] if subgraph2_glob_pos_head_sim[e] is not None else None,
                    'loc_neigh_head_sim': subgraph2_loc_neigh_head_sim[e] if subgraph2_loc_neigh_head_sim[e] is not None else None,
                    'glob_neigh_head_sim': subgraph2_glob_neigh_head_sim[e] if subgraph2_glob_neigh_head_sim[e] is not None else None,
                    'loc_struct_head_sim': subgraph2_loc_struct_head_sim[e] if subgraph2_loc_struct_head_sim[e] is not None else None,
                    'glob_struct_head_sim': subgraph2_glob_struct_head_sim[e] if subgraph2_glob_struct_head_sim[e] is not None else None
                    }

        # Reorganize subgraph data into batched lists
        for key, sub_data in sorted(batch_subgraphs_data.items()):
            batch_subgraph_idx.append(sub_data['subgraph_idx'])
            batch_subgraph_node_ids.append(sub_data['subgraph_node_ids'])
            batch_cc_ids.append(sub_data['cc_ids'])
            batch_loc_pos_head_sim.append(sub_data['loc_pos_head_sim'])
            batch_glob_pos_head_sim.append(sub_data['glob_pos_head_sim'])
            batch_loc_neigh_head_sim.append(sub_data['loc_neigh_head_sim'])
            batch_glob_neigh_head_sim.append(sub_data['glob_neigh_head_sim'])
            batch_loc_struct_head_sim.append(sub_data['loc_struct_head_sim'])
            batch_glob_struct_head_sim.append(sub_data['glob_struct_head_sim'])
       
        # Stack edges and labels
        batch_DDI_edge = torch.stack(DDI_edge)
        batch_DDI_edge_label = torch.stack(DDI_edge_label)
        
        # Stack subgraph indices
        batch_subgraph_idx = torch.stack(batch_subgraph_idx)
        
        # Pad node ID sequences to the longest one in the batch
        batch_subgraph_node_ids = pad_sequence(batch_subgraph_node_ids, batch_first=True, padding_value=config.PAD_VALUE)

        # Trim unnecessary padding from CC IDs (originally padded to max CC length across the full dataset)
        batch_cc_ids = torch.stack(batch_cc_ids)
        batch_sz, max_n_cc, a = batch_cc_ids.shape
        batch_cc_ids_reshaped = batch_cc_ids.view(batch_sz*max_n_cc, -1)
        ind = (torch.sum(torch.abs(batch_cc_ids_reshaped), dim=0) != 0)
        batch_cc_ids = batch_cc_ids_reshaped[:,ind].view(batch_sz, max_n_cc, -1)
        

        def stack_batch_sim_tensors(batch_sim_data):
            """
            Stack per-layer similarity tensors for each attention head across all subgraphs in the batch.
            """
            stacked_tensors = {}
            for key in batch_sim_data[0].keys():
                stacked_tensors[key] = torch.stack([item[key] for item in batch_sim_data], dim=0)
                
            return stacked_tensors
        

        # Stack each attention head similarity tensors across all subgraphs in the batch, if the head is available.
        batch_loc_pos_head_sim = stack_batch_sim_tensors(batch_loc_pos_head_sim) if None not in batch_loc_pos_head_sim else None
        batch_glob_pos_head_sim = stack_batch_sim_tensors(batch_glob_pos_head_sim) if None not in batch_glob_pos_head_sim else None
        batch_loc_neigh_head_sim = stack_batch_sim_tensors(batch_loc_neigh_head_sim) if None not in batch_loc_neigh_head_sim else None
        batch_glob_neigh_head_sim = stack_batch_sim_tensors(batch_glob_neigh_head_sim) if None not in batch_glob_neigh_head_sim else None
        batch_loc_struct_head_sim = torch.stack(batch_loc_struct_head_sim) if None not in batch_loc_struct_head_sim else None
        batch_glob_struct_head_sim = torch.stack(batch_glob_struct_head_sim) if None not in batch_glob_struct_head_sim else None
       
        return {
            'DDI_edges': batch_DDI_edge,
            'DDI_edge_labels': batch_DDI_edge_label,
            'subgraph_idx': batch_subgraph_idx,
            'subgraph_ids': batch_subgraph_node_ids,
            'cc_ids': batch_cc_ids,
            'loc_pos_head_sim': batch_loc_pos_head_sim,
            'glob_pos_head_sim': batch_glob_pos_head_sim,
            'loc_neigh_head_sim':batch_loc_neigh_head_sim,
            'glob_neigh_head_sim': batch_glob_neigh_head_sim,
            'loc_struct_head_sim': batch_loc_struct_head_sim,
            'glob_struct_head_sim': batch_glob_struct_head_sim
            }


            
    def train_dataloader(self):
        
        """
        Return a DataLoader for the train dataset.
        """
 
        print(" >>> Initializing train Dataloader <<< ")
        dataset = DDISubgraphDataset(self.train_DDI_edges, self.train_DDI_edges_label, self.train_subgraphs_dict,
                                  self.train_subgraphs_indices, self.train_cc_ids, self.train_local_position_head_similarities,
                                  self.train_global_position_head_similarities, self.train_local_neighborhood_head_similarities,
                                  self.train_global_neighborhood_head_similarities, self.train_local_structure_head_similarities,
                                  self.train_global_structure_head_similarities) 
        loader = DataLoader(dataset, batch_size = self.hyperparameters['batch_size'] , shuffle=True, collate_fn=self._pad_collate)
  
        return loader



    def val_dataloader(self):
        
        """
        Return a DataLoader for the validation dataset.
        """
        
        print(" >>> Initializing validation Dataloader <<< ")
        dataset = DDISubgraphDataset(self.val_DDI_edges, self.val_DDI_edges_label, self.val_subgraphs_dict,
                                  self.val_subgraphs_indices, self.val_cc_ids, self.val_local_position_head_similarities,
                                  self.val_global_position_head_similarities, self.val_local_neighborhood_head_similarities,
                                  self.val_global_neighborhood_head_similarities, self.val_local_structure_head_similarities,
                                  self.val_global_structure_head_similarities)
  
        loader = DataLoader(dataset, batch_size = self.hyperparameters['batch_size'], shuffle=False, collate_fn=self._pad_collate , drop_last=False)
        
        return loader


    
    def test_dataloader(self):
        
        """
        Return a DataLoader for the test dataset.
        """
        
        self.prepare_test_data()
        print(" >>> Initializing test Dataloader <<< ")
        dataset = DDISubgraphDataset(self.test_DDI_edges, self.test_DDI_edges_label, self.test_subgraphs_dict,
                                  self.test_subgraphs_indices, self.test_cc_ids, self.test_local_position_head_similarities,
                                  self.test_global_position_head_similarities, self.test_local_neighborhood_head_similarities,
                                  self.test_global_neighborhood_head_similarities, self.test_local_structure_head_similarities,
                                  self.test_global_structure_head_similarities)

        loader = DataLoader(dataset, batch_size = self.hyperparameters['batch_size'], shuffle=False, collate_fn=self._pad_collate)
        
        return loader


    
# =============================
# Optimize model and log gradients
# =============================

    def on_after_backward(self):

        """
        Log the gradient norm of each trainable parameter after the backward pass.
        """
        
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.log(f"{name}_grad_norm", param.grad.norm(), prog_bar=False)


                
    def optimizer_step(self, *args, **kwargs):
        
        """
        Log parameter norms before and after the optimizer step.
        """
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.log(f"{name}_param_norm_before_update", param.data.norm())
                
        super().optimizer_step(*args, **kwargs)
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.log(f"{name}_param_norm_after_update", param.data.norm())



    def configure_optimizers(self):
        
        """
        Set up the Adam optimizer with the specified learning rate.
        """
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        return optimizer

    


