# PyTorch 
import torch
import torch.nn as nn
import torch.nn.functional as F



class MHADisSubFormer(nn.Module):

    """
    Multi-Head Attention module for DisSubFormer.

    This module supports three types of single-head attention mechanisms:
    - Neighborhood-based attention
    - Position-based attention
    - Structure-based attention

    Each mechanism computes attention between connected components and
    anchor patches , integrating corresponding relational similarity terms.
    """
    
    def __init__(self, hyperparameters):
        
        super(MHADisSubFormer, self).__init__()
        self.hyperparameters = hyperparameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        
    def neighborhood_based_attention(self, head_Q, head_K, head_V, NH_sims, cc_ids, cc_embeds, cc_embed_mask, 
                anchor_patches, anchor_embeds, anchor_mask):
        
        """
        Neighborhood-based attention between connected components and neighborhood anchor patches, integrating the neighborhood similarity relational term (NH_sims).

        Returns:
            - output (Tensor): Updated embeddings for each connected component.
        """
        
        Q = head_Q(cc_embeds).unsqueeze(2)  # Query: linear projection of connected components (batch_size, max_n_cc, 1, embed_dim)
        K = head_K(anchor_embeds)  # Key: linear projection of anchor patches (batch_size, max_n_cc, n_anchors, embed_dim)
        V = head_V(anchor_embeds)  # Value: linear projection of anchor patches (batch_size, max_n_cc, n_anchors, embed_dim)

        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (K.size(-1) ** 0.5)

        # Incorporate neighborhood similarity term into the attention scores.
        adj_attn = attn_scores + NH_sims.unsqueeze(2)

        # Normalize attention 
        attn_probs = torch.where((adj_attn.sum(dim=-1, keepdim=True) != 0), F.softmax(adj_attn, dim=-1), adj_attn)
        
        # Weighted sum of values to get updated CC embeddings
        output = torch.matmul(attn_probs, V)
        
        return output


   
    def position_based_attention(self, head_Q, head_K, head_V, PH_sims, cc_ids, cc_embeds, cc_embed_mask, 
                anchor_patches, anchor_embeds, anchor_mask):

        """
        Position-based attention between connected components and positional anchor patches, integrating the positional similarity relational term (PH_sims).

        Returns:
            - output (Tensor): Updated embeddings for each connected component.
        """
        
        Q = head_Q(cc_embeds).unsqueeze(2)   
        K = head_K(anchor_embeds)  
        V = head_V(anchor_embeds)  
   
        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (K.size(-1) ** 0.5)

        # Incorporate positional similarity term into the attention scores
        adj_attn = attn_scores + PH_sims.unsqueeze(2)
        
        # Normalize attention 
        attn_probs = torch.where((adj_attn.sum(dim=-1, keepdim=True) != 0), F.softmax(adj_attn, dim=-1), adj_attn)
        
        # Weighted sum of values to get updated CC embeddings
        output = torch.matmul(attn_probs, V)
        
        return output


    
    def structure_based_attention(self, head_Q, head_K, head_V, SH_sims, struct_anchors_sim_idx, cc_ids, cc_embeds, cc_embed_mask, 
                              anchor_patches, anchor_embeds, anchor_mask):
        
        """
        Structure-based attention between connected components and structural anchor patches, integrating the structural similarity relational term (SH_sims).

        Returns:
            - output (Tensor): Updated embeddings for each connected component.
        """
        
        Q = head_Q(cc_embeds).unsqueeze(2)  
        K = head_K(anchor_embeds)          
        V = head_V(anchor_embeds)       

        # Select relevant structural similarities for the sampled anchor patches
        current_sims = SH_sims[:, :, struct_anchors_sim_idx].unsqueeze(2)  # Shape: (batch_size, max_n_cc, 1, len(struct_anchors_sim_idx))
        
        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (K.size(-1) ** 0.5)  

         # Incorporate structural similarity term into the attention scores
        adj_attn = attn_scores + current_sims 

        # Normalize attention 
        attn_probs = torch.where((adj_attn.sum(dim=-1, keepdim=True) != 0), F.softmax(adj_attn, dim=-1), adj_attn)  # Shape: (batch_size, max_n_cc, 1, n_anchors)

        # Weighted sum of values to get updated CC embeddings
        output = torch.matmul(attn_probs, V) 
        
        return output



    def forward(self, head_Q, head_K, head_V, sims, struct_anchors_sim_idx, cc_ids, cc_embeds, cc_embed_mask, 
                anchor_patches, anchor_embeds, anchor_mask, head_type):
        """
        Forward pass for a single attention head between connected components and anchor patches.

        Returns:
            cc_embeddings_out (Tensor): The attention-weighted embeddings for connected components.
        """
        
        if head_type == 'position':
                cc_embeddings_out = self.position_based_attention(head_Q, head_K, head_V, sims, cc_ids, cc_embeds, cc_embed_mask, 
                anchor_patches, anchor_embeds, anchor_mask)

        if head_type == 'neighborhood':
                cc_embeddings_out = self.neighborhood_based_attention(head_Q, head_K, head_V, sims, cc_ids, cc_embeds, cc_embed_mask, 
                anchor_patches, anchor_embeds, anchor_mask)
           
        if head_type == 'structure':
                cc_embeddings_out = self.structure_based_attention(head_Q, head_K, head_V, sims, struct_anchors_sim_idx, cc_ids, cc_embeds, cc_embed_mask, 
                anchor_patches, anchor_embeds, anchor_mask)

        return cc_embeddings_out
