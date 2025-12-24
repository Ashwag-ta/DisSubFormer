# PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# Ours
import dissubformer_input_config as config



class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, conv_type, dropout, edge_attr_dim=None):
        
        """
        Initialize the GNN model.
        
        Args:
            - in_dim (int): Input embeddings dimension per node.
            - hidden_dim (int): Hidden layer size.
            - out_dim (int): Output embeddings dimension per node.
            - conv_type (str): Type of GNN layer (GCN or GAT).
            - dropout (float): Dropout rate.
            - edge_attr_dim (int, optional): Dimension of edge attributes (used for GAT with edge weights).
        """
        
        super(GNNModel, self).__init__()
        self.conv_type = conv_type
        self.edge_attr_dim = edge_attr_dim
        self.dropout = dropout

        # GCN layers for PPI graph
        if conv_type == "GCN":
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)

        # GAT layers with edge attributes for GO_FS graph
        if conv_type == "GAT":
            self.conv1 = GATConv(in_dim, hidden_dim, heads=config.POSSIBLE_GAT_HEADS_1[0], concat=True, dropout=self.dropout, edge_dim=self.edge_attr_dim)
            self.conv2 = GATConv(hidden_dim * config.POSSIBLE_GAT_HEADS_1[0], out_dim, heads=config.POSSIBLE_GAT_HEADS_2[0], concat=False, dropout=self.dropout, edge_dim=self.edge_attr_dim)


            
    def forward(self, x, edge_index, edge_attr=None):
        
        if self.conv_type == "GAT":
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = self.conv2(x, edge_index)
            
        return x
