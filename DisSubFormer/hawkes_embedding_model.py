#General Imports
import sys
import random

# PyTorch
import torch
import torch.nn as nn

# Sci-kit Learn 
from sklearn.metrics import accuracy_score

# Ours
sys.path.insert(0, '..') 
import main_config as config



class HawkesSeqEmbedding:
    
    """
    Train node embeddings using the Hawkes process
    
    """
    
    def __init__(self, PPI_graph, node_embeddings, loc_anchors_structure, glob_anchors_structure, hyperparameters, hawkes_embeddings_path):

        """
        Initialize the HawkesSeqEmbedding trainer.

        Args:
            PPI_graph (Graph object): A NetworkX graph.
            node_embeddings (Tensor): Tensor containing embeddings for each node.
            loc_anchors_structure (Tensor): Local anchor patch sequences.
            glob_anchors_structure (Tensor): Global anchor patch sequences.
            hyperparameters (dict): Dictionary containing training hyperparameters.
            hawkes_embeddings_path (str): Path to save learned embeddings.
            
        """
        
        self.PPI_graph = PPI_graph
        self.loc_anchors_structure = loc_anchors_structure
        self.glob_anchors_structure = glob_anchors_structure
        self.hyperparameters = hyperparameters
        self.node_embeddings = nn.Embedding.from_pretrained(node_embeddings.weight.clone(), freeze=False)        
        self.hawkes_embeddings_path = hawkes_embeddings_path
        self.emb_mask = (node_embeddings.weight.sum(dim=1) == config.PAD_VALUE)
        
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.node_embeddings.parameters(), lr=self.hyperparameters['hawkes_learning_rate'])


   
    def sample_negative_pool(self, all_seq_nodes):
        
        """
        Sample a global pool of negative nodes based on degree-weighted sampling.

        Args:
            - all_seq_nodes (list): All nodes appearing in structure anchor patch sequences.

        Returns:
            - global_negative_pool (list): Sampled negative node IDs.
        """
        
        negative_candidates = list(set(self.PPI_graph.nodes()) - set(all_seq_nodes))
        candidate_degrees = torch.tensor([self.PPI_graph.degree[node] for node in negative_candidates], dtype=torch.float32)
        probabilities = (candidate_degrees ** (3 / 4)).numpy()
        probabilities /= probabilities.sum()
       
        pool_size = min(self.hyperparameters['hawkes_global_negative_pool_size'], len(negative_candidates))
        top_indices = torch.topk(candidate_degrees, k=pool_size).indices
        global_negative_pool = [negative_candidates[idx] for idx in torch.topk(candidate_degrees, k=pool_size).indices.tolist()]

        return global_negative_pool  


    
    def prepare_all_sequences(self, loc_anchors_structure, glob_anchors_structure):
        
        """
        Combine local and global anchor patch sequences into a unified list.

        Returns:
            - all_sequences (list): A list of all unique sequences (local and global).
        """
        
        all_sequences = []
        anchors_structure = torch.cat([loc_anchors_structure, glob_anchors_structure], dim=1)
        reshaped_sequences = anchors_structure.view(-1, anchors_structure.size(-1))  
        valid_sequences = reshaped_sequences[reshaped_sequences.sum(dim=1) != config.PAD_VALUE]
        all_sequences.extend([[value.item() for value in seq if value != config.PAD_VALUE] for seq in valid_sequences])
        all_sequences = sorted(all_sequences, key=lambda x: x[0] if len(x) > 0 else 0)
        
        return all_sequences


    
    def compute_hawkes_intensities(self, sequence=None, negative_sample=None, is_positive=True):

        """
        Compute the Hawkes intensity values for a given sequence of nodes.

        Args:
            - sequence (Tensor): Input sequence for positive samples.
            - negative_sample (tuple): Tuple of (pos_node, neg_node) for negative samples. 
            - is_positive (bool): Flag to distinguish between positive and negative sample processing.

        Returns:
            - intensities (list): Computed intensities.
            - labels (list): Corresponding binary labels (1 for positive, 0 for negative).
        """
        
        intensities = [] 
        labels = [] 

        if is_positive:
            for t in range(1, len(sequence)):  
                curr_node = sequence[t]
                prev_node = sequence[t - 1]  

                e_curr_node = self.node_embeddings.weight[curr_node].unsqueeze(0)  
                e_prev_node = self.node_embeddings.weight[prev_node].unsqueeze(0) 

                # Base rate based on pairwise Euclidean distance
                base_rate = -torch.cdist(e_curr_node, e_prev_node, p=2).squeeze()
               
                influence_sum = 0.0
                historical_nodes = sequence[:t - 1] if t > 1 else torch.tensor([], dtype=torch.long)

                if len(historical_nodes) > 0:
                    historical_embeddings = self.node_embeddings.weight[historical_nodes]
                    dis_for_each_h_node = torch.exp(-torch.cdist(historical_embeddings, e_curr_node, p=2).squeeze())
                    dis_sum_across_all_h_nodes = torch.sum(dis_for_each_h_node)
                    weights = dis_for_each_h_node / dis_sum_across_all_h_nodes
                    
                    for idx_h, h_embedding in enumerate(historical_embeddings):
                        reshaped_weight = weights.item() if weights.dim() == 0 else weights[idx_h] 
                        alpha_h_nei = -torch.cdist(h_embedding.unsqueeze(0), e_curr_node, p=2).squeeze() * reshaped_weight
                        time_diff = t - (idx_h)  # Time difference is based on position of historical nodes
                        decay = torch.exp(-self.hyperparameters['hawkes_decay_rate'] * torch.tensor(time_diff, dtype=torch.float32))
                        influence_sum += alpha_h_nei * decay

                positive_intensity = torch.exp(base_rate + influence_sum)
                intensities.append(positive_intensity)
                labels.append(1)
                
        else:
            pos_node, neg_node = negative_sample  
            e_pos_node = self.node_embeddings.weight[pos_node].unsqueeze(0)
            e_neg_node = self.node_embeddings.weight[neg_node].unsqueeze(0)
            neg_intensity = torch.exp(-torch.cdist(e_pos_node, e_neg_node, p=2).squeeze())
            intensities.append(neg_intensity)
            labels.append(0)  
       
        return intensities, labels


    
    def compute_loss(self, positive_intensities, negative_intensities):
        
        """
        Compute the total loss over the epoch.
        
        Args:
            - positive_intensities (Tensor): Positive sample intensities.
            - negative_intensities (Tensor): Negative sample intensities.

        Returns:
            - total_loss (Tensor): Total loss value.
        """

        pos_loss = torch.log(torch.sigmoid(positive_intensities)).sum()
        neg_loss = (-torch.log(torch.sigmoid(negative_intensities)) / len(negative_intensities)).sum()
        total_loss = -(pos_loss + neg_loss)
        
        return total_loss



    def train_hawkes_node_embeddings(self):
        
        """
        Train the node embeddings using the sampled anchor patches and save the best embeddings.
        """
        
        self.optimizer.zero_grad()  
        best_accuracy = 0.0  
        best_embeddings = None  
        sampled_negative_pairs = []
        
        all_sequences = self.prepare_all_sequences (self.loc_anchors_structure, self.glob_anchors_structure) 
        all_seq_nodes = list({node for sequence in all_sequences for node in sequence})
        global_negative_pool = self.sample_negative_pool(all_seq_nodes)
        num_negative_samples = sum(len(sequence) - 1 for sequence in all_sequences) // 2  
    
        for _ in range(num_negative_samples):
            pos_node = random.choice(all_seq_nodes)  
            neg_node = random.choice(global_negative_pool)
            sampled_negative_pairs.append((pos_node, neg_node))

        for epoch in range(self.hyperparameters['hawkes_epochs']):  
            self.optimizer.zero_grad()
            all_positive_intensities = []
            all_negative_intensities = []
            all_positive_labels = []
            all_negative_labels = []
            
            for sequence in all_sequences:
                sequence = torch.LongTensor(sequence)
                pos_seq_intensities, pos_seq_labels = self.compute_hawkes_intensities(sequence , negative_sample=None, is_positive=True)
                all_positive_intensities.extend(pos_seq_intensities)
                all_positive_labels.extend(pos_seq_labels)
                
            for pos_node, neg_node in sampled_negative_pairs:
                neg_sample_intensity, neg_sample_label = self.compute_hawkes_intensities( sequence=None, negative_sample=(pos_node, neg_node), is_positive=False)                              
                all_negative_intensities.extend(neg_sample_intensity)
                all_negative_labels.extend(neg_sample_label)                                                                          

            all_positive_intensities = torch.stack(all_positive_intensities)
            all_negative_intensities = torch.stack(all_negative_intensities)

            loss = self.compute_loss(all_positive_intensities, all_negative_intensities)
            
            loss.backward()
            self.node_embeddings.weight.grad[self.emb_mask] = config.PAD_VALUE
            self.optimizer.step()

            reshaped_positive_intensities = torch.cat([t.view(-1) for t in all_positive_intensities])
            reshaped_negative_intensities = torch.cat([t.view(-1) for t in all_negative_intensities])
            all_intensities  = torch.cat([reshaped_positive_intensities, reshaped_negative_intensities])
            all_labels = torch.tensor(all_positive_labels + all_negative_labels, dtype=torch.long)

            accuracy = self.evaluate_accuracy(all_intensities, all_labels)
            print(f"Epoch {epoch}: Hawkes process-based accuracy for node embeddings = {accuracy.item():.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_embeddings = self.node_embeddings.weight.clone()
                print(f"Best accuracy improved to {accuracy.item():.4f}; current node embeddings saved as optimal.")
            
        self.hawkes_node_embeddings = best_embeddings.detach()
        torch.save(self.hawkes_node_embeddings, self.hawkes_embeddings_path)



    def evaluate_accuracy(self, intensities, labels):
        
        """
        Evaluate model performance. 

        Args:
            - intensities (Tensor): Predicted intensities.
            - labels (Tensor): Ground-truth binary labels.

        Returns:
            - Tensor: Accuracy value. 
        """
        
        preds = (intensities >= 0.5).float() 
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

        return torch.tensor([accuracy])



    def retrieve_hawkes_embeddings(self):
        
        """
        Retrieve the trained node embeddings.

        Returns:
            - Tensor: Embedding matrix of shape (num_nodes, embedding_dim).
        """
        
        return self.hawkes_node_embeddings
        
        
  
