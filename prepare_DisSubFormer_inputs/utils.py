# PyTorch 
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

# Matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Sci-kit Learn 
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve, precision_recall_curve



def compute_loss(data, cos_sim):
    
    """
    Compute the binary cross-entropy loss for link prediction using cosine similarity logits.

    Args:
        - data (Data object): Input PyG batch Data object.
        - cos_sim (Tensor): Cosine similarity logits (one logit per edge).

    Returns:
        - loss (Tensor): Computed loss.
    """
    
    loss_fn = BCEWithLogitsLoss()
    loss = loss_fn(cos_sim, data.y.to(cos_sim.device).float())
    
    return loss



def compute_cosine_similarity(embeddings, edge_pairs, test=False):
    
    """
    Compute cosine similarity between node pairs for link prediction.

    Args:
        - embeddings (Tensor): Node embeddings.
        - edge_pairs (Tensor): List of edges.
        - test (bool): If True, return only the sigmoid scores.

    Returns:
        - cos_sim (Tensor): Computed cosine similarities.
    """

    source_embeddings = torch.index_select(embeddings, 0, edge_pairs[0, :])
    target_embeddings = torch.index_select(embeddings, 0, edge_pairs[1, :])
    cos_sim = F.cosine_similarity(source_embeddings, target_embeddings, dim=-1)
    
    if test: 
        return torch.sigmoid(cos_sim)
    
    return cos_sim



def compute_evaluation_scores(pred_all, pos_edges, neg_edges, dataset_name, save_plots=""):

    """
    Compute AUROC, Accuracy, Average Precision, and F1 for link prediction.

    Args:
        - pred_all (Tensor): Predicted scores (one score per edge).
        - pos_edges (Tensor): Mask selecting positive edges.
        - neg_edges (Tensor): Mask selecting negative edges.
        - dataset_name (str): Dataset identifier (PPI or GO_FS) for labeling plots.
        - save_plots (str): Optional path to save ROC and PR plots as a PDF.

    Returns:
        - AUROC, Accuracy, Average Precision, F1.
    """
    
    pred_pos = pred_all[pos_edges]
    pred_neg = pred_all[neg_edges]
    pred_pos_neg = torch.cat((pred_pos, pred_neg), 0).cpu().detach().numpy()
    true_pos_neg = torch.cat((torch.ones(len(pred_pos)), torch.zeros(len(pred_neg))), 0).cpu().detach().numpy()

    # Compute metrics
    auroc = roc_auc_score(true_pos_neg, pred_pos_neg)
    acc = accuracy_score(true_pos_neg, (pred_pos_neg > 0.5))
    ap_score = average_precision_score(true_pos_neg, pred_pos_neg)
    f1 = f1_score(true_pos_neg, (pred_pos_neg > 0.5))
    
    # Plot if path is provided
    if save_plots != "":
        plot_roc_ap(true_pos_neg, pred_pos_neg, dataset_name, save_plots)
        
    return auroc, acc, ap_score, f1



def plot_roc_ap(y_true, y_pred, dataset_name, save_plots):

    """
    Plot ROC and Precision-Recall curves and save them to a PDF.
    """
    
    with PdfPages(save_plots) as pdf:

        # ROC 
        fpr = dict()
        tpr = dict()
        auroc = dict()
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_pred)
        plt.plot(fpr, tpr, label = "AUROC = {:.4f}".format(auroc))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.title(f"ROC for {dataset_name}")
        pdf.savefig()
        plt.close()

        # Precision-Recall curve
        precision = dict()
        recall = dict()
        ap = dict()
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        n_true = sum(y_true)/len(y_true)
        plt.plot(recall, precision, label = "AP = {:.4f}".format(ap))
        plt.plot([0, 1], [n_true, n_true], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.title(f"Precision-Recall Curve for {dataset_name}")
        pdf.savefig()
        plt.close()

        return auroc, ap


    
@torch.no_grad()
def retrieve_embeddings(model, all_data, device):
    
    """
    Retrieve the final node embeddings from the best trained model.

    Args:
        - model (torch model object): Trained GNN model.
        - all_data (Data object): The PyG input graph Data object.
        - device (torch device object): Device to run the model on (cpu or cuda).

    Returns:
        - best_embeddings (Tensor): Tensor containing the final node embeddings.
    """
    
    model.eval() 
    all_data = all_data.to(device) 
    best_embeddings = model(all_data.x, all_data.edge_index) 

    return best_embeddings 
