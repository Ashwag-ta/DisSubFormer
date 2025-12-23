# General Imports
import os
import numpy as np
import random

# PyTorch and PyTorch Geometric
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling

# Ours
import dissubformer_input_config as config
import graph_preprocess as preprocess
import model as mdl
import utils

# Global Variables
RESULTS_TRAINING_DIR = None
all_data = None
dataset_name = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = 10e-4



# Initialize hyperparameter 
def initialize_hyperparameters():
    
    if config.MINIBATCH == "NeighborLoader":
         hyperparameter_space = {'batch_size': config.POSSIBLE_BATCH_SIZES, 'hidden': config.POSSIBLE_HIDDEN, 
                'output': config.POSSIBLE_OUTPUT, 'lr': config.POSSIBLE_LR, 'wd': config.POSSIBLE_WD, 
                'nb_size': config.POSSIBLE_NB_SIZE, 'dropout': config.POSSIBLE_DROPOUT}
                
         current_hyperparameters = {'batch_size': config.POSSIBLE_BATCH_SIZES[0], 'hidden': config.POSSIBLE_HIDDEN[0], 
                'output': config.POSSIBLE_OUTPUT[0], 'lr': config.POSSIBLE_LR[0], 'wd': config.POSSIBLE_WD[0], 
                'nb_size': config.POSSIBLE_NB_SIZE[0], 'dropout': config.POSSIBLE_DROPOUT[0]}
         
    return hyperparameter_space, current_hyperparameters



# Read and preprocess the PPI and GO_FS datasets.
def read_datasets():
    
    all_data_PPI = preprocess.read_dataset((str(config.DATA_RESULTS_DIR / 'Data' / "PPI.txt")), "PPI")
    
    all_data_GO_FS = preprocess.read_dataset((str(config.DATA_RESULTS_DIR / 'Data' / "GO_FS.txt")), "GO_FS",
                                              (str(config.DATA_RESULTS_DIR / 'Data' / "GO_FS_node_features_ANC2VEC.csv")))
    
    return all_data_PPI, all_data_GO_FS



# Set up the mini-batch loader using NeighborLoader
def setup_loader(all_data, current_hyperparameters):
    
    if config.MINIBATCH == "NeighborLoader":
        loader = NeighborLoader(all_data, input_nodes=None, num_neighbors=[current_hyperparameters['nb_size']],
                    batch_size=current_hyperparameters['batch_size'], shuffle=True)
        
    return loader



# Train the model for one epoch
def train_epoch(epoch, model, optimizer, loader):
    
    global all_data, dataset_name
    epoch_loss = 0
    auroc_score_val, acc_score_val, ap_score_val, f1_score_val = [], [], [], []

    # Iterate over mini-batches
    for batch in loader:
        
        batch = preprocess.set_batch_data(batch, all_data)
        batch.to(device)
        
        # Prepare training data
        curr_train_pos = batch.edge_index[:, batch.train_mask]
        curr_train_neg = negative_sampling(curr_train_pos, num_neg_samples=curr_train_pos.size(1) // 4)
        curr_train_total = torch.cat([curr_train_pos, curr_train_neg], dim=-1)
        curr_train_pos_mask = torch.zeros(curr_train_total.size(1)).bool()
        curr_train_pos_mask[:curr_train_pos.size(1)] = 1
        curr_train_neg_mask = (curr_train_pos_mask == 0)
        batch.y = torch.zeros(curr_train_total.size(1)).float()
        batch.y[:curr_train_pos.size(1)] = 1.
        edge_attr = batch.edge_attr if dataset_name == "GO_FS" else None

        # Forward pass and compute loss
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, edge_attr=edge_attr)
        train_logits = utils.compute_cosine_similarity(output, curr_train_total)
        train_prob = torch.sigmoid(train_logits)
        batch_loss = utils.compute_loss(batch, train_logits)

        # Backward pass
        if torch.isnan(batch_loss) == False: 
            epoch_loss += batch_loss
            batch_loss.backward()
        optimizer.step()

        # Compute training metrics 
        auroc_score_train, acc_score_train, ap_score_train, f1_score_train = utils.compute_evaluation_scores(train_prob, curr_train_pos_mask, curr_train_neg_mask, dataset_name)     
        print(f"Train Epoch {epoch+1}: (AUROC) {auroc_score_train:.4f}, (ACC) {acc_score_train:.4f}, (AP) {ap_score_train:.4f},  (F1) {f1_score_train:.4f}")

        # Validation on the current batch
        auroc_batch_val, acc_batch_val, ap_batch_val, f1_batch_val = validate_batch(output, batch, model)
        auroc_score_val.append(auroc_batch_val)
        acc_score_val.append(acc_batch_val) 
        ap_score_val.append(ap_batch_val) 
        f1_score_val.append(f1_batch_val) 

    return epoch_loss, auroc_score_val, acc_score_val, ap_score_val, f1_score_val



# Validate step on a batch
def validate_batch(output, batch, model):
    
    # Prepare validation data
    curr_val_pos = batch.edge_index[:, batch.val_mask]
    curr_val_neg = negative_sampling(curr_val_pos, num_neg_samples=curr_val_pos.size(1) // 4)
    curr_val_total = torch.cat([curr_val_pos, curr_val_neg], dim=-1)
    curr_val_pos_mask = torch.zeros(curr_val_total.size(1)).bool()
    curr_val_pos_mask[:curr_val_pos.size(1)] = 1
    curr_val_neg_mask = (curr_val_pos_mask == 0)
    batch.y = torch.zeros(curr_val_total.size(1)).float()
    batch.y[:curr_val_pos.size(1)] = 1.

    # Compute validation metrics
    val_logits = utils.compute_cosine_similarity(output, curr_val_total)
    val_prob = torch.sigmoid(val_logits)
    auroc_batch_val, acc_batch_val, ap_batch_val, f1_batch_val = utils.compute_evaluation_scores(val_prob, curr_val_pos_mask, curr_val_neg_mask, dataset_name)
    print(f"Validation: (AUROC) {auroc_batch_val:.4f}, (ACC) {acc_batch_val:.4f}, (AP) {ap_batch_val:.4f}, (F1) {f1_batch_val:.4f}")

    return auroc_batch_val,  acc_batch_val, ap_batch_val, f1_batch_val



# Log and save the best-performing model
def log_and_save_best_model (epoch, loss, model, auroc_score_val, acc_score_val, ap_score_val, f1_score_val):
    
    global dataset_name, best_val_acc, best_model, current_hyperparameters, best_hyperparameters, train_log_file

    result = (
        f"Val AUROC score = {np.mean(auroc_score_val):.4f}\t"
        f"Val Accuracy score = {np.mean(acc_score_val):.4f}\t"
        f"Val AP score = {np.mean(ap_score_val):.4f}\t"
        f"Val F1 score = {np.mean(f1_score_val):.4f}")
    
    print(f"=== Validation Results for Epoch {epoch + 1} ===\n")
    print(result)
    train_log_file.write(result + "\n")

    # Save the best model and hyperparameters
    if best_val_acc <= np.mean(acc_score_val) + eps:
        best_val_acc = np.mean(acc_score_val)
        with open(str( RESULTS_TRAINING_DIR / f"best_model_{dataset_name}.pth"), 'wb') as state_dict_file:
            torch.save(model.state_dict(), state_dict_file)
        best_hyperparameters = current_hyperparameters
        best_model = model



# Test the model on the test data
def test_model(model, best_embeddings):
    
    global all_data, dataset_name, train_log_file
    
    model.load_state_dict(torch.load(str( RESULTS_TRAINING_DIR / f"best_model_{dataset_name}.pth")))
    model.to(device)
    model.eval()

    # Prepare test data
    test_pos = all_data.edge_index[:, all_data.test_mask]
    test_neg = negative_sampling(test_pos, num_neg_samples=test_pos.size(1) // 4)
    test_total = torch.cat([test_pos, test_neg], dim=-1)
    test_pos_mask = torch.zeros(test_total.size(1)).bool()
    test_pos_mask[:test_pos.size(1)] = 1
    test_neg_mask = (test_pos_mask == 0)

    # Compute test metrics
    test_prob = utils.compute_cosine_similarity(best_embeddings, test_total, test=True)
    auroc_score_test,  acc_score_test, ap_score_test, f1_score_test = utils.compute_evaluation_scores(test_prob, test_pos_mask, test_neg_mask, dataset_name,
                                                                                                     RESULTS_TRAINING_DIR / f"plot_{dataset_name}.pdf")

    print(f'Test AUROC score: {auroc_score_test:.4f}')
    print(f'Test Accuracy score: {acc_score_test:.4f}')
    print(f'Test AP score: {ap_score_test:.4f}')
    print(f'Test F1 score: {f1_score_test:.4f}')

    train_log_file.write(f'Test AUROC score: {auroc_score_test:.4f}\n')
    train_log_file.write(f'Test Accuracy score: {acc_score_test:.4f}\n')
    train_log_file.write(f'Test AP score: {ap_score_test:.4f}\n')
    train_log_file.write(f'Test F1 score: {f1_score_test:.4f}\n')
 


def generate_embeddings(all_data_par, dataset_name_par):
    
    """
    Runs full training loop: hyperparameter tuning, training, validation, testing, and saving embeddings.
    """
    
    global RESULTS_TRAINING_DIR, all_data, dataset_name, train_log_file, best_val_acc, best_model, best_hyperparameters, current_hyperparameters, device
     
    all_data = all_data_par  
    dataset_name = dataset_name_par
    train_log_file = open(str( RESULTS_TRAINING_DIR / f"{dataset_name}_log.log"), "w")

    best_val_acc = -1
    best_model = None
    best_hyperparameters = {}
      
    # Initialize hyperparameters
    hyperparameter_space, current_hyperparameters  = initialize_hyperparameters()
    
    # Shuffle and iterate through different hyperparameter types
    shuffled_hyperparameter_categories = random.sample(list(hyperparameter_space.keys()), len(hyperparameter_space.keys()))
    for hyperparameter_category in shuffled_hyperparameter_categories:
        shuffled_hyperparameter_values = random.sample(hyperparameter_space[hyperparameter_category], len(hyperparameter_space[hyperparameter_category]))

        # Shuffle and iterate through the values for the current hyperparameter type
        for hyperparameter_value in shuffled_hyperparameter_values:
            
            # Set the current hyperparameter value for this iteration
            current_hyperparameters[hyperparameter_category] = hyperparameter_value
            print("Current Hyperparameters:" , current_hyperparameters)
            train_log_file.write(str(current_hyperparameters) + "\n")

            # Setup the model and optimizer
            if dataset_name == "PPI":
                model= mdl.GNNModel(all_data.x.shape[1], current_hyperparameters['hidden'], current_hyperparameters['output'],
                                    config.CONV_PPI, current_hyperparameters['dropout']).to(device)
                
            elif dataset_name == "GO_FS":
                model = mdl.GNNModel(all_data.x.shape[1], current_hyperparameters['hidden'], current_hyperparameters['output'],
                                     config.CONV_GO_FS, current_hyperparameters['dropout'], all_data.edge_attr.shape[1]).to(device)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=current_hyperparameters['lr'], weight_decay=current_hyperparameters['wd'])

            # Train the model
            model.train()
            epoch_losses = []
            for epoch in range(config.EPOCHS):
                loader = setup_loader(all_data, current_hyperparameters)
                loss, auroc_score_val, acc_score_val, ap_score_val, f1_score_val = train_epoch(epoch, model, optimizer, loader)
                epoch_losses.append(loss)
    
                # Log the training and validation results
                log_and_save_best_model(epoch, loss, model, auroc_score_val, acc_score_val, ap_score_val, f1_score_val)
            
            # Reset the current hyperparameter value to the best value found so far
            current_hyperparameters[hyperparameter_category] = best_hyperparameters[hyperparameter_category]
            
    print("Best Hyperparameters: ", best_hyperparameters)
    print("Optimization Finished!")
    train_log_file.write("Best Hyperparameters: %s \n" % best_hyperparameters)

    device = torch.device('cpu') # Optional: switch to CPU to retrieve the embeddings
    best_model = best_model.to(device)
    best_embeddings = utils.retrieve_embeddings(best_model, all_data, device)

    # Evaluate the model on the test set
    test_model(best_model, best_embeddings)

    # Save the embeddings
    if dataset_name == "PPI":
        torch.save(best_embeddings, str( RESULTS_TRAINING_DIR / f"{config.CONV_PPI}_embeddings_{dataset_name}.pth"))
    elif dataset_name == "GO_FS":
        torch.save(best_embeddings, str( RESULTS_TRAINING_DIR / f"{config.CONV_GO_FS}_embeddings_{dataset_name}.pth"))
 
    print(f"Saved model and embeddings for {dataset_name}")



def main():
    
    """
    Runs training and embedding generation for both PPI and GO_FS datasets.
    """
    
    global RESULTS_TRAINING_DIR

    # Set random seed for reproducibility
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create directory
    RESULTS_PARENT_DIR = config.DATA_RESULTS_DIR / "Results"
    RESULTS_TRAINING_DIR = RESULTS_PARENT_DIR / "GNN_Results"
    for path in [RESULTS_PARENT_DIR, RESULTS_TRAINING_DIR]:
        os.makedirs(path, exist_ok=True)
        
    # Read datasets (PPI and GO_FS)
    all_data_PPI, all_data_GO_FS = read_datasets()
    
    # Generate embeddings for PPI graph nodes
    print("Generating embeddings for PPI graph nodes...")
    generate_embeddings(all_data_PPI, "PPI")

    # Generate embeddings for GO_FS graph nodes
    print("Generating embeddings for GO_FS graph nodes...")
    generate_embeddings(all_data_GO_FS, "GO_FS")
    
    # Combine embeddings
    PPI_embeddings = torch.load(str(RESULTS_TRAINING_DIR / f"{config.CONV_PPI}_embeddings_PPI.pth"))
    GO_FS_embeddings = torch.load(str(RESULTS_TRAINING_DIR / f"{config.CONV_GO_FS}_embeddings_GO_FS.pth"))
    combined_embeddings = torch.cat((PPI_embeddings, GO_FS_embeddings), dim=-1)
    torch.save(combined_embeddings, str(config.DATA_RESULTS_DIR / 'Data' / "Combined_PPI_GO_FS_embeddings.pth"))
    print("Saved combined embeddings for PPI and GO_FS.")



if __name__ == "__main__":
    main()
