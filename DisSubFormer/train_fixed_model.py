# General Imports
import os
import sys
import pathlib
from collections import OrderedDict
import random
import argparse
import commentjson
import numpy as np
import json
from pathlib import Path

# PyTorch
import torch

# Pytorch lightning
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback

# Ours
import DisSubFormer as mdl
sys.path.insert(0, '..')
import main_config as config



def get_config_path():
    
    """
    Parse the command-line argument for the configuration file path.
    """
    
    argument_parser = argparse.ArgumentParser(description="Specify configuration file for training DisSubFormer.")
    argument_parser.add_argument('--train_config_file', '-c', type=str, help='Path to the training configuration file.', required=True)
    parsed_args = argument_parser.parse_args()
    
    return parsed_args



def load_config_file(file_path):

    """
    Load and parse a JSON training configuration file.
    
    Args:
        - file_path (str): Path to the JSON configuration file.

    Returns:
        - config_data(OrderedDict): Parsed configuration data.
    """
    
    with open(file_path, 'rt') as file_handle:
        config_data = commentjson.load(file_handle, object_hook=OrderedDict)
        
    return config_data



def get_hyperparameters(train_config):
    
    """
    Retrieves fixed hyperparameters from the training configuration.

    Args:
        - train_config (dict): Configuration dictionary containing 'fixed_hyperparameters' and 'optuna_hyperparameters'.
        
    Returns:
        - dict: Dictionary of fixed hyperparameters, where each key is a hyperparameter name and the value is its value.
    """
    
    fixed_hyperparameters = dict(train_config["fixed_hyperparameters"])
    
    return fixed_hyperparameters    



def build_model(train_config):
    
    """
    Create DisSubFormer using the specified configuration.
    
    Returns:
        - model (DisSubFormer): Initialized DisSubFormer model with loaded configuration.
    """

    # Get fixed hyperparameters 
    hyperparameters = get_hyperparameters(train_config)
    
    # Set random seeds for reproducibility
    random.seed(hyperparameters['seed'])
    np.random.seed(hyperparameters['seed'])
    torch.manual_seed(hyperparameters['seed'])
    torch.cuda.manual_seed(hyperparameters['seed'])
    torch.cuda.manual_seed_all(hyperparameters['seed']) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the DisSubFormer model with paths and hyperparameters
    model = mdl.DisSubFormer(
        hyperparameters,                                        # Model hyperparameters
        train_config["PPI_graph_path"],                         # Path to PPI graph
        train_config["DDI_RR0_graph_path"],                     # Path to DDI RR0 graph
        train_config["Subgraphs_path"],                         # Path to subgraph data
        train_config["Embedding_path"],                         # Path to embedding file
        train_config["AP_sampling_similarities_path"],          # Path for anchor patch sampling similarities
        train_config["Head_attention_similarities_path"],       # Path for head attention similarities
        train_config['Ego_graph_path'],                         # Path to ego graph dictionary
        train_config["Euclidean_distances_path"],               # Path to Euclidean distances matrix
        train_config["Shortest_paths_path"],                    # Path to shortest path matrix
        train_config["Intermediate_nodes_path"],                # Path to intermediate nodes matrix
        train_config['ADJ_matrix_path'],                        # Path to adjacency matrix
        train_config['Node_degrees_path']                       # Path to node degrees file
        )
    
    return model, hyperparameters



def build_trainer(train_config, hyperparameters):
    
    """
    Create a PyTorch Lightning Trainer with TensorBoard logging.
    """

    enable_progress_bar = bool(hyperparameters.get('enable_progress_bar', True)) 

    # Set up TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=train_config['tensorboard_dir'],
        name=None,
        version="version_" + str(random.Random().randint(0, 1000000000)))
    if not os.path.exists(logger.log_dir):
        os.makedirs(logger.log_dir, exist_ok=True)
   
    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir),
        filename="{epoch}-{val_average_precision:.2f}-{val_accuracy:.2f}-{val_auroc:.2f}",
        save_top_k=-1, # Save all models
        verbose=True,
        monitor=hyperparameters['monitor_metric'],
        mode='max')
 
    # Trainer configuration
    trainer_kwargs = {
        'max_epochs': hyperparameters['max_epochs'],
        "accelerator": "cpu",
        #"accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1, 
        "num_sanity_val_steps": 0,
        "gradient_clip_val": hyperparameters['grad_clip'],
        "enable_progress_bar": enable_progress_bar,
        "callbacks": [checkpoint_callback],
        "logger": logger,
        }
    
    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer, trainer_kwargs, logger.log_dir



def train_model(train_config):
    
    """
    Train, validate, and test the model using the specified configuration.
    """

    #  Build the model and retrieve the hyperparameters
    model, hyperparameters = build_model(train_config)

    # Initialize the trainer and logging configuration
    trainer, trainer_kwargs, results_path = build_trainer(train_config, hyperparameters)

    # Save hyperparameters to the results directory
    with open(os.path.join(results_path, "hyperparameters.json"), "w") as hyperparameters_file:
        hyperparameters_file.write(json.dumps(hyperparameters, indent=4))

    # Save trainer configurations to the results directory
    with open(os.path.join(results_path, "trainer_config.json"), "w") as tkwarg_file:
            pop_keys = [key for key in ['logger','profiler','early_stop_callback','callbacks'] if key in trainer_kwargs.keys()]
            for key in pop_keys:
                trainer_kwargs.pop(key, None) 
            tkwarg_file.write(json.dumps(trainer_kwargs, indent=4))
            
    # Start training
    print(" >>> Start Training the Model <<< ")
    trainer.fit(model)
    print("\n\n>>> Finished Training the Model <<<", flush=True)

    # Compute the best validation metric
    best_val_metrics = max(model.val_metric_scores, key=lambda x: x[hyperparameters['monitor_metric']])
    print("\n >>> Best Model Performance on Validation Set <<< ")
    print(f"Achieved at Epoch {best_val_metrics['epoch']}:")
    for metric, value in best_val_metrics.items():
        if metric != 'epoch':
            print(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")

    # Test the model
    print("\n >>> Testing the Best Model (Once after all training epochs) <<< ")
    trainer.test(model)
    
    # Save test results to file
    test_results = model.test_epoch_metrics
    filtered_test_metrics = {
        'test_loss': test_results['test_loss'],
        'test_accuracy': test_results['test_accuracy'],
        'test_f1': test_results['test_f1'],
        'test_average_precision': test_results['test_average_precision'],
        'test_auroc': test_results['test_auroc']}
    with open(os.path.join(results_path, "test_results.json"), "w") as test_result_file:
            json.dump(filtered_test_metrics, test_result_file, indent=4)
    print("\n\n>>> Finished Testing the Model and Test Results Saved <<<", flush=True)

    return best_val_metrics


    
def main():

    """
    Perform the training and evaluation pipeline using fixed hyperparameters
    and configuration paths specified in the provided train_config_file.
    """
    
    torch.autograd.set_detect_anomaly(True)
    config_args  = get_config_path()
    
    # Set up base directories for data and results
    base_data_path = Path(config.PROJECT_ROOT) / "Data"
    base_result_path = Path(config.PROJECT_ROOT) / "Results" / "DisSubFormer_Results"
    base_result_path.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = base_result_path / "Train_Results" / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    test_resources_dir = base_result_path / "Test_Resources"
    test_resources_dir.mkdir(parents=True, exist_ok=True)

    # Load training configuration from JSON file
    train_config = load_config_file(config_args.train_config_file)

    # Assign file paths for required data
    train_config['tensorboard_dir'] = str(tensorboard_dir)
    train_config["PPI_graph_path"] = os.path.join(base_data_path, "PPI.txt")
    train_config["DDI_RR0_graph_path"] = os.path.join(base_data_path, "DDI_RR0.txt")
    train_config["Subgraphs_path"] = os.path.join(base_data_path, "Subgraphs.pth")
    train_config["Embedding_path"] = os.path.join(base_data_path, "Combined_PPI_GO_FS_embeddings.pth")
    train_config["AP_sampling_similarities_path"] = os.path.join(base_data_path, "AP_sampling_similarities")
    train_config["Head_attention_similarities_path"] = os.path.join(base_data_path, "Head_attention_similarities")
    train_config["Ego_graph_path"] = os.path.join(base_data_path, "Ego_graphs.txt")
    train_config["Euclidean_distances_path"] = os.path.join(base_data_path, "Euclidean_distances_matrix.npy")
    train_config["Shortest_paths_path"] = os.path.join(base_data_path, "Shortest_paths_matrix.npy")
    train_config["Intermediate_nodes_path"] = os.path.join(base_data_path, "Intermediate_nodes_matrix.npy")
    train_config["ADJ_matrix_path"] = os.path.join(base_data_path, "ADJ_matrix.npy")
    train_config["Node_degrees_path"] = os.path.join(base_data_path, "Node_degrees.txt")

    train_model(train_config)


        
if __name__ == "__main__":
    main()
  
