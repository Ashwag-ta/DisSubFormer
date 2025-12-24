# General Imports

import sys
from pathlib import Path

# Ours
# Add parent directory to path to enable importing main config
sys.path.insert(0, '..')
import main_config as config

# Directory where data and results will be saved
DATA_RESULTS_DIR = Path(config.PROJECT_ROOT)

# Parameters for training node embeddings for PPI and GO-FS graphs
CONV_PPI = "GCN" # Convolution type used for the PPI graph
CONV_GO_FS = "GAT" # Convolution type used for the GO-FS graph
MINIBATCH = "NeighborLoader" # Mini-batching strategy for training

# Parameters search space for model tuning
EPOCHS = 100
POSSIBLE_BATCH_SIZES = [512, 1024]
POSSIBLE_HIDDEN = [128, 256]
POSSIBLE_OUTPUT = [64]
POSSIBLE_LR = [0.001, 0.005]
POSSIBLE_WD = [5e-4, 5e-5]
POSSIBLE_DROPOUT = [0.4, 0.5]
POSSIBLE_NB_SIZE = [-1]
POSSIBLE_GAT_HEADS_1 = [8]
POSSIBLE_GAT_HEADS_2 = [1]

# Random Seed
RANDOM_SEED = 3

# Flags for precomputing graph metrics (set to True to enable precomputation)
PRECOMPUTE_EUCLIDEAN_DISTANCES = True # Precompute Euclidean distances between node embeddings
PRECOMPUTE_EGO_GRAPHS = True # Precompute the ego graph for each node
PRECOMPUTE_SHORTEST_PATHS = True  # Precompute the pairwise shortest path lengths between all node pairs in the graph
PRECOMPUTE_INTERMEDIATE_NODES = True  # Precompute the number of intermediate nodes between all node pairs in the graph
PRECOMPUTE_ADJ_MATRIX = True  # Precompute the adjacency matrix for the graph
PRECOMPUTE_NODE_DEGREES = True # Precompute the degree of each node
OVERRIDE = False # If True, overwrite existing precomputed files
N_PROCESSSES = 10 # Number of cores to use for multi-processsing when precomputing graph metrics




