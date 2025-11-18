# DisSubFormer

Repository for DisSubFormer: A Subgraph Transformer Model for Disease Subgraph Representation and Comorbidity Prediction

Authors: Ashwag Altayyar and Li Liao

To use DisSubFormer, follow these steps:
* Install the environment
* Prepare DisSubFormer input data
* Modify PROJECT_ROOT in main_config.py
* Train and evaluate DisSubFormer



Install the Environment:
We provide a .yml file that includes all the required packages for training DisSubFormer. After installing Conda, you can create the environment using the following command:

```bash
conda env create --file DisSubFormer_env.yml
```

Prepare DisSubFormer Input Data:
* Download the provided datasets to prepare the input data for DisSubFormer.
* Set PROJECT_ROOT in main_config.py to the path where the data is downloaded.


Note: The input data for DisSubFormer has already been prepared in the Data_Results directory. We have provided the node embeddings and other files, including precomputed graph matrices, required for sampling biologically informed anchor patches and for computing the head-specific relational terms necessary for training.



Train DisSubFormer:
We provide two options for training DisSubFormer:

1- To train with Optuna hyperparameter optimization, run the following command:

```bash
python train_optuna_model.py --train_config_file train_config_optuna.json
```

2-  To train without Optuna, using fixed hyperparameters, run the following command:

```bash
python train_fixed_model.py --train_config_file train_config_fixed.json
```

3- During training, model checkpoint files for all epochs will be saved in the Train_Results directory. After training, the best model based on validation set performance will be selected, evaluated on the test data using a single random seed, and the evaluation results will be saved in the test_results.json file inside the Train_Results directory.



Evaluate DisSubFormer:
To test the trained DisSubFormer model using multiple random seed runs:

1- Save the checkpoint_model.ckpt and hyperparameters.json files for the best model from the Train_Results directory to the initialized Test_Resources directory.

2- Run the following command:

python test_model.py --model_file checkpoint_model.ckpt --test_config_file hyperparameters.json


3- The evaluation results will be saved in the Test_Results directory and will include:
* The test results of each random run and the mean results across all seed runs, saved in the final_test_results.json file.
* The ROC and PR curves for each individual seed run.
* Summary ROC and PR curves aggregating all runs, including the mean ROC and PR curves along with individual seed run curves.


Note: We provide the checkpoint_model.ckpt and hyperparameters.json files for the best model, saved in the Test_Resources directory, to ensure reproducibility when testing the model.



Data:
1- Processed Data (download from Dropbox):
These are the files used directly by DisSubFormer during training/testing:
* DDI_RR0.txt
Contains a list of comorbid disease pairs in edge list format. Each line represents a pair of diseases identified as comorbid based on a relative risk (RR) score greater than 0.
* PPI.txt
Represents the Protein-Protein Interaction (PPI) network as an undirected edge list. Each line specifies an interaction between two proteins.
* GO_FS.txt
Encodes a protein similarity graph constructed from Gene Ontology (GO)-based functional similarity. Edges connect protein pairs with functional similarity scores greater than 0.5, indicating the edge weights.
* GO_FS_node_features_ANC2VEC.csv
Provides precomputed node features for the GO_FS graph. Each row contains a gene ID and its corresponding embedding derived from GO annotations of the protein product encoded by that gene using the ANC2VEC method.
* Subgraphs.pth
Stores disease-specific subgraphs derived from disease-gene associations. Each subgraph represents one of 299 diseases including the set of associated genes.
* Additional files
Includes precomputed files such as adjacency matrices and similarity scores that support anchor patch sampling and multi-head attention computations used during DisSubFormer training.

Download the complete processed dataset from the following link: 
[Dropbox – DisSubFormer processed data](https://www.dropbox.com/scl/fo/z1zpdlxcm8ntdjet39xtb/ALnL4Kusqy_16XNJbu4hvcg?rlkey=zdl3vaky2gq76ioq4sctb0oxx&st=vo0jax9h&dl=0)

After downloading, place all processed files under: Data_Results/Data/

2- Raw Data Sources (included in GitHub):
Raw datasets used to generate the processed files are stored in: Data_Results/Data/raw_data/

These include:
* Disease–gene associations
* PPI interactions
* Gene ID mapping
* Disease ID mapping
* Disease pairs
* GO annotations

All raw data originate from:
* Menche, J., et al. (2015). Uncovering disease–disease relationships through the incomplete interactome.
Science, 347(6224), 1257601. https://doi.org/10.1126/science.1257601
GO annotations 
* Edera, A.A., Milone, D.H., Stegmayer, G. (2022). Anc2vec: Embedding gene ontology terms by preserving ancestors relationships.
Briefings in Bioinformatics, 23(2), bbac003. https://doi.org/10.1093/bib/bbac003




