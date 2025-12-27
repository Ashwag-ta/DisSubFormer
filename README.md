# DisSubFormer
Repository for DisSubFormer: A Subgraph Transformer Model for Disease Subgraph Representation and Comorbidity Prediction

Authors: Ashwag Altayyar and Li Liao

To use DisSubFormer, follow these steps:
* Install the environment
* Prepare DisSubFormer input data
* Train and evaluate DisSubFormer


## Install the Environment
We provide a .yml file that includes all the required packages for training DisSubFormer. After installing [Conda](https://www.anaconda.com/docs/getting-started/anaconda/install), you can create the environment using the following command:
```bash
conda env create --file DisSubFormer_env.yml
```

## Prepare DisSubFormer Input Data
* Download the preprocessed data from [Dropbox – DisSubFormer preprocessed data](https://www.dropbox.com/scl/fo/z1zpdlxcm8ntdjet39xtb/ALnL4Kusqy_16XNJbu4hvcg?rlkey=zdl3vaky2gq76ioq4sctb0oxx&st=vo0jax9h&dl=0) and place the contents under `Data_Results/Data/`.  
  See the [Data README](Data_Results/Data/README.md) for details about preprocessed and raw data.
* Set `PROJECT_ROOT` in `main_config.py` to your local `Data_Results` path.
* (Optional) Precompute graph metrics needed for anchor patch sampling and multi-head attention computations:
```bash
python precompute_graph_metrics.py
```
* (Optional) Generate protein node embeddings:
```bash
python train_PPI_GO_FS.py
```

**Note:** The Dropbox package already includes the precomputed graph metrics and protein node embeddings required to run DisSubFormer.


## Train DisSubFormer
We provide two options for training DisSubFormer:

1. To train with Optuna hyperparameter optimization, run the following command:
```bash
python train_optuna_model.py --train_config_file train_config_optuna.json
```

2.  To train without Optuna, using fixed hyperparameters, run the following command:
```bash
python train_fixed_model.py --train_config_file train_config_fixed.json
```

3. During training, model checkpoint files for all epochs will be saved in the `Train_Results` directory. After training, the best model based on validation set performance will be selected, evaluated on the test data using a single random seed, and the evaluation results will be saved in the `test_results.json` file inside the `Train_Results` directory.


## Evaluate DisSubFormer
To test the trained DisSubFormer model using multiple random seed runs:

1. Save the `checkpoint_model.ckpt` and `hyperparameters.json` files for the best model from the `Train_Results` directory to the initialized `Test_Resources` directory.

2. Run the following command:
```bash
python test_model.py --model_file checkpoint_model.ckpt --test_config_file hyperparameters.json
```

3. The evaluation results will be saved in the `Test_Results` directory and will include:
* The test results of each random run and the mean results across all seed runs, saved in the `final_test_results.json` file.
* The ROC and PR curves for each individual seed run.
* Summary ROC and PR curves aggregating all runs, including the mean ROC and PR curves along with individual seed run curves.
  
**Note:** We provide the `checkpoint_model.ckpt` and `hyperparameters.json` files for the best model, saved in the `Test_Resources` directory, to ensure reproducibility when testing the model.







