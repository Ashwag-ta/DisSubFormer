# DisSubFormer Data Preparation
This directory documents the datasets used in **DisSubFormer**.  
The data are divided into two categories:

---

## 1- Processed Data (Required)
These files are **directly used by DisSubFormer** during training and testing.
They have been preprocessed by the authors and include graph structures, embeddings, and precomputed graph matrices required for biologically informed anchor patch sampling and multi-head attention.

**Download the complete processed dataset from:**
[Dropbox – DisSubFormer processed data](https://www.dropbox.com/scl/fo/z1zpdlxcm8ntdjet39xtb/ALnL4Kusqy_16XNJbu4hvcg?rlkey=zdl3vaky2gq76ioq4sctb0oxx&st=vo0jax9h&dl=0)

After downloading, place all files directly under:

```text
Data_Results/Data/
```

### Included Processed Files

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
Includes precomputed files such as adjacency matrices and similarity scores that support anchor patch sampling and multi-head attention computations used during DisSubFormer training and testing.


## 2- Raw Data (Reference Only)
Raw datasets used to generate the processed data are provided for reference, transparency, and reproducibility and are stored in: `Data_Results/Data/raw_data/`.

These raw files are not used directly by the DisSubFormer training/testing code.

### Included Raw Data Sources
* Disease–gene associations
* PPI interactions
* Gene ID mapping
* Disease ID mapping
* Disease pairs
* GO annotations

## Original Data Sources
The raw data originate from the following publications:

* Menche, J., et al. (2015). Uncovering disease–disease relationships through the incomplete interactome.
Science, 347(6224), 1257601. https://doi.org/10.1126/science.1257601

* Edera, A.A., Milone, D.H., Stegmayer, G. (2022). Anc2vec: Embedding gene ontology terms by preserving ancestors relationships.
Briefings in Bioinformatics, 23(2), bbac003. https://doi.org/10.1093/bib/bbac003


**Important:**  DisSubFormer expects the processed dataset. The files in `raw_data/` are provided for reference and are not sufficient to run training/testing.



