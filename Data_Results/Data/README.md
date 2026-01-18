# DisSubFormer Data
The data are provided in two forms:

---

## 1- Preprocessed Data (Required)
The preprocessed data are used by DisSubFormer during training and testing.
They were prepared by the authors and include graph structures, embeddings, and precomputed graph matrices required for biologically informed anchor patch sampling and multi-head attention.

**Download the preprocessed data from:**
[Dropbox – DisSubFormer preprocessed data](https://www.dropbox.com/scl/fo/z1zpdlxcm8ntdjet39xtb/ALnL4Kusqy_16XNJbu4hvcg?rlkey=zdl3vaky2gq76ioq4sctb0oxx&st=vo0jax9h&dl=0)

After downloading, place the data contents directly under:

```text
Data_Results/Data/
```

### Included Preprocessed Data

* DDI_RR0.txt
  
Contains a list of comorbid disease pairs in edge list format. Each line represents a pair of diseases identified as comorbid based on a relative risk (RR) score greater than 0.

* PPI.txt
  
Represents the Protein–Protein Interaction (PPI) network as an undirected graph. Each line specifies an undirected edge between two proteins.

* GO_FS.txt
  
Encodes the GO-based functional similarity (GO-FS) network as a weighted, undirected graph. Undirected edges connect protein pairs with GO-based functional similarity scores greater than 0.5, and edge weights correspond to the similarity scores.

* GO_FS_node_features_ANC2VEC.csv
  
Provides initial protein features for the GO-FS network. Each row contains a gene ID and its corresponding embedding derived from GO annotations of the protein product encoded by that gene using the ANC2VEC method.

* Subgraphs.pth
  
Stores disease-specific subgraphs derived from disease-gene associations. Each subgraph represents one of 299 diseases including the set of associated genes.

* Additional data
  
Includes precomputed files such as adjacency matrices and similarity scores that support anchor patch sampling and multi-head attention computations used during DisSubFormer training and testing.


## 2- Raw Data (Reference Only)
The raw data used to generate the preprocessed data are provided for reference in `Data_Results/Data/raw_data/`.

### Included Raw Data
* Disease–gene associations
* PPI interactions
* Gene ID mapping
* Disease ID mapping
* Disease pairs
* GO annotations

### Original Data Sources
The raw data originate from the following publications:

* Menche, J., et al. (2015). Uncovering disease–disease relationships through the incomplete interactome.
Science, 347(6224), 1257601. https://doi.org/10.1126/science.1257601

* Edera, A.A., Milone, D.H., Stegmayer, G. (2022). Anc2vec: Embedding gene ontology terms by preserving ancestors relationships.
Briefings in Bioinformatics, 23(2), bbac003. https://doi.org/10.1093/bib/bbac003


**Important:** DisSubFormer expects the preprocessed data. The files in `raw_data/` are provided for reference and are not sufficient to run training/testing.



