# PepXML

Prediction of antimicrobial peptide-targeted pathogens based on deep extreme multi-label classification.



## Related Files

| **File Name**               | **Description**                                              |
| :-------------------------- | :----------------------------------------------------------- |
| main.py                     | The main file of PepXML predictor (include cluster, extract peptide features, hard negative sampling, training, prediction, evaluation, etc.) |
| Node2Vec.py                 | Construct a co-occurrence graph of labels and use node2vec to obtain the embedding of each label. |
| K_Means.ipynb               | Using the elbow method of the K_Means algorithm to choose the optimal number of clusters. |
| K_Means.py                  | Cluster the labels using the K_Means algorithm based on the optimal number of clusters. |
| EachClusterSeqLabMatrix.py  | The peptides-labels data for each cluster is obtained on the original peptides -labels data and labels-clusters data. |
| ESM2ClassifierforCluster.py | In each cluster, extract the embedding representation of peptides using ESM2 and train an MLP classifier, while also incorporating hard negative sampling. |
| MergePredictions.py         | Merge the predictions for each cluster.                      |
| Evaluation.py               | Evaluate metrics (for evaluating prediction results).        |
| Utils.py                    | It is used for ESM2 to sample features and hard negative sampling. |
| datasets                    | Data.                                                        |
| generate_data               | Data generated during training.                              |

## Installation

- Requirement

  `OS`:

  - `Windows`：Windows10 or later

  - `Linux`：Ubuntu 16.04 LTS or later

  `Python`:

  - `python==3.8`

- Download `PepXML` to your computer

```Linux
git https://github.com/Family-of-four/PepXML.git
```

- Open the `dir` and install `environment.yaml` with `conda`

```Linux
cd PepXML
conda env create -f environment.yaml
```

## Construct a label co-occurrence matrix and embed labels

```Linux
python Node2Vec.py --label_matrix_path datasets/label_matrix.csv --output_path generate_data/label_embeddings.csv --dimensions 100 --walk_length 80 --num_walks 10 --workers 4 --seed 42 --log_file log_file/node2vec.log
```



`K_Means.ipynb`--Read in the label embedding file `generate_data/label_embeddings.csv`. Plot the `SSE` curve of the K-Means clustering process, determine the optimal number of clusters according to the elbow method, and draw the corresponding `UMAP` visualization.



## Training and test PepXML model

```
python main.py
```







