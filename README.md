# Learning Genomic Sequence Representations using Graph Neural Networks over De Bruijn Graph
Repository for the paper ["Learning Genomic Sequence Representations using Graph Neural Networks over De Bruijn Graph"](https://arxiv.org/abs/2312.03865) in PyTorch. ([paper](https://arxiv.org/abs/2312.03865))
<div style="display: flex; justify-content: space-between;">
    <div style="flex: 1; padding: 5px;">
        <img width="500" alt="Self Supervised Task" src="https://github.com/ratschlab/genomic-gnn/assets/45125008/2decbe1f-b0a6-452a-9c11-ae6427c70136">
        <p><strong>Method overview: Self Supervised Task</strong></p>
    </div>
</div>


## Table of Contents

- [Usage](#usage)
  - [Environment Setup](#environment-setup)
  - [Edit Distance Approximation](#edit-distance-approximation)
    - [Our Contrastive Learning](#our-contrastive-learning)
    - [Baseline Methods](#baseline-methods)
  - [Closest String Retrieval](#closest-string-retrieval)
    - [Our Contrastive Learning](#our-contrastive-learning-2)
    - [Baseline Methods](#baseline-methods-2)
  - [Device Specification](#device-specification)
- [Appendices](#appendices)
  - [Appendix A: Scalable K-mer Graph Construction](#appendix-a-scalable-k-mer-graph-construction)
  - [Appendix C: Analysis of Graph Autoencoder as a Self-Supervised Task](#appendix-c-analysis-of-graph-autoencoder-as-a-self-supervised-task)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [References](#references)


## Usage
### Environment Setup
Install the required packages using the following command:
```
conda env create -f environment.yml
conda activate metagenomic_representation_learning
```
### Edit Distance Approximation
The Edit Distance Approximation task is initialized using one of the embeddings and then fine-tuned with a single linear layer.

#### Our Contrastive Learning
Without minibatching:
```
# k = 3
python -m src.train.main_editDistance --data datasets/edit_distance/edit_qiita_large.pkl --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 3 --representation_small_k 2 --representation_ss_hidden_channels 32_DB,32_KF0 --representation_ss_last_layer_edge_type DB --representation_size 32 --model_class mlp

# k = 4
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_ssgnn --representation_k 4 --representation_small_k 2,3 --representation_ss_task CL --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 2000 --representation_size 64 --model_class mlp

# k = 5
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_ssgnn --representation_k 4 --representation_small_k 2,3,4 --representation_ss_task CL --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 1000 --representation_size 64 --model_class mlp

# k = 6
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_ssgnn --representation_k 6 --representation_small_k 2,5 --representation_ss_task CL --representation_ss_hidden_channels 128_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 500 --representation_size 64 --representation_ss_edges_keep_top_k 0.01 --model_class mlp
```
With minibatching:
```
# k = 7
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_ssgnn_miniBatch --representation_k 7 --representation_small_k 2,5 --representation_ss_task CL --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 100 --representation_size 64 --representation_ss_edges_keep_top_k 0.01 --representation_ss_edges_threshold 0.8 --representation_ss_batch_size 1024 --model_class mlp

# k = 8
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_ssgnn_miniBatch --representation_k 8 --representation_small_k 2,5 --representation_ss_task CL --representation_ss_hidden_channels 32_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 100 --representation_size 32 --representation_ss_edges_keep_top_k 0.01 --representation_ss_edges_threshold 0.8 --representation_ss_batch_size 256 --model_class mlp

```
For full list of hyperparameters see [`src/train/parsers.py`](src/train/parsers.py).

#### Baseline Methods
Use the `--representation_data`  flag to specify the dataset path, and the `--representation_k` flag for the desired k value.
```
# OneHot
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_onehot --representation_k 3 --model_class mlp

# Word2Vec
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_word2vec --representation_k 3 --model_class mlp

# Node2Vec
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_node2vec --representation_k 3 --model_class mlp
```
For full list of hyperparameters see [`src/train/parsers.py`](src/train/parsers.py).

### Closest String Retrieval

#### Our Contrastive Learning
<a id="our-contrastive-learning-2"></a>
For  method fine-tuned on Edit Distance Approximation, without minibatching, use:
```
# k = 3
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 3 --representation_small_k 2 --representation_ss_hidden_channels 32_DB,32_KF0 --representation_ss_last_layer_edge_type DB --representation_size 32 --model_class cnn1d

# k = 4
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 4 --representation_small_k 2,3 --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 2000 --representation_size 64 --model_class cnn1d

# k = 5
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 5 --representation_small_k 2,3,4 --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 1000 --representation_size 64 --model_class cnn1d

# k = 6
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 6 --representation_small_k 2,5 --representation_ss_hidden_channels 128_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 500 --representation_size 64 --representation_ss_edges_keep_top_k 0.01 --model_class cnn1d
```

For  method fine-tuned on Edit Distance Approximation, with minibatching, use:
```
# k = 7
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_ssgnn_miniBatch --representation_ss_task CL --representation_k 7 --representation_small_k 2,5 --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 100 --representation_size 64 --representation_ss_edges_keep_top_k 0.01 --representation_ss_edges_threshold 0.8 --representation_ss_batch_size 1024 --model_class cnn1d --representation_ss_batch_size 1024

# k = 8
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_ssgnn_miniBatch --representation_ss_task CL --representation_k 8 --representation_small_k 2,5 --representation_ss_hidden_channels 32_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 100 --representation_size 32 --representation_ss_edges_keep_top_k 0.01 --representation_ss_edges_threshold 0.8 --representation_ss_batch_size 256 --model_class cnn1d --representation_ss_batch_size 256

```


For zero-shot method, without minibatching, use:
```
# k = 3
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 3 --representation_small_k 2 --representation_ss_hidden_channels 32_DB,32_KF0 --representation_ss_last_layer_edge_type DB --representation_size 32

# k = 4
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 4 --representation_small_k 2,3 --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 2000 --representation_size 64

# k = 5
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 5 --representation_small_k 2,3,4 --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 1000 --representation_size 64

# k = 6
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_ssgnn --representation_ss_task CL --representation_k 6 --representation_small_k 2,5 --representation_ss_hidden_channels 128_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 500 --representation_size 64 --representation_ss_edges_keep_top_k 0.01
```
For zero-shot method, with minibatching, use:
```
# k = 7
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_ssgnn_miniBatch --representation_ss_task CL --representation_k 7 --representation_small_k 2,5 --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 100 --representation_size 64 --representation_ss_edges_keep_top_k 0.01 --representation_ss_edges_threshold 0.8 --representation_ss_batch_size 1024

# k = 8
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_ssgnn_miniBatch --representation_ss_task CL --representation_k 8 --representation_small_k 2,5 --representation_ss_hidden_channels 32_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 100 --representation_size 32 --representation_ss_edges_keep_top_k 0.01 --representation_ss_edges_threshold 0.8 --representation_ss_batch_size 256

```
For full list of hyperparameters see [`src/train/parsers.py`](src/train/parsers.py).


#### Baseline Methods
<a id="baseline-methods-2"></a>
Use the `--representation_data`  flag to specify the dataset path, `--retrieval_data` flag to specify the retrieval dataset path, and the `--representation_k` flag for the desired k value. For method fine-tuned on Edit Distance Approximation, use:
```
# OneHot
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_onehot --representation_k 3 --model_class cnn1d

# Word2Vec
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_word2vec --representation_k 3 --model_class cnn1d

# Node2Vec
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --representation_method kmer_node2vec --representation_k 3 --model_class cnn1d
```
For zero-shot method, use:
```
# OneHot
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_onehot --representation_k 3 --model_class cnn1d

# Word2Vec
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_word2vec --representation_k 3 --model_class cnn1d

# Node2Vec
python -m src.train.main_editDistance --data path_to_dataset --retrieval_data path_to_retrieval_dataset --zero_shot_retrieval --representation_method kmer_node2vec --representation_k 3 --model_class cnn1d
```

For full list of hyperparameters see [`src/train/parsers.py`](src/train/parsers.py).

### Device Specification
To specify the device type, use the `--accelerator` flag. For example, to use a GPU, enter `--accelerator gpu`.


## Appendices

### Appendix A: Scalable K-mer Graph Construction

To use FAISS for approximate nearest neighbor search [[3]](#references) instead of cosine similarity to find nodes with close sub-k-mer frequency vectors, use the flag `--representation_ss_faiss_ann`. For example, in the case of Edit Distance Approximation:

``` 
# k = 10
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_ssgnn_miniBatch --representation_ss_faiss_ann --representation_ss_faiss_distance L2 --representation_ss_edges_keep_top_k 0.00008 --representation_k 10 --representation_small_k 2,5 --representation_ss_task CL --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 3 --representation_size 32 --model_class mlp

# k = 30
python -m src.train.main_editDistance --data path_to_dataset --representation_method kmer_ssgnn_miniBatch --representation_ss_faiss_ann --representation_ss_faiss_distance IP --representation_ss_edges_keep_top_k 0.00001 --representation_k 30 --representation_small_k 2,5 --representation_ss_task CL --representation_ss_hidden_channels 64_KF0 --representation_ss_last_layer_edge_type DB --representation_ss_epochs 3 --representation_size 32 --model_class mlp
```

Only supported for ```--representation_ss_task CL``` (Contrastive Learning) and ```--representation_method kmer_ssgnn_miniBatch```

### Appendix C: Analysis of Graph Autoencoder as a Self-Supervised Task
To use Graph Autoencoder, replace the 'CL' with 'AE' in flag ```--representation_ss_task```: ```--representation_ss_task AE```.





## Datasets

Our tasks and datasets for Edit Distance Approximation and Closest String Retrieval were taken from Corso et al. [[1]](#references). The datasets can be obtained directly from [the official repository of that paper](https://github.com/gcorso/NeuroSEED). The directories of the datasets can be used directly with our flags `--data` and `--retrieval_data`.

&ast;_Supplementary Content: Outside the research scope of this paper:_&ast;our Gene Prediction task and datasets were taken from Silva et al. [[2]](#references).


## Repository structure
```
.
├── README.md                               
├── environment.yml                      # conda env              
└── src                                  
    ├── downstream_tasks                 # Folder defining downstream tasks
    │   ├── coding_metagenomics          # Supplementary Folder: *Outside the research scope of the paper*         
    │   │   ├── cnn1d.py
    │   │   ├── coding_datasets.py
    │   │   └── train.py
    │   ├── datasets_factory             # Reading datasets
    │   │   ├── coding_metagenomics.py   # Supplementary File: *Outside the research scope of the paper*
    │   │   └── edit_distance.py         # Reading datasets from Corso et al. [1]
    │   └── edit_distance_models         # EDIT DISTANCE APPROXIMATION and CLOSEST STRING RETRIEVAL tasks
    │       ├── cnn1d.py                 
    │       ├── distance_datasets.py    
    │       ├── distances.py             # Hyperbolic Function
    │       ├── mlp.py                   # Single Linear Layer by default
    │       ├── retrieval_test.py        # Closest String Retrieval tests
    │       ├── train.py                 # Edit Distance Approximation
    │       └── zero_shot_model.py       # Concat, Mean, Max of k-mer embeddings
    ├── representations
    │   ├── gnn_common                   # Models and Utils for Our Contrastive Learning Method
    │   │   ├── gnn_models.py            # GNN, other models *Outside the research scope of the paper*
    │   │   └── gnn_utils.py             # Edge Computations, including FAISS method
    │   ├── gnn_tasks                    # Without Mini-Batching/Neighborhood sampling
    │   │   ├── autoencoder_task.py      # Supplementary Method: Appendix C in our paper 
    │   │   ├── sampling_task.py         # GNN Contrastive Learning
    │   │   └── utils.py
    │   ├── gnn_tasks_miniBatch          # With Mini-Batching/Neighborhood sampling
    │   │   ├── autoencoder_task.py      # Supplementary Method: Appendix C in our paper
    │   │   ├── dataloader.py            
    │   │   ├── sampling_task.py         # GNN Contrastive Learning
    │   │   └── utils.py
    │   ├── kmer_node2vec.py             
    │   ├── kmer_onehot.py
    │   ├── kmer_ssgnn.py                # Our Workflow Without Mini-Batching/Neighborhood sampling
    │   ├── kmer_ssgnn_miniBatch.py      # Our Workflow With Mini-Batching/Neighborhood sampling
    │   ├── kmer_word2vec.py
    │   └── representations_factory.py
    ├── train
    │   ├── main_editDistance.py         # Main Workflow
    │   ├── main_geneFinder.py           # Supplementary Task: *Outside the research scope of the paper*
    │   ├── param_search_optuna.py       # Can be used with yaml file for grid search
    │   └── parsers.py
    └── utils.py
```

## References

1. Corso, G., Ying, Z., Pándy, M., Veličković, P., Leskovec, J., & Liò, P. (2021). 
   [*Neural distance embeddings for biological sequences*](https://github.com/gcorso/NeuroSEED). 
   Advances in Neural Information Processing Systems, 34, 18539-18551.

2. Silva, R., Padovani, K., Góes, F., & Alves, R. (2021). 
   [*geneRFinder: gene finding in distinct metagenomic data complexities*](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-03997-w). 
   BMC bioinformatics, 22(1), 1-17. BioMed Central.

3. Johnson, J., Douze, M. and Jégou, H., 2019.
    [*Billion-scale similarity search with gpus.*](https://ieeexplore.ieee.org/abstract/document/8733051?casa_token=u2u5Pe1S8ksAAAAA:isPuVesexwv0uvwu6gVLH61jOvrdHneE3PFjNG9zIyq0FWwtKacl96RV5ronX1-6pkeFmPW-)
    IEEE Transactions on Big Data, 7(3), pp.535-547.

