import torch
import torch.nn as nn
from node2vec import Node2Vec
import numpy as np

from src.utils import (
    build_deBruijn_graph,
    weighted_directed_edges,
    seq_to_kmers,
)


def kmer_node2vec_fit(dfs_list: list, k: int, args: dict):
    """
    Trains a Node2Vec model on k-mers from sequences and constructs a PyTorch embedding layer.

    Parameters:
    - dfs_list (list): List of dataframes containing sequence data.
    - k (int): Length of k-mer for encoding.
    - args (dict): Dictionary of arguments. Important keys include:
        - "representation_stride": Stride length for k-mer encoding during training.
        - "representation_size": Size of the resulting embedding.
        - "window_size": Maximum distance between the current and predicted node in a graph walk.
        - "walk_len": Length of each random walk.
        - "num_walks": Number of random walks per node.
        - "p": Random Walk return parameter.
        - "q":  Random Walk in-out parameter.

    Returns:
    - index_dict (dict): Dictionary mapping k-mers to indices.
    - layer (nn.Embedding): Embedding layer initialized with weights from the trained Node2Vec model.
    """
    stride = args["representation_stride"]
    embedding_size = args["representation_size"]
    window_size = args["window_size"]
    walk_len = args["walk_len"]
    num_walks = args["num_walks"]
    p = args["p"]
    q = args["q"]

    node2vec_dataset = []
    for df in dfs_list:
        node2vec_dataset.extend(df["sequence"].tolist())
    _, pair_frequency = weighted_directed_edges(
        node2vec_dataset, k=k, stride=stride[0], inlcudeUNK=False
    )
    graph = build_deBruijn_graph(pair_frequency)

    node2vec_model = Node2Vec(
        graph,
        dimensions=embedding_size,
        walk_length=walk_len,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=1,
    )
    node2vec_model = node2vec_model.fit(window=window_size, min_count=1)

    keyed_vectors = node2vec_model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

    unk_index = max(keyed_vectors.key_to_index.values()) + 1
    weights = np.vstack([weights, [0] * embedding_size])

    index_dict = keyed_vectors.key_to_index
    index_dict["N" * k] = unk_index

    layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights))
    return index_dict, layer


def kmer_node2vec_transform(dfs_list: list, index_dict: dict, k: int, args: dict):
    """
    Transforms sequences in the provided dataframes to indices of k-mers.

    Parameters:
    - dfs_list (list): List of dataframes containing sequence data.
    - index_dict (dict): Dictionary mapping k-mers to indices.
    - k (int): Length of k-mer for encoding.
    - args (dict): Dictionary of arguments. An important key includes:
        - "inference_stride": Stride length for k-mer encoding during inference.

    Returns:
    - dfs_list (list): List of dataframes with an added column 'kmers_index_k' containing encoded k-mer indices.
    """
    stride = args["inference_stride"]

    def word2index(seq):
        return np.array(
            [index_dict[kmer] for kmer in seq_to_kmers(seq, k, stride, inlcudeUNK=True)]
        )

    for df in dfs_list:
        df[("kmers_index_" + str(k))] = df["sequence"].apply(word2index)

    return dfs_list
