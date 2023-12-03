import torch.nn as nn
import torch
import numpy as np
from itertools import product

from src.utils import seq_to_kmers

ALPHABET = ["A", "C", "G", "T"]


def kmer_onehot_fit(dfs_list: list, k: int, args: dict):
    """
    Generates a one-hot encoding for k-mers and constructs an embedding layer.

    Parameters:
    - dfs_list (list): List of dataframes containing sequence data.
    - k (int): Length of k-mer for encoding.
    - args (dict): Dictionary of arguments. Important keys include:
        - "random_representation": Boolean, if True generates random representations.
        - "representation_k": String, comma-separated k values for representation.
        - "representation_size": int, size of representation vector.

    Returns:
    - index_dict (dict): Dictionary mapping k-mers to indices.
    - layer (nn.Embedding): Embedding layer initialized with one-hot (or random) weights.
    """

    all_kmers = ["".join(i) for i in product(ALPHABET, repeat=k)]
    all_kmers.append("N" * k)
    random_representation = args["random_representation"]

    if not random_representation:
        assert len(args["representation_k"].split(",")) == 1
        args["representation_size"] = 4 ** int(args["representation_k"].split(",")[0])

        weights = np.zeros((len(all_kmers), len(all_kmers) - 1))
        weights.flat[0 :: len(all_kmers)] = 1

    else:
        weights = np.random.rand(len(all_kmers), args["representation_size"])
        weights[-1, :] = 0
    index_dict = {k: v for v, k in enumerate(all_kmers)}

    layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights))

    return index_dict, layer


def kmer_onehot_transform(dfs_list: list, index_dict: dict, k: int, args: dict):
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

    # include UNK for final representations
    for df in dfs_list:
        df[("kmers_index_" + str(k))] = df["sequence"].apply(word2index)

    return dfs_list
