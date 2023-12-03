from gensim.models import Word2Vec
import torch.nn as nn
import torch
import numpy as np

from src.utils import add_kmers, seq_to_kmers


def kmer_word2vec_fit(dfs_list: list, k: int, args: dict):
    """
    Trains a Word2Vec model on k-mers and constructs an embedding layer.

    Parameters:
    - dfs_list (list): List of dataframes containing sequence data.
    - k (int): Length of k-mer for encoding.
    - args (dict): Dictionary of arguments. Important keys include:
        - "representation_stride": Stride length for k-mer encoding during training.
        - "representation_size": int, size of the resulting embedding.
        - "window_size": int, maximum distance between the current and predicted word in a sentence.
        - "seed": int, seed for reproducibility.

    Returns:
    - index_dict (dict): Dictionary mapping k-mers to indices.
    - layer (nn.Embedding): Embedding layer initialized with weights from the trained Word2Vec model.
    """
    stride = args["representation_stride"]
    embedding_size = args["representation_size"]
    window_size = args["window_size"]

    # exclude UNK for embedding training
    for df in dfs_list:
        add_kmers(df, k=k, stride=stride[0], inlcudeUNK=False)

    word2vec_dataset = []
    for df in dfs_list:
        word2vec_dataset.extend(df["kmers_" + str(k)].tolist())
    del df["kmers_" + str(k)]

    word2vec_model = Word2Vec(
        sentences=word2vec_dataset,
        vector_size=embedding_size,
        window=window_size,
        workers=1,
        min_count=1,
        seed=int(args["seed"]),
    )

    keyed_vectors = word2vec_model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

    unk_index = max(keyed_vectors.key_to_index.values()) + 1
    weights = np.vstack([weights, [0] * embedding_size])

    index_dict = keyed_vectors.key_to_index
    index_dict["N" * k] = unk_index

    layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights))
    return index_dict, layer


def kmer_word2vec_transform(dfs_list: list, index_dict: dict, k: int, args: dict):
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
