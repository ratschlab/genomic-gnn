from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
import torch
from itertools import product
import os
from typing import Tuple, Dict


ALPHABET = ["A", "C", "G", "T"]


def seq_to_kmers(seq: str, k: int = 3, stride: int = 1, inlcudeUNK: bool = True):
    """
    Converts a sequence into a list of kmers.

    Parameters:
    - seq (str): The sequence to be converted into kmers.
    - k (int, optional): Length of kmers. Defaults to 3.
    - stride (int, optional): The step size between kmers in the sequence. Defaults to 1.
    - includeUNK (bool, optional): If True, keeps kmers containing the 'N' nucleotide, and replaces them with "N" repeated k times. If False, discards kmers containing 'N'. Defaults to True.

    Returns:
    - list of str: List of kmers derived from the input sequence.
    """
    kmers = [seq[i : i + k] for i in range(0, len(seq), stride) if i + k <= len(seq)]
    if inlcudeUNK:
        return ["N" * k if "N" in kmer else kmer for kmer in kmers]
    return [kmer for kmer in kmers if "N" not in kmer]


def add_kmers(df, k: int = 3, stride: int = 1, inlcudeUNK: bool = True):
    """
    Adds a new column to a DataFrame containing kmers derived from the "sequence" column.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing a 'sequence' column with sequences from which kmers will be derived.
    - k (int, optional): Length of kmers. Defaults to 3.
    - stride (int, optional): The step size between kmers in the sequence. Defaults to 1.
    - includeUNK (bool, optional): Whether to include padding Nucleotide N. Defaults to True.

    Returns:
    - None: The function modifies the input DataFrame in-place.
    """
    df[("kmers_" + str(k))] = df["sequence"].apply(
        seq_to_kmers, args=(k, stride, inlcudeUNK)
    )


def weighted_directed_edges(
    sequences: list,
    k: int = 3,
    stride: int = 1,
    inlcudeUNK: bool = True,
    disable_tqdm: bool = True,
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
    """
    Compute the frequencies of individual kmers and their transitions in given sequences.

    Parameters:
    - sequences (list of str): A list of sequences from which kmers and their transitions are to be extracted.
    - k (int, optional): The length of kmers. Defaults to 3.
    - stride (int, optional): The step size between kmers in the sequence. Defaults to 1.
    - inlcudeUNK (bool, optional): Whether to include unknown kmers. Defaults to True.

    Returns:
    - item_frequency (dict): A dictionary where keys are kmers and values are their respective frequencies.
    - pair_frequency (dict): A dictionary where keys are kmer transitions (tuple of two kmers) and values are their transition frequencies.
    """
    pair_frequency = defaultdict(int)
    item_frequency = defaultdict(int)
    for seq in tqdm(
        sequences,
        desc="Compute kmer transition frequencies",
        disable=disable_tqdm,
    ):
        kmers = seq_to_kmers(seq, k=k, stride=stride, inlcudeUNK=inlcudeUNK)
        for i, kmer in enumerate(kmers[:-1]):
            item_frequency[kmer] += 1
            pair_frequency[(kmer, kmers[i + 1])] += 1

    return item_frequency, pair_frequency


def build_deBruijn_graph(
    pair_frequency: dict,
    normalise: bool = False,
    remove_N: bool = False,
    edge_weight_threshold: float = None,
    keep_top_k: float = None,
    create_all_kmers: bool = True,
    disable_tqdm: bool = True,
) -> nx.DiGraph:
    """Builds a De Bruijn graph from a dictionary of kmer transition frequencies.

    Args:
        pair_frequency (dict): Kmer transition frequencies from weighted_directed_edges function.
        normalise (bool, optional): If normalise weights to ones between 0 to 1. Defaults to False.
        remove_N (bool, optional): If we remove nodes with padding nucleotide. Defaults to False.
        edge_weight_threshold (float, optional): If removes edges with weights below treshold. Defaults to None.
        keep_top_k (float, optional): Percentage (0.0 to 1.0) of the top weighted edges to retain. Defaults to None.
        create_all_kmers (bool, optional):If True, creates nodes for all possible kmers. Defaults to True.

    Returns:
        nx.DiGraph: A directed de Bruijn graph.
    """
    if normalise:
        # Group by the source node and sum the weights
        source_weights = defaultdict(int)
        for pair, weight in pair_frequency.items():
            source_weights[pair[0]] += weight

        # Normalize weights
        for pair, weight in pair_frequency.items():
            pair_frequency[pair] /= source_weights[pair[0]]

    # Create deBruji directed graph.
    deBruijn_graph = nx.DiGraph()

    k = len(list(pair_frequency.keys())[0][0])

    if create_all_kmers:
        all_kmers = ["".join(i) for i in product(ALPHABET, repeat=k)]
        all_kmers.append("N" * k)
        for kmer in all_kmers:
            deBruijn_graph.add_node(kmer)

    ###
    deBruijn_graph.add_node("N" * k)
    ###

    for pair in tqdm(
        pair_frequency, desc="Creating DeBruij Graph", disable=disable_tqdm
    ):
        x, y = pair
        weight = pair_frequency[pair]
        if edge_weight_threshold is not None:
            if weight > edge_weight_threshold:
                deBruijn_graph.add_edge(x, y, weight=weight)
        else:
            deBruijn_graph.add_edge(x, y, weight=weight)

    print(
        "Propotion of nodes in dataset to all possible nodes: ",
        deBruijn_graph.number_of_nodes() / 4**k,
    )

    # Only keeps top k percentage of the edges with highest kmer transition frequency
    if keep_top_k is not None:
        weights = [deBruijn_graph[u][v]["weight"] for u, v in deBruijn_graph.edges()]
        threshold = np.quantile(weights, 1 - keep_top_k)
        edges_to_remove = []
        for u, v in deBruijn_graph.edges():
            if deBruijn_graph[u][v]["weight"] < threshold:
                edges_to_remove.append((u, v))

        for edges_to_remove in edges_to_remove:
            deBruijn_graph.remove_edge(*edges_to_remove)

    if remove_N:
        deBruijn_graph.remove_node("N" * k)

    return deBruijn_graph


def set_all_seeds(seed: int = 0):
    """Set all seeds for reproducibility

    Args:
        seed (int, optional): Seed number. Defaults to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
