import pickle
import pandas as pd
import numpy as np

MAP = np.array(["A", "C", "G", "T", "N"])


def edit_distance_benchmark(
    ds_path: str,
):
    """
    Load the benchmark dataset for the EDIT DISTANCE APPROXIMATION task.

    This function reads the benchmark dataset from the specified path
    and processes it to extract sequences and their distances for training,
    validation, and testing purposes.

    Parameters:
        ds_path (str): The path to the benchmark dataset.

    Returns:
        tuple: A tuple containing dataframes for training, validation, and testing sequences
               along with their corresponding distances.
    """
    with open(ds_path, "rb") as f:
        sequences_int, distances = pickle.load(f)

    for key in sequences_int.keys():
        if key == "train":
            sequences = np.array(["".join(MAP[seq]) for seq in sequences_int[key]])
            train_df = pd.DataFrame(sequences, columns=["sequence"])
            train_dist = distances[key]

        elif key == "val":
            sequences = np.array(["".join(MAP[seq]) for seq in sequences_int[key]])
            val_df = pd.DataFrame(sequences, columns=["sequence"])
            val_dist = distances[key]

        else:
            sequences = np.array(["".join(MAP[seq]) for seq in sequences_int[key]])
            test_df = pd.DataFrame(sequences, columns=["sequence"])
            test_dist = distances[key]

    return train_df, train_dist, val_df, val_dist, test_df, test_dist


def retrieval_benchmark(
    ds_path: str,
):
    """
    Load the benchmark dataset for the CLOSEST STRING RETRIEVAL task.

    This function reads the benchmark dataset from the specified path
    and processes it to extract reference sequences, query sequences,
    and corresponding labels.

    Parameters:
        ds_path (str): The path to the benchmark dataset.

    Returns:
        tuple: A tuple containing dataframes for reference sequences and query sequences with their labels.
    """
    with open(ds_path, "rb") as f:
        sequences_references, sequences_queries, labels = pickle.load(f)

    sequences_references = np.array(["".join(MAP[seq]) for seq in sequences_references])
    references_df = pd.DataFrame(sequences_references, columns=["sequence"])

    sequences_queries = np.array(["".join(MAP[seq]) for seq in sequences_queries])
    queries_df = pd.DataFrame(sequences_queries, columns=["sequence"])
    queries_df["labels"] = labels

    return references_df, queries_df
