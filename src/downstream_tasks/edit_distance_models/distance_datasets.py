from torch.utils.data import Dataset, BatchSampler, Sampler
import pandas as pd
import numpy as np
import torch
import random


class SampledDistanceDataset(Dataset):
    """
    A PyTorch Dataset representing sequences for which only a random subset
    of pairwise distances are available. These distances are randomly sampled
    at every epoch.

    Attributes:
    -----------
    sequences : list[torch.Tensor]
        The list of sequences represented as tensors.
    distances : np.array
        The pairwise distances between the sequences.
    multiplicity : int
        Half of the number of distances sampled per sequence at every epoch.
    N_sequences : int
        The number of sequences in the dataset.
    max_seq_lengths : list[int]
        The maximum lengths of sequences.
    """

    # only some pairwise distances available are randomly sampled at every epoch
    def __init__(self, df: pd.DataFrame, dist: np.array, ks_str: str, multiplicity=10):
        # multiplicity indicates (1/2) the number of distances sampled per sequence at every epoch
        self.sequences = [
            torch.from_numpy(np.stack(df["kmers_index_" + k].to_numpy(), axis=0))
            for k in ks_str.split(",")
        ]

        self.distances = dist
        self.multiplicity = multiplicity
        self.N_sequences = self.sequences[0].shape[0]
        self.max_seq_lengths = [seq.shape[-1] for seq in self.sequences]

        # Normalise labels by length
        self.distances = self.distances / self.sequences[0].shape[-1]

    def __len__(self):
        return self.N_sequences * self.multiplicity

    def __getitem__(self, index):
        index = index // self.multiplicity
        sequences = None
        d = torch.Tensor([0.0])
        while torch.all(
            d == 0
        ):  # avoid equal sequences that might give numerical problems
            idx2 = random.randint(0, self.N_sequences - 1)
            sequences = (
                [
                    self.sequences[i][index % self.N_sequences].unsqueeze(0)
                    for i in range(len(self.sequences))
                ],
                [
                    self.sequences[i][idx2].unsqueeze(0)
                    for i in range(len(self.sequences))
                ],
            )
            d = self.distances[index % self.N_sequences, idx2]
        return sequences, d


class DistanceDataset(Dataset):
    """
    A PyTorch Dataset representing sequences where every pairwise distance
    is loaded at every epoch.

    Attributes:
    -----------
    sequences : list[torch.Tensor]
        The list of sequences represented as tensors.
    distances : np.array
        The pairwise distances between the sequences.
    N_sequences : int
        The number of sequences in the dataset.
    max_seq_lengths : list[int]
        The maximum lengths of sequences.
    """

    def __init__(self, df: pd.DataFrame, dist: np.array, ks_str: str):
        self.sequences = [
            torch.from_numpy(np.stack(df["kmers_index_" + k].to_numpy(), axis=0))
            for k in ks_str.split(",")
        ]

        self.distances = dist
        self.N_sequences = self.sequences[0].shape[0]

        # Normalise labels
        self.max_seq_lengths = [seq.shape[-1] for seq in self.sequences]
        self.distances = self.distances / self.sequences[0].shape[-1]

    def __len__(self):
        return self.N_sequences * (self.N_sequences - 1)

    def __getitem__(self, index):
        # calculate the right indices avoiding pairs (i, i)
        idx1 = index // (self.N_sequences - 1)
        idx2 = index % (self.N_sequences - 1)
        if idx2 >= idx1:
            idx2 += 1

        sequences = (
            [self.sequences[i][idx1].unsqueeze(0) for i in range(len(self.sequences))],
            [self.sequences[i][idx2].unsqueeze(0) for i in range(len(self.sequences))],
        )
        d = self.distances[idx1, idx2]
        return sequences, d


class ReferenceDataset(Dataset):
    """
    A PyTorch Dataset representing reference sequences.

    Attributes:
    -----------
    sequences : list[torch.Tensor]
        The list of sequences represented as tensors.
    """

    def __init__(self, df: pd.DataFrame, ks_str: str):
        self.sequences = [
            torch.from_numpy(np.stack(df["kmers_index_" + k].to_numpy(), axis=0))
            for k in ks_str.split(",")
        ]

    def __len__(self):
        return self.sequences[0].shape[0]

    def __getitem__(self, index):
        return [seq[index] for seq in self.sequences]


class QueryDataset(Dataset):
    """
    A PyTorch Dataset representing query sequences along with their labels.

    Attributes:
    -----------
    sequences : list[torch.Tensor]
        The list of sequences represented as tensors.
    labels : torch.Tensor
        The labels corresponding to the query sequences.
    """

    def __init__(self, df: pd.DataFrame, ks_str: str):
        self.sequences = [
            torch.from_numpy(np.stack(df["kmers_index_" + k].to_numpy(), axis=0))
            for k in ks_str.split(",")
        ]
        self.labels = torch.from_numpy(np.stack(df["labels"].to_numpy(), axis=0))

    def __len__(self):
        return self.sequences[0].shape[0]

    def __getitem__(self, index):
        return [seq[index] for seq in self.sequences], self.labels[index]
