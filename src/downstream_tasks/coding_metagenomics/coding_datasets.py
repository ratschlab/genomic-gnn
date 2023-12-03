from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch


class CodingDataset(Dataset):
    """
    A PyTorch Dataset for the coding regions.

    Attributes:
    ----------
    - sequences: A list of sequences represented as torch tensors.
    - labels: Tensor containing the class labels.
    - max_seq_lengths: Maximum sequence lengths for each k in the k-mer representation.
    """

    def __init__(self, df: pd.DataFrame, ks_str: str):
        """
        Initialize the CodingDataset.

        Parameters:
        ----------
        - df (pd.DataFrame): Dataframe containing the sequences and class labels.
        - ks_str (str): Comma-separated string of k values for the k-mer representation.
        """
        self.sequences = [
            torch.from_numpy(np.stack(df["kmers_index_" + k].to_numpy(), axis=0))
            for k in ks_str.split(",")
        ]
        self.labels = torch.from_numpy(
            np.stack(df["class"].to_numpy(dtype=np.float32), axis=0)
        )
        self.max_seq_lengths = [seq.shape[-1] for seq in self.sequences]

    def __len__(self):
        return self.sequences[0].shape[0]

    def __getitem__(self, index):
        return [seq[index] for seq in self.sequences], self.labels[index]


def create_dataloaders(
    training_df,
    validation_df,
    test_1_df,
    test_2_low_df,
    test_2_medium_df,
    test_2_high_df,
    args: dict,
):
    """
    Create DataLoader instances for the provided dataframes.

    Parameters:
    ----------
    - training_df, validation_df, test_1_df, test_2_low_df, test_2_medium_df, test_2_high_df (pd.DataFrame):
        Dataframes containing sequences and class labels.
    - args (dict): Dictionary containing hyperparameters and configurations.

    Returns:
    --------
    - tuple: DataLoaders for training, validation, testing datasets and the input embedding size.
    """
    batch_size = args["batch_size"]
    ks_str = args["representation_k"]
    representation_size = args["representation_size"]

    train_dataset = CodingDataset(df=training_df, ks_str=ks_str)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = CodingDataset(validation_df, ks_str=ks_str)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    test_1_dataset = CodingDataset(test_1_df, ks_str=ks_str)
    test_1_dataloader = DataLoader(test_1_dataset, batch_size=batch_size)

    test_2_low_dataset = CodingDataset(test_2_low_df, ks_str=ks_str)
    test_2_low_dataloader = DataLoader(test_2_low_dataset, batch_size=batch_size)

    test_2_medium_dataset = CodingDataset(test_2_medium_df, ks_str=ks_str)
    test_2_medium_dataloader = DataLoader(test_2_medium_dataset, batch_size=batch_size)

    test_2_high_dataset = CodingDataset(test_2_high_df, ks_str=ks_str)
    test_2_high_dataloader = DataLoader(test_2_high_dataset, batch_size=batch_size)

    test_oveall_df = pd.concat(
        [test_1_df, test_2_low_df, test_2_medium_df, test_2_high_df], ignore_index=True
    )
    test_oveall_dataset = CodingDataset(test_oveall_df, ks_str=ks_str)
    test_oveall_dataloader = DataLoader(test_oveall_dataset, batch_size=batch_size)

    input_embedding_size = [
        representation_size * l for l in train_dataset.max_seq_lengths
    ]

    return (
        train_dataloader,
        val_dataloader,
        test_1_dataloader,
        test_2_low_dataloader,
        test_2_medium_dataloader,
        test_2_high_dataloader,
        test_oveall_dataloader,
        input_embedding_size,
    )
