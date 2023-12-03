import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import numpy as np
import wandb

from src.downstream_tasks.edit_distance_models.distance_datasets import (
    QueryDataset,
    ReferenceDataset,
)
from src.downstream_tasks.edit_distance_models.distances import DISTANCES_FACTORY


def retrieval_test(
    model,
    references_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    args: dict,
    distance: str = "hyperbolic",
    batch_size: int = 128,
):
    """
    Test the retrieval capability of a model based on embeddings.
    CLOSEST STRING RETREIVAL task

    Given a model and datasets of reference and queries, this function computes the retrieval accuracy
    by determining the distances between the embedded representations of reference and query samples.
    It then evaluates the accuracy of the retrieval process based on ranked distances (e.g., Top1, Top5, Top10).

    Parameters:
    -----------
    model: torch.nn.Module
        The PyTorch model to be tested.

    references_df: pd.DataFrame
        The dataframe containing the reference sequences.

    queries_df: pd.DataFrame
        The dataframe containing the query sequences.

    args: dict
        Dictionary containing various arguments/settings related to the model and test settings.

    distance: str, optional (default = "hyperbolic")
        The distance metric to use. Must be a key in the DISTANCES_FACTORY dictionary.

    batch_size: int, optional (default = 128)
        The size of batches to use when processing the data.

    Returns:
    --------
    None
    """
    ks_str = args["representation_k"]
    distance += "_matrix"

    references_dataloader = DataLoader(
        ReferenceDataset(references_df, ks_str=ks_str),
        batch_size=batch_size,
        shuffle=False,
    )
    queries_dataloader = DataLoader(
        QueryDataset(queries_df, ks_str=ks_str), batch_size=batch_size, shuffle=False
    )

    distance_fun = DISTANCES_FACTORY[distance]

    trainer = pl.Trainer(accelerator=args["accelerator"])
    references_embedded = trainer.predict(model, references_dataloader)

    n_batches = 0
    acc = np.array([0.0] * 10)
    for queries, labels in queries_dataloader:
        n_batches += 1

        queries_embedded = model.forward(queries)

        distance_matrix = distance_fun(torch.cat(references_embedded), queries_embedded)
        label_distances = distance_matrix[
            labels.long(), torch.arange(0, distance_matrix.shape[1])
        ]

        rank = torch.sum(
            torch.le(distance_matrix, label_distances.unsqueeze(0)).float(), dim=0
        )
        acc += np.array([torch.mean((rank <= i + 1).float()) for i in range(10)])

    avg_acc = acc / n_batches

    print(
        "Top1: {:.3f}  Top5: {:.3f}  Top10: {:.3f}".format(
            avg_acc[0], avg_acc[4], avg_acc[9]
        )
    )
    if hasattr(model, "combination_function"):
        wandb.log(
            {
                "Top1_" + model.combination_function: avg_acc[0],
                "Top5_" + model.combination_function: avg_acc[4],
                "Top10_" + model.combination_function: avg_acc[9],
            }
        )
    else:
        wandb.log(
            {
                "Top1": avg_acc[0],
                "Top5": avg_acc[4],
                "Top10": avg_acc[9],
            }
        )
