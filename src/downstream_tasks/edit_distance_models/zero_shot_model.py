import torch
import torch.nn as nn
import pytorch_lightning as pl


def zero_shot_model(
    embedding_layers_list,
    combination_function="mean",
):
    """
    Initializes and returns a ZeroShotEmbeddingModel with the provided embedding layers and combination function.

    Parameters:
    -----------
    embedding_layers_list : List[torch.nn.Module]
        List of pre-trained embedding layers.

    combination_function : str, optional (default="mean")
        Function to combine the word embeddings into a sentence representation.
        Options are "mean", "sum", "max", or "concat".

    Returns:
    --------
    ZeroShotEmbeddingModel
        The initialized ZeroShotEmbeddingModel.

    """
    model = ZeroShotEmbeddingModel(embedding_layers_list, combination_function)

    return model


class ZeroShotEmbeddingModel(pl.LightningModule):
    """
    Zero Shot Embedding Model Class.

    This model takes multiple embeddings of a text and combines them
    into a single representation using a specified combination function.

    Attributes:
    -----------
    embedding_layers_list : torch.nn.ModuleList
        List of pre-trained embedding layers.

    combination_function : str
        Function to combine the word embeddings into a sentence representation.
        Options are "mean", "sum", "max", or "concat".

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor:
        Passes the input tensor through each embedding layer and then combines
        the embeddings using the specified combination function.

    """

    def __init__(self, embedding_layers_list, combination_function="mean"):
        super(ZeroShotEmbeddingModel, self).__init__()
        self.embedding_layers_list = nn.ModuleList(embedding_layers_list)
        self.combination_function = combination_function

    def forward(self, x):
        sentence_representations = []
        for i, embedding_layer in enumerate(self.embedding_layers_list):
            embedding_output = embedding_layer(x[i].squeeze())
            # Apply the combination function at the word level
            if self.combination_function == "mean":
                sentence_representation = torch.mean(embedding_output, dim=1)
            elif self.combination_function == "sum":
                sentence_representation = torch.sum(embedding_output, dim=1)
            elif self.combination_function == "max":
                sentence_representation = torch.max(embedding_output, dim=1)[0]
            elif self.combination_function == "concat":
                sentence_representation = embedding_output.view(
                    embedding_output.size(0), -1
                )
            else:
                raise ValueError("Invalid combination function.")
            sentence_representations.append(sentence_representation)

        # Concatenate the sentence representations
        combined_output = torch.cat(sentence_representations, dim=1)
        return combined_output
