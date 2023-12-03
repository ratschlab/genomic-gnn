import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
import wandb

from src.downstream_tasks.edit_distance_models.distances import DISTANCES_FACTORY


class CNN1D(pl.LightningModule):
    """
    A PyTorch Lightning module representing a 1D Convolutional Neural Network
    designed for the task of computing distances between sequences.

    The model processes sequences by passing them through parallel convolutional
    layers and then through subsequent layers after concatenation. The distance
    between the resulting embeddings is computed using the specified distance metric.

    Attributes:
    -----------
    - num_k: int
        Number of k-length subsequences.
    - trainable_embedding: bool
        Specifies if the embeddings should be trainable.
    - activation_fn: nn.Module
        Activation function used in the network.
    - pre_cat_layers: nn.ModuleList
        List of parallel convolutional layers before concatenation.
    - post_cat_layers: nn.Module
        Linear layers after the parallel convolutional layers.
    - loss: nn.Module
        Loss function for the model, Mean Squared Error in this context.
    - distance_name: str
        Name of the distance metric being used.
    - distance: Callable
        Function to compute the distance metric.
    - metrics: list
        List to store evaluation metrics.
    - best_val_loss: float
        The best validation loss achieved during training.
    - val_losses: list
        List to store validation losses for each epoch.

    Methods:
    --------
    normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        Normalizes the provided embeddings based on the distance metric.

    __generic_step(batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        Computes the loss for a given batch of data.

    forward(x: List[torch.Tensor]) -> torch.Tensor:
        Forward pass through the network.

    training_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        Executes one training step.

    validation_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        Computes validation metrics for one batch.

    on_validation_epoch_end():
        Actions to perform at the end of a validation epoch.

    test_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        Computes test metrics for one batch.

    configure_optimizers() -> torch.optim.Optimizer:
        Specifies the optimizer to be used during training.
    """

    def __init__(
        self,
        embedding_layers_list: list,
        input_embedding_sizes_list: list,
        args: dict,
        distance: str = "hyperbolic",
    ):
        super().__init__()
        scaling = args["scaling"]
        self.learning_rate = args["learning_rate"]

        self.scaling = None
        if scaling:
            self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
            self.scaling = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.__training_step = self.__generic_step

        batch_norm = args["batch_norm"]
        pooling = args["pooling"]
        channels = args["channels"]
        network_layers_after_cat = args["network_layers_after_cat"]
        activation = args["activation"]
        kernel_size = args["kernel_size"]
        if not kernel_size:
            kernel_size = args["representation_size"]

        self.num_k = len(embedding_layers_list)
        self.trainable_embedding = args["trainable_embedding"]
        self.embedding_layers_list = nn.ModuleList(embedding_layers_list)
        if self.trainable_embedding:
            for embedding in self.embedding_layers_list:
                embedding.weight.requires_grad = True
        if activation:
            self.activation_fn = getattr(nn, activation)()

        # Define the parallel layers
        self.pre_cat_layers = nn.ModuleList()
        pre_cat_output_shapes = []
        for i in range(self.num_k):
            parallel_layer_sequence = nn.ModuleList()
            number_input_channels = args["representation_size"]
            input_shape = input_embedding_sizes_list[i] // args["representation_size"]
            for i, number_out_channels in enumerate(channels):
                conv_layer = nn.Conv1d(
                    number_input_channels, number_out_channels, kernel_size
                )
                input_shape = input_shape - kernel_size + 1
                parallel_layer_sequence.append(conv_layer)

                if batch_norm:
                    parallel_layer_sequence.append(nn.BatchNorm1d(number_out_channels))
                if activation:
                    parallel_layer_sequence.append(self.activation_fn)
                if pooling == "max":
                    parallel_layer_sequence.append(nn.MaxPool1d(2))
                    input_shape = (input_shape) // (2)
                elif pooling == "avg":
                    parallel_layer_sequence.append(nn.AvgPool1d(2))
                    input_shape = (input_shape) // (2)
                number_input_channels = number_out_channels

            pre_cat_output_shapes.append(input_shape * number_input_channels)

            parallel_layer_sequence.append(nn.Flatten())
            self.pre_cat_layers.append(nn.Sequential(*parallel_layer_sequence))

        # Define the sequential layers post concatenation
        input_size_after_cnn = sum(pre_cat_output_shapes)
        self.post_cat_layers = []
        self.post_cat_layers.append(
            nn.Linear(input_size_after_cnn, network_layers_after_cat[0])
        )
        for i in range(len(network_layers_after_cat) - 1):
            if activation:
                self.post_cat_layers.append(self.activation_fn)
            layer = nn.Linear(
                network_layers_after_cat[i], network_layers_after_cat[i + 1]
            )
            self.post_cat_layers.append(layer)

        self.post_cat_layers = nn.Sequential(*self.post_cat_layers)

        self.loss = nn.MSELoss()
        self.distance_name = distance
        self.distance = DISTANCES_FACTORY[distance]
        self.metrics = []
        self.best_val_loss = float("inf")
        self.val_losses = []

    def normalize_embeddings(self, embeddings):
        min_scale = 1e-7

        if self.distance_name == "hyperbolic":
            max_scale = 1 - 1e-3
        else:
            max_scale = 1e10

        return nn.functional.normalize(embeddings, p=2, dim=1) * self.radius.clamp_min(
            min_scale
        ).clamp_max(max_scale)

    def __generic_step(self, batch):
        x, y = batch
        seq1, seq2 = x
        seq1_embed = self.forward(seq1)
        seq2_embed = self.forward(seq2)
        if self.scaling:
            seq1_embed = self.normalize_embeddings(seq1_embed)
            seq2_embed = self.normalize_embeddings(seq2_embed)
        distance_hat = self.distance(seq1_embed, seq2_embed)
        if self.scaling:
            distance_hat = distance_hat * self.scaling
        loss = self.loss(distance_hat, y)
        return loss

    def forward(self, x):
        for i, embedding_layer in enumerate(self.embedding_layers_list):
            with torch.set_grad_enabled(self.trainable_embedding):
                x[i] = x[i].squeeze()
                x[i] = embedding_layer(x[i])
                x[i] = x[i].permute(0, 2, 1)
            x[i] = self.pre_cat_layers[i](x[i])
        x = torch.cat(x, 1)
        x = x.view(x.size(0), -1)
        return self.post_cat_layers(x)

    def training_step(self, batch, batch_idx):
        loss = self.__training_step(batch)
        self.log("train_loss", torch.sqrt(loss) * 100, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.__generic_step(batch)
        self.log("val_loss", torch.sqrt(loss) * 100, prog_bar=True)
        self.val_losses.append(torch.sqrt(loss) * 100)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.val_losses = []  # reset the list for the next epoch
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            wandb.log({"best_val_loss": self.best_val_loss})

    def test_step(self, batch, batch_idx):
        loss = self.__generic_step(batch)
        self.log("test_loss", torch.sqrt(loss) * 100, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
