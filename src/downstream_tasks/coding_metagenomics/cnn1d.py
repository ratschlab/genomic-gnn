import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics import Accuracy, AUROC


class CNN1D(pl.LightningModule):
    """
    1D Convolutional Neural Network implemented with PyTorch Lightning.

    This class provides a 1D CNN model with possibly multiple parallel layers for the input data,
    the multiple parallel layers potentially correspnd to multiple k values with conventional embedding methods,
    followed by sequential layers after the parallel layers are concatenated.
    The network is designed for binary classification tasks.
    """

    def __init__(
        self,
        embedding_layers_list: list,
        input_embedding_sizes_list: list,
        args: dict,
    ):
        """
        Initialize the CNN1D module.

        Parameters:
        ----------
        - embedding_layers_list (list): List of embedding layers.
        - input_embedding_sizes_list (list): List of sizes for each embedding layer input.
        - args (dict): Dictionary of hyperparameters and configurations for the network.
        """
        super().__init__()
        self.learning_rate = args["learning_rate"]
        batch_norm = args["batch_norm"]
        pooling = args["pooling"]
        channels = args["channels"]
        network_layers_after_cat = args["network_layers_after_cat"]
        activation = args["activation"]
        dropout = args["dropout"]
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
        if activation:
            self.post_cat_layers.append(self.activation_fn)
        if dropout:
            self.post_cat_layers.append(nn.Dropout(p=dropout))
        for i in range(len(network_layers_after_cat) - 1):
            layer = nn.Linear(
                network_layers_after_cat[i], network_layers_after_cat[i + 1]
            )
            self.post_cat_layers.append(layer)
            if activation:
                self.post_cat_layers.append(self.activation_fn)
            if dropout:
                self.post_cat_layers.append(nn.Dropout(p=dropout))

        layer = nn.Linear(network_layers_after_cat[-1], 1)
        self.post_cat_layers.append(layer)
        self.post_cat_layers.append(nn.Sigmoid())

        self.post_cat_layers = nn.Sequential(*self.post_cat_layers)

        self.loss = nn.BCELoss()
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def forward(self, x):
        """
        Forward propagation of input x through the network.

        Parameters:
        ----------
        - x: Input data.

        Returns:
        --------
        - Tensor: Model's prediction for the input data.
        """
        for i, embedding_layer in enumerate(self.embedding_layers_list):
            with torch.set_grad_enabled(self.trainable_embedding):
                x[i] = x[i].squeeze()
                x[i] = embedding_layer(x[i])
                x[i] = x[i].permute(0, 2, 1)
            x[i] = self.pre_cat_layers[i](x[i])
        x = torch.cat(x, 1)
        # x = x.view(x.size(0), -1)
        return self.post_cat_layers(x)

    def training_step(self, batch, batch_idx):
        """
        Logic for one training step.

        Parameters:
        ----------
        - batch: Current batch of data.
        - batch_idx: Index of the current batch.

        Returns:
        --------
        - Tensor: Training loss for the current batch.
        """
        x, y = batch
        y_pred = self.forward(x).squeeze()
        loss = self.loss(y_pred, y.float())
        self.train_accuracy(y_pred.round(), y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Logic for one validation step.

        Parameters:
        ----------
        - batch: Current batch of data.
        - batch_idx: Index of the current batch.
        """
        x, y = batch
        y_pred = self.forward(x).squeeze()
        loss = self.loss(y_pred, y.float())
        self.val_accuracy(y_pred.round(), y.int())
        self.val_auroc(y_pred, y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log(
            "val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx, dataloader_idx):
        """
        Logic for one testing step.

        Parameters:
        ----------
        - batch: Current batch of data.
        - batch_idx: Index of the current batch.
        - dataloader_idx: Index of the current dataloader.
        """
        x, y = batch
        y_pred = self.forward(x).squeeze()
        loss = self.loss(y_pred, y.float())
        self.test_accuracy(y_pred.round(), y.int())
        self.test_auroc(y_pred, y.int())

        self.log("test_loss", loss, prog_bar=True)
        self.log(
            "test_acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        """
        Configure optimizers for the model.

        Returns:
        --------
        - Optimizer object.
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
