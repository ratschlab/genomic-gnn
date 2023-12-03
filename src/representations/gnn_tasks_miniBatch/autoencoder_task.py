import torch
from torch import nn
from torch_geometric.utils import negative_sampling
from pytorch_lightning import LightningModule
from torch.optim import Adam

from src.representations.gnn_common.gnn_models import (
    InnerProductEdgeDecoder,
    MLPNodeDecoder,
)
from src.representations.gnn_common.gnn_utils import compute_dirichlet_energy


class AutoEncoderGeneral(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        edge_weight_loss_weight: float = 0,
        kmer_frequency_loss_weight: float = 0,
        edit_distance_loss_weight: float = 0,
        negative_sampling_loss_weight: float = 0,
        proportion_negative_to_positive_samples: float = 1,
        node_p: int = 0,  # proportion of nodes to mask
        edge_p: int = 0,  # proportion of edges to mask
        encoder_output_channels: int = None,
        num_labels: list = None,
        lr: float = 0.001,  # learning rate
        num_of_additional_strides: int = 0,
    ):
        """
        Initializes the AutoEncoderGeneral model. Autoencoder model based for metagenomic graphs.
        The model uses a given encoder to obtain node embeddings from input features and then
        decodes them to recreate graph structures including edge weights, k-mer frequencies, or edit distances.
        Inlcudes Neighbourhood Sampling.

        Parameters:
            encoder (nn.Module): The graph encoder module.
            edge_weight_loss_weight (float, optional): Weight for the edge weight loss. Default is 0.
            kmer_frequency_loss_weight (float, optional): Weight for the kmer frequency loss. Default is 0.
            edit_distance_loss_weight (float, optional): Weight for the edit distance loss. Default is 0.
            negative_sampling_loss_weight (float, optional): Weight for negative sampling loss. Default is 0.
            proportion_negative_to_positive_samples (float, optional): Proportion of negative samples to positive samples. Default is 1.
            node_p (int, optional): Proportion of nodes to mask. Default is 0.
            edge_p (int, optional): Proportion of edges to mask. Default is 0.
            encoder_output_channels (int, optional): Output channels from the encoder. Default is None.
            num_labels (list, optional): List of label counts for the node decoders. Default is None.
            lr (float, optional): Learning rate for optimization. Default is 0.001.
            num_of_additional_strides (int, optional): Number of additional strides. Default is 0.
        """
        super(AutoEncoderGeneral, self).__init__()

        assert (
            edge_weight_loss_weight != 0
            or kmer_frequency_loss_weight != 0
            or edit_distance_loss_weight != 0
        ), "At least one positive loss must be non-zero"

        assert not (
            edge_weight_loss_weight == 1 and edit_distance_loss_weight == 1
        ), "Cannot have both edge weight and edit distance loss for inner product decoder"

        # general parameters
        self.encoder = encoder
        print("!!!encoder!!!", self.encoder)
        self.node_p = node_p
        self.edge_p = edge_p
        self.lr = lr
        self.negative_sampling_loss_weight = negative_sampling_loss_weight
        self.proportion_negative_to_positive_samples = (
            proportion_negative_to_positive_samples
        )
        # edge weight
        self.edge_weight_loss_weight = edge_weight_loss_weight
        self.num_of_additional_strides = num_of_additional_strides
        if edge_weight_loss_weight != 0:
            self.edge_decoder = InnerProductEdgeDecoder()
            self.edge_loss_func = nn.L1Loss()

        # kmer frequency
        self.kmer_frequency_loss_weight = kmer_frequency_loss_weight
        if kmer_frequency_loss_weight != 0:
            self.node_decoders = nn.ModuleList()
            for i, num_label in enumerate(num_labels):
                self.node_decoders.append(
                    MLPNodeDecoder([encoder_output_channels, num_label])
                )
            self.node_loss_func = nn.MSELoss()

        # edit distance
        self.edit_distance_loss_weight = edit_distance_loss_weight
        if edit_distance_loss_weight != 0:
            self.edit_distance_decoder = InnerProductEdgeDecoder()
            self.edit_distance_loss_func = nn.L1Loss()

    def forward(self, x, edge_index, edge_weight_masked=None):
        """
        Passes the input data through the encoder and decoders.

        Parameters:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge indices.
            edge_weight_masked (torch.Tensor, optional): Masked edge weights. Default is None.

        Returns:
            tuple: Contains the decoded edge weights, node features, and edit distances.
        """
        edges_config = self.edges_config.copy()
        if self.edge_weight_loss_weight != 0:
            edges_config["edge_index_DB"] = edge_index
            edges_config["edge_weight_DB"] = edge_weight_masked
        elif self.edit_distance_loss_weight != 0:
            edges_config["edge_index_ED"] = edge_index
            edges_config["edge_weight_ED"] = edge_weight_masked

        x = self.encoder(x, edges_config)
        edge_y = None
        node_y = []
        edit_distance_y = None
        if self.edge_weight_loss_weight != 0:
            edge_y = self.edge_decoder(x, edges_config["edge_index_DB"])
        if self.kmer_frequency_loss_weight != 0:
            for node_decoder in self.node_decoders:
                node_y.append(node_decoder(x))
        if self.edit_distance_loss_weight != 0:
            edit_distance_y = self.edit_distance_decoder(
                x, edges_config["edge_index_ED"]
            )
        return edge_y, node_y, edit_distance_y

    def inference(self, graph):
        """
        Computes the embeddings for the provided graph.

        Parameters:
            graph (dict): Input graph data.

        Returns:
            torch.Tensor: Node embeddings.
        """
        # move to cpu as more memory is available
        self.encoder = self.encoder.to("cpu")
        x = graph["node"].x.to("cpu")

        edges_config = {}
        for edge_type in graph.edge_types:
            edges_config["edge_index_" + edge_type[1]] = graph[edge_type].edge_index.to(
                "cpu"
            )
            edges_config["edge_weight_" + edge_type[1]] = graph[edge_type].edge_attr.to(
                "cpu"
            )

        res = self.encoder(x, edges_config)
        compute_dirichlet_energy(edges_config, res)

        return res

    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.

        Parameters:
            batch (dict): Batch of graph data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        self.x = batch["node"].x
        self.edges_config = {}
        batch_size = self.x.size(0)

        for edge_type in batch.edge_types:
            self.edges_config["edge_index_" + edge_type[1]] = batch[
                edge_type
            ].edge_index
            self.edges_config["edge_weight_" + edge_type[1]] = batch[
                edge_type
            ].edge_attr

        if self.edge_weight_loss_weight != 0:
            edge_index, edge_weight = (
                self.edges_config["edge_index_DB"],
                self.edges_config["edge_weight_DB"],
            )
        elif self.edit_distance_loss_weight != 0:
            edge_index, edge_weight = (
                self.edges_config["edge_index_ED"],
                self.edges_config["edge_weight_ED"],
            )
        elif self.kmer_frequency_loss_weight != 0:
            edge_index, edge_weight = (
                self.edges_config["edge_index_DB"],
                self.edges_config["edge_weight_DB"],
            )
        # Mask node features
        num_nodes = self.x.shape[0]
        node_mask = torch.randperm(num_nodes) < int(num_nodes * (1 - self.node_p))
        node_mask = (
            node_mask.float().unsqueeze(-1).to(self.device)
        )  # make mask same dimension as features
        x_masked = self.x * node_mask  # multiply features with mask
        # Mask edge weights info
        num_edges = edge_weight.shape[0]
        mask = torch.randperm(num_edges) < int(num_edges * (self.edge_p))
        edge_weight_masked = edge_weight.clone()
        edge_weight_masked[mask] = 0

        # add negative samples to x, edge_index, edge_weight
        if self.negative_sampling_loss_weight != 0:
            num_neg_samples = int(
                self.x.shape[0] * self.proportion_negative_to_positive_samples
            )
            neg_edge_index = negative_sampling(
                edge_index,
                num_nodes=self.x.size(0),
                num_neg_samples=num_neg_samples,
            )
            edge_index = torch.cat(
                [edge_index.to(self.device), neg_edge_index.to(self.device)], dim=-1
            )
            edge_weight = torch.cat(
                [
                    edge_weight.to(self.device),
                    torch.zeros(num_neg_samples).to(self.device),
                ],
            )
            edge_weight_masked = torch.cat(
                [
                    edge_weight_masked.to(self.device),
                    torch.zeros(num_neg_samples).to(self.device),
                ],
            )
            mask = torch.cat(
                [
                    mask.to(self.device),
                    torch.ones(num_neg_samples, dtype=bool).to(self.device),
                ],
            )

        # Encode and decode
        edge_y, node_y, edit_distance_y = self.forward(
            x_masked, edge_index, edge_weight_masked
        )
        # Compute loss
        loss = 0
        if self.edge_weight_loss_weight != 0:
            edge_loss = self.edge_loss_func(edge_y.squeeze(), edge_weight)
            loss += edge_loss * self.edge_weight_loss_weight
        if self.kmer_frequency_loss_weight != 0:
            for i, node_y_i in enumerate(node_y):
                node_loss = self.node_loss_func(
                    node_y_i.squeeze(), batch["node"][f"y_{i}"]
                )
                loss += node_loss * self.kmer_frequency_loss_weight / len(node_y)
        if self.edit_distance_loss_weight != 0:
            edit_distance_loss = self.edit_distance_loss_func(
                edit_distance_y.squeeze(), edge_weight
            )
            loss += edit_distance_loss * self.edit_distance_loss_weight
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size
        )

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer for training.
        """
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
