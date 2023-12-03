import torch
from torch import nn
from torch_geometric.utils import negative_sampling
from pytorch_lightning import LightningModule
from torch.optim import Adam
import numpy as np

from src.representations.gnn_common.gnn_utils import (
    cosine_similarity_to_edge_index_weight,
    edit_distance_to_edge_index_weight,
    compute_dirichlet_energy,
    kf_faiss_to_edge_index_weight,
)
from src.representations.gnn_common.gnn_models import (
    InnerProductEdgeDecoder,
    MLPNodeDecoder,
)


class AutoEncoderGeneral(LightningModule):
    """
    Autoencoder model based for metagenomic graphs. The model uses a given encoder to obtain node
    embeddings from input features and then decodes them to recreate graph structures including edge weights,
    k-mer frequencies, or edit distances.

    Attributes:
    - x (torch.Tensor): Node features tensor.
    - y (torch.Tensor): Node labels tensor.
    - edges_config (dict): Dictionary containing various edge index and edge weight information.

    Args:
    - graph (torch.Tensor): Tensor representation of the graph.
    - encoder (nn.Module): Encoder neural network module.
    - edge_weight_loss_weight (float, optional): Loss weight for edge weight prediction. Defaults to 0.
    - kmer_frequency_loss_weight (float, optional): Loss weight for k-mer frequency prediction. Defaults to 0.
    - edit_distance_loss_weight (float, optional): Loss weight for edit distance prediction. Defaults to 0.
    - negative_sampling_loss_weight (float, optional): Weight for the negative sampling loss. Defaults to 0.
    - proportion_negative_to_positive_samples (float, optional): Ratio of negative to positive samples. Defaults to 1.
    - node_p (int, optional): Proportion of nodes to mask. Defaults to 0.
    - edge_p (int, optional): Proportion of edges to mask. Defaults to 0.
    - k (int, optional): Big k-mer size. Defaults to 3.
    - kmer_freq_labels (list, optional): Labels for the k-mer frequencies. Defaults to None.
    - encoder_output_channels (int, optional): Number of output channels from the encoder. Defaults to None.
    - num_labels (list, optional): List of number of labels for each k-mer frequency. Defaults to None.
    - edges_threshold (float, optional): Threshold for edge weights. Defaults to 0.0.
    - edges_keep_top_k (int, optional): Number of top-k edges to keep based on weights. Defaults to None.
    - lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
    - normalise_embeddings (bool, optional): Whether to normalize embeddings after inference. Defaults to True.
    - faiss_ann (bool, optional): Whether to use faiss for approximate nearest neighbours. Defaults to False.

    Raises:
    - AssertionError: If no positive loss weight is provided.
    """

    def __init__(
        self,
        graph: torch.Tensor,
        encoder: nn.Module,
        edge_weight_loss_weight: float = 0,
        kmer_frequency_loss_weight: float = 0,
        edit_distance_loss_weight: float = 0,
        negative_sampling_loss_weight: float = 0,
        proportion_negative_to_positive_samples: float = 1,
        node_p: int = 0,  # proportion of nodes to mask
        edge_p: int = 0,  # proportion of edges to mask
        k: int = 3,  # big kmer size
        kmer_freq_labels: list = None,
        encoder_output_channels: int = None,
        num_labels: list = None,
        edges_threshold: float = 0.0,
        edges_keep_top_k: int = None,
        lr: float = 0.001,  # learning rate
        normalise_embeddings: bool = True,
        faiss_ann: bool = False,
    ):
        super(AutoEncoderGeneral, self).__init__()

        assert (
            edge_weight_loss_weight != 0
            or kmer_frequency_loss_weight != 0
            or edit_distance_loss_weight != 0
        ), "At least one positive loss must be non-zero"

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
        self.normalise_embeddings = normalise_embeddings

        self.edges_config = dict()
        self.x = graph.x
        self.y = graph.y
        self.edges_config["edge_index_DB"] = graph.edge_index
        self.edges_config["edge_weight_DB"] = graph.weight

        self.num_of_additional_strides = 0
        for key, value in graph.items():
            if key.startswith("edge_index") and key != "edge_index":
                self.edges_config["edge_index_DB" + key.split("_")[2]] = value
                self.num_of_additional_strides += 1

            elif key.startswith("edge_weight") and key != "edge_weight":
                self.edges_config["edge_weight_DB" + key.split("_")[2]] = value

        # edge weight
        self.edge_weight_loss_weight = edge_weight_loss_weight
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
                if faiss_ann:
                    (
                        edge_index_KF,
                        weights_KF,
                    ) = kf_faiss_to_edge_index_weight(
                        np.vstack(kmer_freq_labels[i]),
                        k=int(edges_keep_top_k * len(kmer_freq_labels[i])),
                    )
                else:
                    (
                        edge_index_KF,
                        weights_KF,
                    ) = cosine_similarity_to_edge_index_weight(
                        np.vstack(kmer_freq_labels[i]),
                        threshold=edges_threshold,
                        keep_top_k=edges_keep_top_k,
                    )

                self.edges_config[f"edge_index_KF{i}"] = torch.tensor(
                    edge_index_KF, dtype=torch.long
                )
                self.edges_config[f"edge_weight_KF{i}"] = torch.tensor(
                    weights_KF, dtype=torch.float
                )

            self.node_loss_func = nn.MSELoss()

        # edit distance
        self.edit_distance_loss_weight = edit_distance_loss_weight
        if edit_distance_loss_weight != 0:
            (
                edge_index_ED,
                weights_ED,
            ) = edit_distance_to_edge_index_weight(
                k,
                threshold=edges_threshold,
                keep_top_k=edges_keep_top_k,
            )
            self.edges_config["edge_index_ED"] = torch.tensor(
                edge_index_ED, dtype=torch.long
            )
            self.edges_config["edge_weight_ED"] = torch.tensor(
                weights_ED, dtype=torch.float
            )

            self.edit_distance_true = torch.tensor(weights_ED, dtype=torch.float)
            self.all_pairs = torch.tensor(edge_index_ED, dtype=torch.long)
            self.edit_distance_decoder = InnerProductEdgeDecoder()
            self.edit_distance_loss_func = nn.L1Loss()

    def on_train_start(self):
        """
        Function executed at the beginning of the training process.
        Moves the graph, tensors, and associated components to the appropriate device.
        """
        # Ensure the graph and tensors are on the right device
        if self.edge_weight_loss_weight != 0:
            self.edge_decoder = self.edge_decoder.to(self.device)
        if self.kmer_frequency_loss_weight != 0:
            for node_decoder in self.node_decoders:
                node_decoder = node_decoder.to(self.device)
        if self.edit_distance_loss_weight != 0:
            self.edit_distance_decoder = self.edit_distance_decoder.to(self.device)
            self.all_pairs = self.all_pairs.to(self.device)
            self.edit_distance_true = self.edit_distance_true.to(self.device)
        self.x = self.x.to(self.device)
        for i in range(len(self.y)):
            self.y[i] = self.y[i].to(self.device)
        for key in self.edges_config:
            self.edges_config[key] = self.edges_config[key].to(self.device)

    def forward(self, x, edge_index, edge_weight_masked=None):
        """
        Defines the forward pass of the autoencoder.

        Args:
        - x (torch.Tensor): Node feature tensor.
        - edge_index (torch.Tensor): Edge index tensor.
        - edge_weight_masked (torch.Tensor, optional): Edge weights tensor with masked values.

        Returns:
        - tuple: Edge predictions, Node predictions, Edit distance predictions.
        """
        edges_config = self.edges_config.copy()
        edges_config["edge_index_DB"] = edge_index
        if edge_weight_masked is not None:
            edges_config["edge_weight_DB"] = edge_weight_masked
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
            edit_distance_y = self.edit_distance_decoder(x, self.all_pairs)
        return edge_y, node_y, edit_distance_y

    def inference(self):
        """
        Run the inference process.

        Returns:
        - torch.Tensor: The inferred node embeddings.
        """
        self.encoder = self.encoder.to(self.device)
        self.x = self.x.to(self.device)
        for i in range(len(self.y)):
            self.y[i] = self.y[i].to(self.device)
        for key in self.edges_config:
            self.edges_config[key] = self.edges_config[key].to(self.device)

        res = self.encoder(self.x, self.edges_config)
        if self.normalise_embeddings:
            res = torch.nn.functional.normalize(res, p=2.0, dim=0)

        compute_dirichlet_energy(self.edges_config, res)

        return res

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder. Computes the forward pass and calculates the loss.

        Args:
        - batch (list): Batch of data.
        - batch_idx (int): Batch index.

        Returns:
        - torch.Tensor: Computed loss for the batch.
        """
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
            num_neg_samples = (
                edge_index.size(1) * self.proportion_negative_to_positive_samples
            )

            neg_edge_index = negative_sampling(
                edge_index=edge_index,
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
                node_loss = self.node_loss_func(node_y_i.squeeze(), self.y[i])
                loss += node_loss * self.kmer_frequency_loss_weight / len(node_y)
        if self.edit_distance_loss_weight != 0:
            edit_distance_loss = self.edit_distance_loss_func(
                edit_distance_y.squeeze(), self.edit_distance_true
            )
            loss += edit_distance_loss * self.edit_distance_loss_weight

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configuration for the optimizer.

        Returns:
        - torch.optim.Optimizer: Optimizer with provided parameters.
        """
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
