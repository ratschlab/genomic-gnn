import torch
from torch import nn
from torch_geometric.utils import negative_sampling
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
import numpy as np
from copy import deepcopy

from src.representations.gnn_tasks_miniBatch.utils import (
    sample_biased_random_walks_miniBatch as sample_biased_random_walks,
)
from src.representations.gnn_common.gnn_utils import compute_dirichlet_energy


class SamplingGeneral(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        edge_weight_loss_weight: float = 0,
        kmer_frequency_loss_weight: float = 0,
        edit_distance_loss_weight: float = 0,
        negative_sampling_loss_weight: float = 0,
        walk_length: int = 5,
        num_walks: int = 20,
        window_size: int = 1,
        proportion_nodes_to_sample: float = 0.5,
        proportion_negative_to_positive_samples: float = 1,
        rw_p: float = 1,
        rw_q: float = 1,
        node_p: float = 0,
        edge_p: float = 0,
        lr: float = 0.001,
    ):
        """
        Initializes the SamplingGeneral class with specified parameters for
        metagenomic graph representation learning via CONTRASTIVE LEARNING.
        USES MINI-BATCHES.

        Parameters:
            encoder (nn.Module): The graph encoder module.
            edge_weight_loss_weight (float, optional): Weight for the edge weight loss. Default is 0.
            kmer_frequency_loss_weight (float, optional): Weight for kmer frequency loss. Default is 0.
            edit_distance_loss_weight (float, optional): Weight for edit distance loss. Default is 0.
            negative_sampling_loss_weight (float, optional): Weight for negative sampling loss. Default is 0.
            walk_length (int, optional): Walk length for biased random walks. Default is 5.
            num_walks (int, optional): Number of walks for biased random walks. Default is 20.
            window_size (int, optional): Window size for biased random walks. Default is 1.
            proportion_nodes_to_sample (float, optional): Proportion of nodes to sample. Default is 0.5.
            proportion_negative_to_positive_samples (float, optional): Proportion of negative samples to positive samples. Default is 1.
            rw_p (float, optional): Return parameter for biased random walks. Default is 1.
            rw_q (float, optional): In-out parameter for biased random walks. Default is 1.
            node_p (float, optional): Proportion of nodes to mask. Default is 0.
            edge_p (float, optional): Proportion of edges to mask. Default is 0.
            lr (float, optional): Learning rate. Default is 0.001.
        """

        super(SamplingGeneral, self).__init__()

        assert (
            edge_weight_loss_weight != 0
            or kmer_frequency_loss_weight != 0
            or edit_distance_loss_weight != 0
        ), "At least one positive loss must be non-zero"

        # general parameters
        self.encoder = encoder
        print("!!!encoder!!!\n", self.encoder)
        self.negative_sampling_loss_weight = negative_sampling_loss_weight
        self.proportion_nodes_to_sample = proportion_nodes_to_sample
        self.proportion_negative_to_positive_samples = (
            proportion_negative_to_positive_samples
        )
        self.lr = lr
        self.node_p = node_p
        self.edge_p = edge_p
        self.edge_weight_loss_weight = edge_weight_loss_weight
        if edge_weight_loss_weight != 0:
            self.walk_length = walk_length
            self.num_walks = num_walks
            self.window_size = window_size
            self.rw_p = rw_p
            self.rw_q = rw_q

        self.kmer_frequency_loss_weight = kmer_frequency_loss_weight

        self.edit_distance_loss_weight = edit_distance_loss_weight

        ####

    def forward(self, x, edges_config=None):
        """
        Passes the input node features through the encoder.

        Parameters:
            x (torch.Tensor): Node features.
            edges_config (dict, optional): Configuration of edges for encoding. If None, uses the internal configuration.

        Returns:
            torch.Tensor: Encoded node features.
        """
        if edges_config:
            return self.encoder(x, edges_config)
        return self.encoder(x, self.edges_config)

    def inference(self, graph):
        """
        Computes node embeddings for the given graph using the encoder.

        Parameters:
            graph (dict): Input graph data.

        Returns:
            torch.Tensor: Node embeddings.
        """
        # move to cpu as more memory is available
        x = graph["node"].x.to("cpu")
        self.encoder = self.encoder.to("cpu")

        edges_config = {}
        for edge_type in graph.edge_types:
            edges_config["edge_index_" + edge_type[1]] = graph[edge_type].edge_index.to(
                "cpu"
            )
            edges_config["edge_weight_" + edge_type[1]] = graph[edge_type].edge_attr.to(
                "cpu"
            )

        res = self.forward(x, edges_config)
        compute_dirichlet_energy(edges_config, res)

        return res

    def masking(self):
        """
        Masks a specified proportion of nodes and edges in the graph.

        Returns:
            tuple: Masked node features and edge configurations.
        """
        x = self.x
        if self.node_p != 0:
            num_nodes = x.shape[0]
            node_mask = torch.randperm(num_nodes) < int(num_nodes * (1 - self.node_p))
            node_mask = node_mask.float().unsqueeze(-1).to(self.device)
            x = x * node_mask

        edges_config = deepcopy(self.edges_config)
        if self.edge_p != 0:
            for key in edges_config.keys():
                if key.startswith("edge_weight"):
                    num_edges = edges_config[key].shape[0]
                    mask = torch.randperm(num_edges) < int(num_edges * (self.edge_p))
                    edges_config[key][mask] = 0

        return x, edges_config

    def sample_pairs(self):
        """
        Samples positive node pairs based on various graph tasks (edge weight, kmer frequency, edit distance)
        and negative samples for the tasks.
        """
        num_of_nodes_to_sample = int(self.x.shape[0] * self.proportion_nodes_to_sample)
        total_positive_samples = 0

        # edge weight sampling
        if self.edge_weight_loss_weight != 0:
            self.pos_edge_index = sample_biased_random_walks(
                self.batch,
                num_of_nodes_to_sample,
                num_walks=self.num_walks,
                walk_length=self.walk_length,
                window_size=self.window_size,
                p=self.rw_p,
                q=self.rw_q,
            )
            if self.pos_edge_index.shape[0]:
                total_positive_samples += self.pos_edge_index.size(1)
                num_of_nodes_to_sample = self.pos_edge_index.size(1)

        # kmer frequency sampling
        if self.kmer_frequency_loss_weight != 0:
            self.pos_edge_index_KF = []
            for i, edge_type in enumerate(self.batch.edge_types):
                if not edge_type[1].startswith("KF"):
                    continue

                edge_index = self.batch[edge_type]["edge_index"]
                edge_weight = self.batch[edge_type]["edge_attr"]

                p_vector_KF = edge_weight / edge_weight.sum()

                indices = np.random.choice(
                    edge_index.shape[1],
                    size=num_of_nodes_to_sample,
                    p=p_vector_KF.cpu().numpy(),
                    replace=True,
                )
                indices = edge_index[:, indices.astype(int)]

                self.pos_edge_index_KF.append(indices)
                total_positive_samples += num_of_nodes_to_sample

        # edit distance sampling
        if self.edit_distance_loss_weight != 0:
            edge_index = self.edges_config["edge_index_ED"]
            edge_weight = self.edges_config["edge_weight_ED"]

            p_vector_ED = edge_weight / edge_weight.sum()

            indices = np.random.choice(
                edge_index.shape[1],
                size=num_of_nodes_to_sample,
                p=p_vector_ED.cpu().numpy(),
                replace=True,
            )
            indices = edge_index[:, indices.astype(int)]
            self.pos_edge_index_ED = indices
            total_positive_samples += num_of_nodes_to_sample

        # negative sampling
        if self.negative_sampling_loss_weight != 0:
            self.neg_edge_index = negative_sampling(
                torch.tensor([[], []]),
                num_nodes=self.x.size(0),
                num_neg_samples=int(
                    total_positive_samples
                    * self.proportion_negative_to_positive_samples
                ),
            )

    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.

        Parameters:
            batch (dict): Batch of graph data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        self.batch = batch
        self.x = batch["node"].x
        self.edges_config = {}

        batch_size = self.x.size(0)
        self.batch.num_nodes = batch_size

        for edge_type in batch.edge_types:
            self.edges_config["edge_index_" + edge_type[1]] = batch[
                edge_type
            ].edge_index
            self.edges_config["edge_weight_" + edge_type[1]] = batch[
                edge_type
            ].edge_attr

        # positive sampling
        self.sample_pairs()

        # loss calculation
        x, edges_config = self.masking()
        node_embeddings = self.forward(x, edges_config)

        loss = 0

        if self.edge_weight_loss_weight != 0 and self.pos_edge_index.shape[0]:
            loss += (
                -F.logsigmoid(
                    torch.sum(
                        node_embeddings[self.pos_edge_index[0]]
                        * node_embeddings[self.pos_edge_index[1]],
                        dim=-1,
                    )
                ).mean()
                * self.edge_weight_loss_weight
            )

        if self.kmer_frequency_loss_weight != 0:
            for i in range(len(self.pos_edge_index_KF)):
                if self.pos_edge_index_KF[i].shape[0]:
                    loss += (
                        -F.logsigmoid(
                            torch.sum(
                                node_embeddings[self.pos_edge_index_KF[i][0]]
                                * node_embeddings[self.pos_edge_index_KF[i][1]],
                                dim=-1,
                            )
                        ).mean()
                        * self.kmer_frequency_loss_weight
                    )

        if self.edit_distance_loss_weight != 0 and self.pos_edge_index_ED.shape[0]:
            loss += (
                -F.logsigmoid(
                    torch.sum(
                        node_embeddings[self.pos_edge_index_ED[0]]
                        * node_embeddings[self.pos_edge_index_ED[1]],
                        dim=-1,
                    )
                ).mean()
                * self.edit_distance_loss_weight
            )

        if self.negative_sampling_loss_weight != 0:
            loss += (
                -F.logsigmoid(
                    -torch.sum(
                        node_embeddings[self.neg_edge_index[0]]
                        * node_embeddings[self.neg_edge_index[1]],
                        dim=-1,
                    )
                ).mean()
                * self.negative_sampling_loss_weight
            )

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
        return Adam(self.parameters(), lr=self.lr)
