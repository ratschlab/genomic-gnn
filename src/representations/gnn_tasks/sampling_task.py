import torch
from torch import nn
from torch_geometric.utils import negative_sampling
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
import numpy as np
from copy import deepcopy

from src.representations.gnn_common.gnn_utils import (
    sample_biased_random_walks,
    cosine_similarity_to_edge_index_weight,
    kf_faiss_to_edge_index_weight,
    edit_distance_to_edge_index_weight,
    compute_dirichlet_energy,
)


class SamplingGeneral(LightningModule):
    def __init__(
        self,
        graph: torch.Tensor,
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
        resample_every_num_epochs: int = -1,
        kmer_freq_labels: list = None,
        k: int = 3,
        node_p: float = 0,
        edge_p: float = 0,
        edges_threshold: float = 0.0,
        edges_keep_top_k: int = None,
        lr: float = 0.001,
        faiss_ann: bool = False,
    ):
        """
        Initializes the SamplingGeneral class with specified parameters for
        metagenomic graph representation learning via CONTRASTIVE LEARNING.

        Args:
        - graph (torch.Tensor): Tensor representation of the graph.
        - encoder (nn.Module): Encoder neural network module.
        - edge_weight_loss_weight (float, optional): Weight for the edge weight loss. Defaults to 0.
        - kmer_frequency_loss_weight (float, optional): Weight for the k-mer frequency loss. Defaults to 0.
        - edit_distance_loss_weight (float, optional): Weight for the edit distance loss. Defaults to 0.
        - negative_sampling_loss_weight (float, optional): Weight for the negative sampling loss. Defaults to 0.
        - walk_length (int, optional): Length of the random walk for sampling. Defaults to 5.
        - num_walks (int, optional): Number of random walks per node. Defaults to 20.
        - window_size (int, optional): Context window size for the skip-gram model. Defaults to 1.
        - proportion_nodes_to_sample (float, optional): Proportion of nodes to sample in each epoch. Defaults to 0.5.
        - proportion_negative_to_positive_samples (float, optional): Proportion of negative to positive samples for negative sampling. Defaults to 1.
        - rw_p (float, optional): Return parameter for biased random walk. Defaults to 1.
        - rw_q (float, optional): In-out parameter for biased random walk. Defaults to 1.
        - resample_every_num_epochs (int, optional): Number of epochs after which to resample nodes. If set to -1, nodes won't be resampled. Defaults to -1.
        - kmer_freq_labels (list, optional): List of labels for k-mer frequencies. Defaults to None.
        - k (int, optional): Size of the big k-mer. Defaults to 3.
        - node_p (float, optional): Proportion of nodes to mask. Defaults to 0.
        - edge_p (float, optional): Proportion of edges to mask. Defaults to 0.
        - edges_threshold (float, optional): Threshold value for edge weights. Defaults to 0.0.
        - edges_keep_top_k (int, optional): Number of top-k edges to keep based on weights. Defaults to None.
        - lr (float, optional): Learning rate for optimization. Defaults to 0.001.
        - faiss_ann (bool, optional): Whether to use faiss for k-mer frequency loss. Defaults to False.

        Raises:
        - AssertionError: If all positive loss weights are zero.

        Note:
        - The class implements an approach that combines random walks, k-mer frequencies, and edit distances
          for graph representation learning. It supports negative sampling, biased random walks, and masking techniques.
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
        self.resample_every_num_epochs = resample_every_num_epochs
        if resample_every_num_epochs == -1:
            self.resample_every_num_epochs = 999999999

        self.edges_config = dict()
        self.graph = graph
        self.x = graph.x

        # edge weight
        self.edges_config["edge_index_DB"] = graph.edge_index
        self.edges_config["edge_weight_DB"] = graph.weight

        self.num_of_additional_strides = 0
        for key, value in graph.items():
            if key.startswith("edge_index") and key != "edge_index":
                self.edges_config["edge_index_DB" + key.split("_")[2]] = value
                self.num_of_additional_strides += 1

            elif key.startswith("edge_weight") and key != "edge_weight":
                self.edges_config["edge_weight_DB" + key.split("_")[2]] = value

        self.edge_weight_loss_weight = edge_weight_loss_weight
        if edge_weight_loss_weight != 0:
            self.walk_length = walk_length
            self.num_walks = num_walks
            self.window_size = window_size
            self.rw_p = rw_p
            self.rw_q = rw_q

        # kmer frequency
        self.kmer_frequency_loss_weight = kmer_frequency_loss_weight
        if kmer_frequency_loss_weight != 0:
            self.p_vectors_KF = []
            self.sample_index_KF = []
            for i, kmer_freq_l in enumerate(kmer_freq_labels):
                if faiss_ann:
                    (
                        edge_index_KF,
                        weights_KF,
                    ) = kf_faiss_to_edge_index_weight(
                        np.vstack(kmer_freq_l),
                        k=int(edges_keep_top_k * len(kmer_freq_l)),
                    )
                else:
                    (
                        edge_index_KF,
                        weights_KF,
                    ) = cosine_similarity_to_edge_index_weight(
                        np.vstack(kmer_freq_l),
                        threshold=edges_threshold,
                        keep_top_k=edges_keep_top_k,
                    )
                self.edges_config[f"edge_index_KF{i}"] = torch.tensor(
                    edge_index_KF, dtype=torch.long
                )
                self.edges_config[f"edge_weight_KF{i}"] = torch.tensor(
                    weights_KF, dtype=torch.float
                )

                weights_KF /= weights_KF.sum()
                self.p_vectors_KF.append(weights_KF)
                self.sample_index_KF.append(edge_index_KF.T)
                del weights_KF, edge_index_KF

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
            weights_ED /= weights_ED.sum()
            self.p_vectors_ED = weights_ED
            self.sample_index_ED = edge_index_ED.T
            del edge_index_ED, weights_ED

    def on_train_start(self):
        """Ensures the graph and tensors are loaded onto the correct device."""
        self.x = self.x.to(self.device)
        for key in self.edges_config:
            self.edges_config[key] = self.edges_config[key].to(self.device)

    def forward(self, x, edges_config=None):
        """
        Computes the node embeddings for the graph.

        Parameters:
            x (torch.Tensor): Input node features.
            edges_config (dict, optional): Configuration of edges. Uses class-defined edges_config if not provided.

        Returns:
            torch.Tensor: Node embeddings.
        """
        if edges_config:
            return self.encoder(x, edges_config)
        return self.encoder(x, self.edges_config)

    def inference(self):
        """Inference method that computes node representations and evaluates their quality using Dirichlet energy."""
        self.encoder = self.encoder.to(self.device)
        self.x = self.x.to(self.device)
        for key in self.edges_config:
            self.edges_config[key] = self.edges_config[key].to(self.device)

        res = self.encoder(self.x, self.edges_config)

        compute_dirichlet_energy(self.edges_config, res)

        return res

    def masking(self):
        """
        Applies masking on nodes and edges based on given probabilities.

        Returns:
            tuple: Masked node features and edges configuration.
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
        """Performs positive and negative sampling based on the provided configurations."""
        num_of_nodes_to_sample = int(
            self.edges_config["edge_index_DB"].size(1) * self.proportion_nodes_to_sample
        )
        total_positive_samples = 0

        # edge weight sampling
        if self.edge_weight_loss_weight != 0:
            self.pos_edge_index = sample_biased_random_walks(
                self.graph,
                num_of_nodes_to_sample,
                num_walks=self.num_walks,
                walk_length=self.walk_length,
                window_size=self.window_size,
                p=self.rw_p,
                q=self.rw_q,
            )
            num_of_nodes_to_sample = self.pos_edge_index.size(1)
            total_positive_samples += self.pos_edge_index.size(1)

        # kmer frequency sampling
        if self.kmer_frequency_loss_weight != 0:
            self.pos_edge_index_KF = []
            for i in range(len(self.p_vectors_KF)):
                indices = np.random.choice(
                    range(self.sample_index_KF[i].shape[0]),
                    size=num_of_nodes_to_sample // len(self.p_vectors_KF),
                    p=self.p_vectors_KF[i],
                    replace=True,
                )
                indices = self.sample_index_KF[i][indices.astype(int)]

                self.pos_edge_index_KF.append(torch.tensor(indices.T, dtype=torch.long))
                total_positive_samples += num_of_nodes_to_sample // len(
                    self.p_vectors_KF
                )
                if self.resample_every_num_epochs == 999999999:
                    self.p_vectors_KF[i] = None
                    self.sample_index_KF[i] = None

        if self.edit_distance_loss_weight != 0:
            indices = np.random.choice(
                self.sample_index_ED.shape[0],
                size=num_of_nodes_to_sample,
                p=self.p_vectors_ED,
                replace=True,
            )
            indices = self.sample_index_ED[indices.astype(int)]
            self.pos_edge_index_ED = torch.tensor(indices.T, dtype=torch.long)
            total_positive_samples += num_of_nodes_to_sample

            if self.resample_every_num_epochs == 999999999:
                self.p_vectors_KF = None
                self.sample_index_KF = None

        # negative sampling
        if self.negative_sampling_loss_weight != 0:
            self.neg_edge_index = negative_sampling(
                edge_index=torch.tensor([[], []]),
                num_nodes=self.x.size(0),
                num_neg_samples=int(
                    total_positive_samples
                    * self.proportion_negative_to_positive_samples
                ),
            )

    def training_step(self, batch, batch_idx):
        """
        Defines the training loop for one iteration.

        Parameters:
            batch: Batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the current batch.
        """
        # positive sampling
        if self.current_epoch % self.resample_every_num_epochs == 0:
            self.sample_pairs()

        # loss calculation
        x, edges_config = self.masking()
        node_embeddings = self.forward(x, edges_config)

        loss = 0

        if self.edge_weight_loss_weight != 0:
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

        if self.edit_distance_loss_weight != 0:
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

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Specifies the optimizer to use during training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return Adam(self.parameters(), lr=self.lr)
