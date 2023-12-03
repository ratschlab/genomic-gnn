import numpy as np
import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
import wandb

from src.utils import (
    build_deBruijn_graph,
    weighted_directed_edges,
    seq_to_kmers,
)
from src.representations.gnn_tasks.autoencoder_task import AutoEncoderGeneral
from src.representations.gnn_tasks.sampling_task import SamplingGeneral
from src.representations.gnn_common.gnn_models import GCN, GraphSage, GAT, MLP, RGCN
from src.representations.gnn_tasks.utils import networkx_to_dataloader
from src.representations.gnn_common.gnn_utils import (
    node_embedding_initial,
    attach_higher_strides_to_torch_graph,
)

MODEL_FACTORY = {"GCN": GCN, "SAGE": GraphSage, "GAT": GAT, "MLP": MLP, "RGCN": RGCN}


class KmerSsgnn:
    """
    Self-Supervised Graph Neural Network for Kmer representations without MiniBatching/Neighbourhood Sampling.

    Attributes:
    - ssgnn_dataset: List that stores sequence data for self-supervised training.
    """

    def __init__(self):
        """
        Initialize the KmerSsgnn class.
        """
        self.ssgnn_dataset = []

    def fit(self, dfs_list: list, k: int, args: dict):
        """
        Trains the self-supervised graph neural network for k-mer representations.

        Parameters:
        - dfs_list (list): List of dataframes containing sequence data.
        - k (int): Length of k-mer for encoding.
        - args (dict): Dictionary of arguments for model configuration and training.

        Returns:
        - index_dict (dict): Dictionary mapping kmers to indices.
        - layer (nn.Embedding): Embedding layer initialized with weights from the trained model.
        """
        # parameters for defining self-supervised task
        self.initialise_variables(args)

        # build dataset for self-supervised task
        for df in dfs_list:
            self.ssgnn_dataset.extend(df["sequence"].tolist())

        # build graph
        self.create_graphs(k=k)

        # initialise encoder
        self.encoder = MODEL_FACTORY[self.encoder_name](
            num_features=self.num_features,
            layers_config=[
                *self.layers_config,
                str(self.embedding_size)
                + "_"
                + self.representation_ss_last_layer_edge_type,
            ],
            activation=self.activation,
        )

        # initialise task model
        self.initialise_task(k=k)

        for i, task_model in enumerate(self.task_models_list):
            # train encoder
            trainer = Trainer(
                max_epochs=self.representation_epochs,
                accelerator=args["accelerator"],
                # precision=16,
                default_root_dir=args["wandb_dir"],
                callbacks=[TQDMProgressBar(refresh_rate=200)],
            )
            trainer.fit(task_model, self.dataloader)

            wandb.log(
                {
                    ("ssgnn_loss_" + str(i)): trainer.callback_metrics[
                        "train_loss"
                    ].item()
                }
            )

        self.compute_embedding_layer(k=k)

        return self.index_dict, self.layer

    def transform(self, dfs_list: list, index_dict: dict, k: int, args: dict):
        """
        Transforms sequences in the provided dataframes to indices of k-mers.

        Parameters:
        - dfs_list (list): List of dataframes containing sequence data.
        - index_dict (dict): Dictionary mapping k-mers to indices.
        - k (int): Length of k-mer for encoding.
        - args (dict): Dictionary of arguments for model configuration.

        Returns:
        - dfs_list (list): List of dataframes with an added column 'kmers_index_k' containing k-mer indices.
        """
        stride = args["inference_stride"]

        # Function to convert sequence to kmers
        def seq_to_kmers_set(seq):
            return set(seq_to_kmers(seq, k, stride, inlcudeUNK=True))

        # Build a set of all kmers present in the dfs list
        all_kmers = set()
        for df in dfs_list:
            current_kmers = df["sequence"].apply(seq_to_kmers_set)
            for kmer in current_kmers:
                all_kmers.update(kmer)

        # Check if all kmers are in index_dict
        missing_kmers = all_kmers.difference(set(index_dict.keys()))

        if len(missing_kmers) > 0:
            # Compute kmer transition frequencies for missing kmers
            for df in dfs_list:
                self.ssgnn_dataset.extend(df["sequence"].tolist())

            saved_encoders = []

            for i, task_model in enumerate(self.task_models_list):
                saved_encoders.append(task_model.encoder)

            self.create_graphs(k=k)
            self.initialise_task(k=k)
            for i, task_model in enumerate(self.task_models_list):
                task_model.encoder = saved_encoders[i]
            self.compute_embedding_layer(k=k)

            self.new_layer = self.layer
            index_dict = self.index_dict

        def word2index(seq):
            return np.array(
                [
                    index_dict[kmer]
                    for kmer in seq_to_kmers(seq, k, stride, inlcudeUNK=True)
                ]
            )

        # include UNK for final representations
        for df in dfs_list:
            df[("kmers_index_" + str(k))] = df["sequence"].apply(word2index)

        return dfs_list

    def create_graphs(
        self,
        k: int,
    ):
        """
        Creates de Bruijn graphs based on the k-mer representation and data.

        Parameters:
        - k (int): Length of k-mer for graph construction.
        """

        # build deBruijn graph
        _, pair_frequency = weighted_directed_edges(
            self.ssgnn_dataset,
            k=k,
            stride=self.strides[0],
            inlcudeUNK=False,
            disable_tqdm=self.disable_tqdm,
        )
        self.graph = build_deBruijn_graph(
            pair_frequency,
            normalise=True,
            remove_N=True,
            create_all_kmers=self.create_all_kmers,
            disable_tqdm=self.disable_tqdm,
        )

        # generate labels
        self.kf_labels = []
        for s_k in self.small_k.split(","):
            self.kf_labels.append(
                node_embedding_initial(
                    self.graph, method="kmer_frequency", k=k, small_k=int(s_k)
                )
            )

        if self.representation_ss_initial_labels == "one_hot":
            onehot_labels = node_embedding_initial(self.graph, method="onehot", k=k)
            # convert to dataloader
            self.dataloader, self.torch_graph = networkx_to_dataloader(
                self.graph,
                onehot_labels,
                labels_to_predict=self.kf_labels,
            )
            self.num_features = 4**k
        else:
            self.dataloader, self.torch_graph = networkx_to_dataloader(
                self.graph,
                self.kf_labels[0],
                labels_to_predict=self.kf_labels,
            )
            self.num_features = 4 ** int(self.small_k[0])

        # attach additional edges and weights to graph based on higher strides
        for i, stride in enumerate(self.strides[1:]):
            _, pair_frequency = weighted_directed_edges(
                self.ssgnn_dataset, k=k, stride=stride, inlcudeUNK=False
            )
            new_graph = build_deBruijn_graph(
                pair_frequency,
                normalise=True,
                remove_N=True,
                create_all_kmers=self.create_all_kmers,
                edge_weight_threshold=1 / float(4**k),
                keep_top_k=self.edges_keep_top_k,
            )
            self.torch_graph = attach_higher_strides_to_torch_graph(
                new_graph, self.torch_graph, i + 1
            )

    def initialise_task(self, k: int):
        """
        Initialize the self-supervised task based on the model configuration.

        Parameters:
        - k (int): Length of k-mer for the task.
        """
        self.task_models_list = []

        if self.ss_task == "AE":
            num_labels = []
            for s_k in self.small_k.split(","):
                num_labels.append(4 ** int(s_k))
            self.task_models_list.append(
                AutoEncoderGeneral(
                    graph=self.torch_graph,
                    encoder=self.encoder,
                    edge_weight_loss_weight=self.edge_weight_loss_weight,
                    kmer_frequency_loss_weight=self.kmer_frequency_loss_weight,
                    edit_distance_loss_weight=self.edit_distance_loss_weight,
                    negative_sampling_loss_weight=self.negative_sampling_loss_weight,
                    proportion_negative_to_positive_samples=self.proportion_negative_to_positive_samples,
                    node_p=self.p_node,
                    edge_p=self.p_edge,
                    k=k,
                    kmer_freq_labels=self.kf_labels,
                    encoder_output_channels=self.embedding_size,
                    num_labels=num_labels,
                    edges_threshold=self.edges_threshold,
                    edges_keep_top_k=self.edges_keep_top_k,
                    lr=self.lr,
                    normalise_embeddings=self.normalise_embeddings,
                    faiss_ann=self.faiss_ann,
                )
            )

        elif self.ss_task == "CL":
            self.task_models_list.append(
                SamplingGeneral(
                    graph=self.torch_graph,
                    encoder=self.encoder,
                    edge_weight_loss_weight=self.edge_weight_loss_weight,
                    kmer_frequency_loss_weight=self.kmer_frequency_loss_weight,
                    edit_distance_loss_weight=self.edit_distance_loss_weight,
                    negative_sampling_loss_weight=self.negative_sampling_loss_weight,
                    walk_length=self.sampling_walk_length,
                    num_walks=self.sampling_num_walks,
                    window_size=self.sampling_window_size,
                    proportion_nodes_to_sample=self.sampling_proportion_nodes_to_sample,
                    proportion_negative_to_positive_samples=self.proportion_negative_to_positive_samples,
                    rw_p=self.rw_p,
                    rw_q=self.rw_q,
                    resample_every_num_epochs=self.resample_every_num_epochs,
                    kmer_freq_labels=self.kf_labels,
                    k=k,
                    node_p=self.p_node,
                    edge_p=self.p_edge,
                    edges_threshold=self.edges_threshold,
                    edges_keep_top_k=self.edges_keep_top_k,
                    lr=self.lr,
                    faiss_ann=self.faiss_ann,
                )
            )
        else:
            raise ValueError("Invalid Self-supervised task")

    def compute_embedding_layer(self, k: int):
        """
        Computes the embedding layer after model training.

        Parameters:
        - k (int): Length of k-mer for embedding.
        """
        with torch.no_grad():
            z = self.task_models_list[0].inference()
            for i, task_model in enumerate(self.task_models_list):
                if i > 0:
                    z_2 = task_model.inference()
                    z = torch.cat((z, z_2), dim=1)

        kmer_to_representation = {
            label: z[i].cpu().numpy() for i, label in enumerate(self.graph.nodes())
        }

        # Check if weights and index_dict exist
        if not hasattr(self, "weights"):
            self.weights = np.zeros(
                (len(kmer_to_representation) + 1, self.embedding_size)
            )
            self.index_dict = {}
            self.index_dict["N" * k] = len(kmer_to_representation)

        new_kmers = []
        for kmer in kmer_to_representation.keys():
            if kmer not in self.index_dict.keys():
                new_kmers.append(kmer)

        # For every new kmer, update the index_dict and the weights
        for kmer in new_kmers:
            if kmer == "N" * k:  # Skip the "N" * k kmer
                continue
            index = len(self.index_dict)
            self.index_dict[kmer] = index

        # Ensure the size of the weights is correct
        old_weights = self.weights
        self.weights = np.zeros((len(self.index_dict) + 1, self.weights.shape[1]))

        # Copy over the old weights
        for kmer, index in self.index_dict.items():
            if kmer in new_kmers:
                self.weights[index] = kmer_to_representation[kmer]
            else:
                self.weights[index] = old_weights[index]

        self.layer = nn.Embedding.from_pretrained(torch.FloatTensor(self.weights))

    def initialise_variables(self, args):
        """
        Initializes various variables and parameters needed for model training and configuration.

        Parameters:
        - args (dict): Dictionary of arguments for model configuration.
        """
        self.strides = args["representation_stride"]
        self.embedding_size = args["representation_size"]
        self.small_k = args["representation_small_k"]
        self.ss_task = args["representation_ss_task"]
        self.edge_weight_loss_weight = args["representation_ss_edge_weight_loss_weight"]
        self.kmer_frequency_loss_weight = args[
            "representation_ss_kmer_frequency_loss_weight"
        ]
        self.edit_distance_loss_weight = args[
            "representation_ss_edit_distance_loss_weight"
        ]
        self.negative_sampling_loss_weight = args[
            "representation_ss_negative_sampling_loss_weight"
        ]
        self.encoder_name = args["representation_ss_encoder"]
        self.layers_config = args["representation_ss_hidden_channels"]
        self.representation_ss_last_layer_edge_type = args[
            "representation_ss_last_layer_edge_type"
        ]
        self.activation = args["representation_ss_activation"]
        self.lr = args["representation_ss_lr"]
        self.representation_epochs = args["representation_ss_epochs"]
        self.p_edge = args["representation_ss_probability_masking_edges"]
        self.p_node = args["representation_ss_probability_masking_nodes"]
        self.sampling_walk_length = args["representation_ss_sampling_walk_length"]
        self.sampling_num_walks = args["representation_ss_sampling_num_walks"]
        self.sampling_window_size = args["representation_ss_sampling_window_size"]
        self.sampling_proportion_nodes_to_sample = args[
            "representation_ss_sampling_proportion_nodes_to_sample"
        ]
        self.resample_every_num_epochs = args[
            "representation_ss_resample_every_num_epochs"
        ]
        self.proportion_negative_to_positive_samples = args[
            "representation_ss_proportion_negative_to_positive_samples"
        ]
        self.rw_p = args["representation_ss_rw_p"]
        self.rw_q = args["representation_ss_rw_q"]
        self.edges_threshold = args["representation_ss_edges_threshold"]
        self.edges_keep_top_k = args["representation_ss_edges_keep_top_k"]
        self.representation_ss_initial_labels = args["representation_ss_initial_labels"]
        self.create_all_kmers = args["representation_ss_create_all_kmers"]
        self.normalise_embeddings = args["representation_ss_normalise_embeddings"]
        self.faiss_ann = args["representation_ss_faiss_ann"]
        self.disable_tqdm = args["disable_tqdm"]
