import torch
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.representations.gnn_tasks_miniBatch.utils import (
    count_edge_types,
)
from src.representations.gnn_common.gnn_utils import (
    attach_higher_strides_to_torch_graph,
    cosine_similarity_to_edge_index_weight,
    edit_distance_to_edge_index_weight,
    kf_faiss_to_edge_index_weight,
)


ALPHABET = ["A", "C", "G", "T"]


def networkx_to_dataloader(
    graph,
    labels,
    labels_to_predict=None,
    batch_size=4096,
    kmer_frequency_loss_weight: float = 0,
    edit_distance_loss_weight: float = 0,
    kmer_freq_labels: list = None,
    k: int = 3,
    edges_threshold: float = 0.0,
    edges_keep_top_k: int = None,
    layers_config: list = None,
    ss_task: str = None,
    graphs_strides: list = None,
    faiss_ann: bool = False,
    faiss_index_type: str = "IVFFlat",
    faiss_distance: str = "L2",
    faiss_nlist: int = None,
    faiss_nprobe: int = None,
    faiss_m: int = None,
    faiss_nbits: int = None,
):
    """
    Converts a NetworkX graph into a DataLoader suitable for GNN tasks.

    This function processes a NetworkX graph, optionally annotates it with multiple graph
    strides, and prepares it for Graph Neural Network (GNN) tasks by converting it into a
    DataLoader for heterogeneous graphs (HeteroData). Supports different edge type like de
    Bruijn (DB), kmer frequency (KF), and edit distance (ED).

    Parameters:
        graph (NetworkX.Graph): The input graph.
        labels (list): Node labels for the graph.
        labels_to_predict (list, optional): Labels to be predicted for tasks. Default is None.
        batch_size (int, optional): The batch size for DataLoader. Default is 4096.
        kmer_frequency_loss_weight (float, optional): Weight for the kmer frequency loss. Default is 0.
        edit_distance_loss_weight (float, optional): Weight for the edit distance loss. Default is 0.
        kmer_freq_labels (list, optional): Labels for kmer frequency task. Default is None.
        k (int, optional): K-mer size for edit distance calculation. Default is 3.
        edges_threshold (float, optional): Threshold for edge creation based on weights. Default is 0.0.
        edges_keep_top_k (int, optional): If set, keeps only the top K edges based on weights. Default is None.
        layers_config (list, optional): Layer configurations for the GNN. Default is None.
        ss_task (str, optional): Task type, can be 'Sampling' or others. Default is None.
        graphs_strides (list, optional): List of graphs for different strides. Default is None.
        faiss_ann (bool, optional): If True, uses faiss for kmer frequency task. Default is False.
        faiss_index_type (str, optional): Faiss index type. Default is 'IVFFlat'.
        faiss_distance (str, optional): Faiss distance type. Default is 'L2'.
        faiss_nlist (int, optional): Faiss nlist. Default is None.
        faiss_nprobe (int, optional): Faiss nprobe. Default is None.
        faiss_m (int, optional): Faiss m. Default is None.
        faiss_nbits (int, optional): Faiss nbits. Default is None.

    Returns:
        DataLoader: DataLoader object suitable for GNN training.
        HeteroData: Heterogeneous graph data.

    """
    torch_graph = from_networkx(graph)
    data = HeteroData()
    data["node"].x = torch.tensor(np.array(labels), dtype=torch.float)

    for i, new_graph in enumerate(graphs_strides):
        torch_graph = attach_higher_strides_to_torch_graph(
            new_graph, torch_graph, i + 1
        )

    # The edge type definition has been modified to fit the (src_node, edge_type, dst_node) format
    data["node", "DB", "node"].edge_index = torch_graph.edge_index
    data["node", "DB", "node"].edge_attr = torch_graph.weight

    for stride_number in range(len(graphs_strides)):
        data["node", f"DB{stride_number+1}", "node"].edge_index = torch_graph[
            "edge_index_" + str(stride_number + 1)
        ]
        data["node", f"DB{stride_number+1}", "node"].edge_attr = torch_graph[
            "edge_weight_" + str(stride_number + 1)
        ]

    if labels_to_predict is not None:
        for i in range(len(labels_to_predict)):
            data["node"][f"y_{i}"] = torch.tensor(
                labels_to_predict[i], dtype=torch.float
            )

    # kmer frequency
    if kmer_frequency_loss_weight != 0:
        for i, kmer_freq_l in enumerate(kmer_freq_labels):
            if faiss_ann:
                (
                    edge_index_KF,
                    weights_KF,
                ) = kf_faiss_to_edge_index_weight(
                    np.vstack(kmer_freq_l),
                    k=int(edges_keep_top_k * len(kmer_freq_l)),
                    index_type=faiss_index_type,
                    distance=faiss_distance,
                    nlist=faiss_nlist,
                    nprobe=faiss_nprobe,
                    m=faiss_m,
                    nbits=faiss_nbits,
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

            data["node", f"KF{i}", "node"].edge_index = torch.tensor(
                edge_index_KF, dtype=torch.long
            )
            data["node", f"KF{i}", "node"].edge_attr = torch.tensor(
                weights_KF, dtype=torch.float
            )

    # edit distance
    if edit_distance_loss_weight != 0:
        edge_index_ED, weights_ED = edit_distance_to_edge_index_weight(
            k, threshold=edges_threshold, keep_top_k=edges_keep_top_k
        )

        data["node", "ED", "node"].edge_index = torch.tensor(
            edge_index_ED, dtype=torch.long
        )
        data["node", "ED", "node"].edge_attr = torch.tensor(
            weights_ED, dtype=torch.float
        )

    # Identify the connected nodes

    if ss_task == "Sampling":
        all_edge_indices = torch.cat(
            [data[rel].edge_index for rel in data.edge_types], dim=1
        )
        non_self_edges_mask = all_edge_indices[0] != all_edge_indices[1]
        non_self_edge_indices = all_edge_indices[:, non_self_edges_mask]
        connected_nodes = torch.unique(non_self_edge_indices)
        # Use connected_nodes as your input_nodes
        input_nodes = ("node", connected_nodes)

    else:
        # Create a DataLoader with the specified batch size
        input_nodes = ("node", torch.arange(data["node"].num_nodes))

    max_hops = count_edge_types(layers_config)
    dataloader = NeighborLoader(
        data,
        num_neighbors=[-1] * max_hops,
        input_nodes=input_nodes,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        directed=False,
    )

    return dataloader, data
