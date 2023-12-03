from collections import Counter
from torch_geometric.data import Data

from src.representations.gnn_common.gnn_utils import sample_biased_random_walks


ALPHABET = ["A", "C", "G", "T"]


def sample_biased_random_walks_miniBatch(
    data,
    num_nodes_to_sample: int,
    num_walks: int = 5,
    walk_length: int = 2,
    window_size: int = 3,
    p: float = 1.0,
    q: float = 1.0,
):
    """
    Samples biased random walks from a given graph in mini-batch mode.

    This function extracts the homogeneous ("DB") edges from the input heterogeneous graph
    and performs biased random walks on it. The biased random walks are performed based on
    the node2vec framework, controlled by parameters p and q.

    Parameters:
        data (HeteroData): The input heterogeneous graph.
        num_nodes_to_sample (int): The number of nodes to sample walks from.
        num_walks (int, optional): The number of walks to sample for each node. Default is 5.
        walk_length (int, optional): The length of each walk. Default is 2.
        window_size (int, optional): The window size for considering adjacent nodes. Default is 3.
        p (float, optional): Return parameter for node2vec. Default is 1.0.
        q (float, optional): In-out parameter for node2vec. Default is 1.0.

    Returns:
        Tensor: A tensor representing sampled biased random walks.
    """
    homo_graph = Data()
    homo_graph.edge_index = data["node", "DB", "node"].edge_index
    homo_graph.weight = data["node", "DB", "node"].edge_attr
    homo_graph.x = data["node"].x

    return sample_biased_random_walks(
        homo_graph, num_nodes_to_sample, walk_length, window_size, p, q
    )


def count_edge_types(layers: list):
    """
    Counts the number of occurrences for each edge type in a given list of layers.

    Parameters:
        layers (list): List of layers where each layer is a string. The expected format
                       of each string is 'nodeType_edgeType_nodeType', e.g., 'node_DB_node'.

    Returns:
        int: The maximum count of any edge type, or 2 if the format doesn't match the
             expected pattern. (for MLP and RGCN models)
    """
    if len(layers[0].split("_")) > 1:
        edge_types = [layer.split("_")[1] for layer in layers]
        counts = Counter(edge_types)
        return max(counts.values())
    return 2
