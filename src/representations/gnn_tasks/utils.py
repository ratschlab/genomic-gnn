import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader

ALPHABET = ["A", "C", "G", "T"]


def networkx_to_dataloader(
    graph,
    labels,
    labels_to_predict=None,
    batch_size=1,
    dummy_dataloader=True,
):
    """
    Converts a NetworkX graph to a PyTorch Geometric DataLoader.

    This function takes in a NetworkX graph, node labels, and (optionally)
    labels to predict, and then converts them into a DataLoader suitable
    for graph-based self supervised learning tasks.

    Parameters:
        graph (networkx.Graph): A NetworkX graph.
        labels (list): A list of labels for nodes in the graph.
        labels_to_predict (list, optional): A list of labels that need to be
            predicted. If provided, they will be added to the data object as 'y'.
        batch_size (int, optional): The batch size for the DataLoader. Default is 1.
        dummy_dataloader (bool, optional): If True, a dummy DataLoader with a
            single value (0) will be returned. If False, the DataLoader will
            be created from the given graph data. Default is True.

    Returns:
        torch_geometric.data.DataLoader: DataLoader object containing the graph
        data for use in PyTorch Geometric.
    """

    data = from_networkx(graph)
    data.x = torch.tensor(labels, dtype=torch.float)

    data.y = []
    if labels_to_predict is not None:
        for label in labels_to_predict:
            data.y.append(torch.tensor(label, dtype=torch.float))

    # Create a DataLoader with the specified batch size
    if dummy_dataloader:
        dataloader = DataLoader([0], batch_size=1, num_workers=0)
    else:
        dataloader = DataLoader([data], batch_size=batch_size, num_workers=0)

    return dataloader, data
