import torch
from torch.nn import Module, ReLU, Linear, Sequential
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, RGCNConv


# convert the simplified config to dict
def interpret_layers_config(layers_config: list):
    """
    Converts a simplified configuration to dictionary format.

    Args:
        layers_config (list): A list of string configurations for each layer.

    Returns:
        list[dict]: A list of dictionaries, each representing a layer's configuration.
    """
    new_config = []
    for config in layers_config:
        channels, edge = config.split("_")
        new_config.append(
            {
                "channels": int(channels),
                "edge_index": f"edge_index_{edge}",
                "edge_weight": f"edge_weight_{edge}",
            }
        )
    return new_config


class MLP(Module):
    """
    A Multi-layer Perceptron (MLP) neural network.

    Args:
        num_features (int): Number of input features.
        layers_config (list): Configuration of network layers.
        activation (str, optional): Activation function. Default is "ReLU".
    """

    def __init__(self, num_features, layers_config: list, activation: str = "ReLU"):
        super(MLP, self).__init__()
        layers_config[-1] = layers_config[-1].split("_")[0]
        self.layers_config = [int(x) for x in layers_config]

        self.conv_layers = Sequential()
        self.conv_layers.add_module("lin0", Linear(num_features, self.layers_config[0]))
        for i, layers_config in enumerate(self.layers_config[1:], start=1):
            self.conv_layers.add_module(f"act{i}", getattr(torch.nn, activation)())
            self.conv_layers.add_module(
                f"lin{i}",
                Linear(self.layers_config[i - 1], layers_config),
            )

    def forward(self, x, edges_config=None):
        return self.conv_layers(x)


class GCN(Module):
    """
    Graph Convolutional Network (GCN) implementation.

    Args:
        num_features (int): Number of input features.
        layers_config (list): Configuration of network layers.
        activation (str, optional): Activation function. Default is "ReLU".
    """

    def __init__(self, num_features, layers_config: list, activation: str = "ReLU"):
        super(GCN, self).__init__()
        self.layers_config = interpret_layers_config(layers_config)

        self.conv_layers = Sequential()
        self.conv_layers.add_module(
            "conv0", GCNConv(num_features, self.layers_config[0]["channels"])
        )

        for i, layer_config in enumerate(self.layers_config[1:], start=1):
            self.conv_layers.add_module(f"act{i}", getattr(torch.nn, activation)())
            self.conv_layers.add_module(
                f"conv{i}",
                GCNConv(
                    self.layers_config[i - 1]["channels"], layer_config["channels"]
                ),
            )

    def forward(self, x, edges_config):
        n_conv = 0
        for layer in self.conv_layers:
            if isinstance(layer, GCNConv):
                edge_index = edges_config[self.layers_config[n_conv]["edge_index"]]
                edge_weight = edges_config.get(
                    self.layers_config[n_conv]["edge_weight"]
                )
                x = layer(x, edge_index, edge_weight)
                n_conv += 1

            else:
                x = layer(x)

        return x


class RGCN(Module):
    """
    Relational Graph Convolutional Network (RGCN) implementation.

    Args:
        num_features (int): Number of input features.
        layers_config (list): Configuration of network layers.
        activation (str, optional): Activation function. Default is "ReLU".
    """

    def __init__(self, num_features, layers_config: list, activation: str = "ReLU"):
        super(RGCN, self).__init__()

        # the last element of layers_config is the edge types
        self.edge_types = layers_config[-1].split("_")[1:]
        layers_config[-1] = layers_config[-1].split("_")[0]
        self.layers_config = [int(x) for x in layers_config]

        self.num_relations = len(
            self.edge_types
        )  # number of relations is the number of distinct edge types

        self.conv_layers = Sequential()
        self.conv_layers.add_module(
            "conv0",
            RGCNConv(num_features, self.layers_config[0], self.num_relations),
        )

        for i, layer_config in enumerate(self.layers_config[1:], start=1):
            self.conv_layers.add_module(f"act{i}", getattr(torch.nn, activation)())
            self.conv_layers.add_module(
                f"conv{i}",
                RGCNConv(
                    self.layers_config[i - 1],
                    layer_config,
                    self.num_relations,
                ),
            )

        self.edge_type_mapping = (
            {}
        )  # a mapping dictionary for edge types to integer IDs

    def forward(self, x, edges_config):
        edge_type = []
        edge_index = []
        edge_weight = []

        for key, value in edges_config.items():
            if key.startswith("edge_index"):
                edge_type_str = key.split("_")[-1]  # original string edge type
                if (
                    edge_type_str in self.edge_types
                ):  # only consider the edge types that are in the layers_config
                    if (
                        edge_type_str not in self.edge_type_mapping
                    ):  # create a new ID if this edge type hasn't been seen
                        self.edge_type_mapping[edge_type_str] = len(
                            self.edge_type_mapping
                        )
                    edge_type_id = self.edge_type_mapping[edge_type_str]

                    edge_type.append(
                        torch.full((value.size(1),), edge_type_id, dtype=torch.long)
                    )
                    edge_index.append(value)

                    weight_key = "edge_weight_" + edge_type_str
                    if weight_key in edges_config:
                        edge_weight.append(edges_config[weight_key])

        edge_type = torch.cat(edge_type)
        edge_index = torch.cat(edge_index, dim=1)
        edge_weight = torch.cat(edge_weight) if edge_weight else None

        n_conv = 0
        for layer in self.conv_layers:
            if isinstance(layer, RGCNConv):
                x = layer(x, edge_index, edge_type)
                n_conv += 1
            else:
                x = layer(x)
        return x


class GAT(Module):
    """
    Graph Attention Network (GAT) implementation.

    Args:
        num_features (int): Number of input features.
        layers_config (list): Configuration of network layers.
        heads (int, optional): Number of attention heads. Default is 1.
        activation (str, optional): Activation function. Default is "ReLU".
    """

    def __init__(
        self,
        num_features,
        layers_config: list,
        heads: int = 1,
        activation: str = "ReLU",
    ):
        super(GAT, self).__init__()
        self.layers_config = interpret_layers_config(layers_config)
        self.heads = heads

        self.conv_layers = Sequential()

        if len(self.layers_config) == 1:
            self.conv_layers.add_module(
                "conv0",
                GATConv(
                    num_features, self.layers_config[0]["channels"], heads=1
                ),  # The final layer averages the heads
            )
        else:
            self.conv_layers.add_module(
                "conv0",
                GATConv(
                    num_features, self.layers_config[0]["channels"] * heads, heads=heads
                ),
            )

            for i, layer_config in enumerate(self.layers_config[1:-1], start=1):
                self.conv_layers.add_module(
                    f"conv{i}",
                    GATConv(
                        self.layers_config[i]["channels"] * heads,
                        layer_config["channels"] * heads,
                        heads=heads,
                    ),
                )

            self.conv_layers.add_module(
                f"conv{len(self.layers_config)}",
                GATConv(
                    self.layers_config[-2]["channels"] * heads,
                    self.layers_config[-1][
                        "channels"
                    ],  # The final layer averages the heads
                    heads=1,
                ),
            )

    def forward(self, x, edges_config):
        for i, layer in enumerate(self.conv_layers):
            edge_index = edges_config[self.layers_config[i]["edge_index"]]
            x = layer(x, edge_index)
        return x


class GraphSage(Module):
    """
    GraphSAGE (Graph Sample and Aggregation) implementation.

    Args:
        num_features (int): Number of input features.
        layers_config (list): Configuration of network layers.
        activation (str, optional): Activation function. Default is "ReLU".
        aggregator (str, optional): Type of aggregator function. Default is "mean".
    """

    def __init__(
        self,
        num_features,
        layers_config: list,
        activation: str = "ReLU",
        aggregator: str = "mean",
    ):
        super(GraphSage, self).__init__()
        self.layers_config = interpret_layers_config(layers_config)

        self.conv_layers = Sequential()
        self.conv_layers.add_module(
            "conv0",
            SAGEConv(num_features, self.layers_config[0]["channels"], aggregator),
        )

        for i, layer_config in enumerate(self.layers_config[1:], start=1):
            self.conv_layers.add_module(f"act{i}", getattr(torch.nn, activation)())
            self.conv_layers.add_module(
                f"conv{i}",
                SAGEConv(
                    self.layers_config[i - 1]["channels"],
                    layer_config["channels"],
                    aggregator,
                ),
            )

    def forward(self, x, edges_config):
        n_conv = 0
        for layer in self.conv_layers:
            if isinstance(layer, SAGEConv):
                edge_index = edges_config[self.layers_config[n_conv]["edge_index"]]
                x = layer(x, edge_index)
                n_conv += 1
            else:
                x = layer(x)
        return x


class InnerProductEdgeDecoder(Module):
    """
    Inner Product Edge Decoder for graph neural networks.
    Decodes edge representations through inner product operations.
    """

    def forward(self, z, edge_index):
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        return torch.sum(z_i * z_j, dim=-1)


class MLPNodeDecoder(Module):
    """
    Multi-layer Perceptron (MLP) based Node Decoder for graph neural networks.

    Args:
        hidden_channels (list): List of hidden channel sizes for the MLP layers.
    """

    def __init__(self, hidden_channels: list):
        super(MLPNodeDecoder, self).__init__()
        self.decoder = Sequential()
        self.decoder.add_module(
            "first layer", Linear(hidden_channels[0], hidden_channels[1])
        )
        for i in range(1, len(hidden_channels) - 1):
            self.decoder.add_module(f"act{i}", ReLU())
            self.decoder.add_module(
                f"linear{i}", Linear(hidden_channels[i], hidden_channels[i + 1])
            )

    def forward(self, z):
        return self.decoder(z)
