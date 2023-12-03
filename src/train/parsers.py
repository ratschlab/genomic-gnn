import argparse

##########################
#### TASK SPECIFIC PARSERS
##########################


def edit_distance_parser():
    """
    Creates an argument parser for configuration related to Edit Distance Approximation and Closest String Retrieval Tasks.

    Arguments:
    - None

    Returns:
    - argparse.ArgumentParser: An argument parser object populated with edit distance-related arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="datasets/edit_distance/edit_qiita_large.pkl",
        help="Edit Distance Approximation Dataset path",
    )
    parser.add_argument(
        "--retrieval_data",
        type=str,
        default=None,
        help="Retrieval Dataset path",
    )
    parser.add_argument(
        "--scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to scale edit distance by learned parameter",
    )
    parser.add_argument(
        "--zero_shot_retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Instead of training use combination of computed embeddings to retrive closest sequences",
    )
    parser.add_argument(
        "--zero_shot_method",
        type=str,
        default="concat,sum,mean",
        help="Method for zero shot retrieval, to choose from: mean, sum, concat, if multiple, separate by comma",
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        default="hyperbolic",
        help="Distance type for retrieval, edit distance, hierarchical, to choose from: hyperbolic, manhattan, cosine, euclidean, square",
    )

    parser = general(parser)
    parser = representations_general_parameters(parser)
    parser = training_parameters(parser)
    parser = mlp_model(parser)
    parser = cnn_model(parser)
    parser = word2vec_args(parser)
    parser = node2vec_args(parser)
    parser = ssgnn_args(parser)

    return parser


def coding_metagenomics_parser():
    """
    Creates an argument parser for configuration related to Gene Prediction Task.

    Arguments:
    - None

    Returns:
    - argparse.ArgumentParser: An argument parser object populated with edit distance-related arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="datasets/geneRFinder",
        help="Datasets folder path",
    )
    parser.add_argument(
        "--include_training_1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include training 1 in the test set",
    )
    parser.add_argument(
        "--data_max_sequence_length",
        type=int,
        default=600,
        help="Datasets folder path",
    )
    parser.add_argument(
        "--only_CAMI_dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include only CAMI dataset",
    )
    parser.add_argument(
        "--proportion_samples_to_keep_supervised_training",
        type=float,
        default=1,
        help="If 1, keeps entire dataset for supervised training, if 0.1, keeps 0.1 of data for supervised training",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="Dropout value for dense part fo the network",
    )
    parser = general(parser)
    parser = representations_general_parameters(parser)
    parser = training_parameters(parser)
    parser = mlp_model(parser)
    parser = cnn_model(parser)
    parser = word2vec_args(parser)
    parser = node2vec_args(parser)
    parser = ssgnn_args(parser)

    return parser


##########################
###### SHARED AMONG TASK
##########################


def general(parser):
    parser.add_argument(
        "--seed",
        type=str,
        default="2137",
        help="set random seed values for which to train",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="./wandb",
        help="set directory for offline wandb logs",
    )
    parser.add_argument(
        "--disable_tqdm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to disable tqdm progress bar",
    )
    return parser


def representations_general_parameters(parser):
    parser.add_argument(
        "--include_val_test_unsupervised",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include validation and test sets in dataset for unsupervised representation learning",
    )
    parser.add_argument(
        "--representation_method",
        type=str,
        default="kmer_node2vec",
        help="Name of Representation methods",
    )
    parser.add_argument(
        "--random_representation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Works only when method is set to onehot - replaces one hot with random vectors",
    )
    parser.add_argument(
        "--representation_k",
        type=str,
        default="3",
        help="K-mer size for representation",
    )
    parser.add_argument(
        "--representation_stride",
        type=str,
        default="1",
        help="Stride for representation, support multiples, separated by commas",
    )
    parser.add_argument(
        "--inference_stride", type=int, default=1, help="Stride for inference"
    )
    return parser


# downstream task training
def training_parameters(parser):
    parser.add_argument(
        "--model_class",
        type=str,
        default="mlp",
        help="Either mlp or cnn1d for downstream task",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="accelerator name for pytorch lightning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs for pytorch model training for downstream task",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch Size for pytorch model training for downstream task",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for pytorch model training for downstream task",
    )

    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=10,
        help="How often evaluate for pytorch model training for downstream task",
    )

    parser.add_argument(
        "--trainable_embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="By default embedding layer is frozen, if set to true, also include embedding in backprop",
    )

    return parser


# downstream task model
def cnn_model(parser):
    parser.add_argument(
        "--channels",
        type=str,
        default="128,256,256,128",
        help="Following channel sizes for each k branch, separated by commas",
    )
    parser.add_argument(
        "--batch_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If batch normalisation should be applied",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="avg",
        help="specify one of pooling: avg, max or None",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="specify kernel size, if none the sama as representation size",
    )

    return parser


# downstream task model
def mlp_model(parser):
    parser.add_argument(
        "--network_layers_before_cat",
        type=str,
        default="",
        help="Layers sizes for seperate #k paths in network",
    )
    parser.add_argument(
        "--network_layers_after_cat",
        type=str,
        default="128",
        help="Layers sizes after concatanated #k paths are concatanated",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        help="Torch activation functions after linear layers, apart from the last one",
    )
    return parser


# kmer representation model
def word2vec_args(parser):
    parser.add_argument(
        "--representation_size",
        type=int,
        default=32,
        help="Embedding Size for Word2vec and Node2Vec",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Window size, only for Word2vec and Node2Vec",
    )
    return parser


# kmer representation model
def node2vec_args(parser):
    parser.add_argument(
        "--walk_len",
        type=int,
        default=20,
        help="Length of a single walk, only for Node2Vec",
    )
    parser.add_argument(
        "--num_walks",
        type=int,
        default=200,
        help="Number of Walks, only for Node2Vec",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=1,
        help="the probability of a random walk getting back to the previous node, only for Node2Vec",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=1,
        help="probability that a random walk can pass through a previously unseen part of the graph, only for Node2Vec",
    )
    return parser


# kmer representation model
def ssgnn_args(parser):
    parser.add_argument(
        "--representation_small_k",
    )
    parser.add_argument(
        "--representation_ss_task",
        type=str,
        default="CL",
        help="Name for Self supervised task for representation learning, to choose from: Contrastive Learning (CL), Autoencoder (AE)",
    )
    parser.add_argument(
        "--representation_ss_edge_weight_loss_weight",
        type=float,
        default=1.0,
        help="Weight for edge weight related loss",
    )
    parser.add_argument(
        "--representation_ss_kmer_frequency_loss_weight",
        type=float,
        default=1.0,
        help="Weight for kmer frequency related loss",
    )
    parser.add_argument(
        "--representation_ss_edit_distance_loss_weight",
        type=float,
        default=0.0,
        help="Weight for edit distance related loss",
    )
    parser.add_argument(
        "--representation_ss_negative_sampling_loss_weight",
        type=float,
        default=1.0,
        help="Weight for negative sampling loss",
    )
    parser.add_argument(
        "--representation_ss_encoder",
        type=str,
        default="GCN",
        help="Name for encoder for representation learning, to choose from: GCN, MLP, GAT, RGCN",
    )
    parser.add_argument(
        "--representation_ss_hidden_channels",
        type=str,
        default="",
        help="Hidden channels for encoder for representation learning, the formating is: 128_ED,128_DB,128_KF0, where the first number is number of channels, and the second is the edges type",
    )
    parser.add_argument(
        "--representation_ss_last_layer_edge_type",
        type=str,
        default="DB",
        help="Last layer edge type for representation learning, to choose from: ED (edit distance), DB, KF0, KF1, ... (Kmer Frequency)",
    )
    parser.add_argument(
        "--representation_ss_activation",
        type=str,
        default="ReLU",
        help="Activation for representation learning, to choose from: ReLU, LeakyReLU, ELU, Tanh, Sigmoid",
    )
    parser.add_argument(
        "--representation_ss_lr",
        type=float,
        default=1e-3,
        help="Learning rate for self supervised representation learning",
    )
    parser.add_argument(
        "--representation_ss_epochs",
        type=int,
        default=10000,
        help="Number of epochs for self supervised representation learning",
    )
    parser.add_argument(
        "--representation_ss_probability_masking_edges",
        type=float,
        default=0.0,
        help="Probability of masking an edge during self supervised autoencoder training",
    )
    parser.add_argument(
        "--representation_ss_probability_masking_nodes",
        type=float,
        default=0.0,
        help="Probability of masking a node during self supervised autoencoder training",
    )
    parser.add_argument(
        "--representation_ss_sampling_walk_length",
        type=int,
        default=5,
        help="Walk length for random walk sampling",
    )
    parser.add_argument(
        "--representation_ss_sampling_num_walks",
        type=int,
        default=20,
        help="Number of walks for random walk sampling",
    )
    parser.add_argument(
        "--representation_ss_sampling_window_size",
        type=int,
        default=3,
        help="Window size for random walk sampling",
    )
    parser.add_argument(
        "--representation_ss_sampling_proportion_nodes_to_sample",
        type=float,
        default=1,
        help="Proportion of nodes to sample for random walk sampling",
    )
    parser.add_argument(
        "--representation_ss_resample_every_num_epochs",
        type=int,
        default=-1,
        help="Resample every num epochs for random walk sampling",
    )
    parser.add_argument(
        "--representation_ss_proportion_negative_to_positive_samples",
        type=float,
        default=1,
        help="Proportion of negative to positive samples for random walk sampling",
    )
    parser.add_argument(
        "--representation_ss_rw_p",
        type=float,
        default=1,
        help="p for random walk sampling",
    )
    parser.add_argument(
        "--representation_ss_rw_q",
        type=float,
        default=1,
        help="q for random walk sampling",
    )
    parser.add_argument(
        "--representation_ss_edges_threshold",
        type=float,
        default=0.0,
        help="Threshold for similarity edges that are kept, helpful for reducing memory usage",
    )
    parser.add_argument(
        "--representation_ss_edges_keep_top_k",
        type=float,
        default=None,
        help="Keep top k most similar edges, helpful for reducing memory usage",
    )
    parser.add_argument(
        "--representation_ss_initial_labels",
        type=str,
        default="kmer_frequency",
        help="Type of labels to be initialised as nodes, with small_k helpfull for reducing memory usage",
    )
    parser.add_argument(
        "--representation_ss_batch_size",
        type=int,
        default=64,
        help="Batch size for self supervised representation learning",
    )
    parser.add_argument(
        "--representation_ss_create_all_kmers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to create all kmers for representation learning",
    )
    parser.add_argument(
        "--representation_ss_normalise_embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to normalise embeddings for representation learning",
    )
    parser.add_argument(
        "--representation_ss_faiss_ann",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use faiss for approximate nearest neighbours for representation learning",
    )
    parser.add_argument(
        "--representation_ss_faiss_index_type",
        type=str,
        default="IVFFlat",
        help="Faiss index type for representation learning",
    )
    parser.add_argument(
        "--representation_ss_faiss_distance",
        type=str,
        default="L2",
        help="Faiss distance for representation learning",
    )
    parser.add_argument(
        "--representation_ss_faiss_nlist",
        type=int,
        default=None,
        help="Faiss nlist for representation learning",
    )
    parser.add_argument(
        "--representation_ss_faiss_nprobe",
        type=int,
        default=None,
        help="Faiss nprobe for representation learning",
    )
    parser.add_argument(
        "--representation_ss_faiss_m",
        type=int,
        default=None,
        help="Faiss m for representation learning",
    )
    parser.add_argument(
        "--representation_ss_faiss_nbits",
        type=int,
        default=None,
        help="Faiss nbits for representation learning",
    )

    return parser


def assert_arguments_correctness(args):
    """
    Verifies the correctness and consistency of given arguments.

    Parameters:
    - args (object): An object, typically an instance of `argparse.Namespace`
    , containing the command-line arguments or configurations to be checked.

    Raises:
    - AssertionError: If any of the conditions for argument correctness are not met.
    """
    if hasattr(args, "representation_ss_faiss_ann"):
        if args.representation_ss_faiss_ann:
            assert (
                args.representation_ss_task == "CL"
            ), "Faiss ANN only supported for CL task."
            assert (
                args.representation_ss_edges_keep_top_k
            ), "Faiss ANN only supported with keep_top_k."
    if hasattr(args, "model_class"):
        if args.network_layers_before_cat == "" and args.model_class == "mlp":
            assert len(args.representation_k.split(",")) == 1

    if hasattr(args, "representation_k"):
        if (
            args.representation_method != "kmer_ssgnn"
            or args.representation_method != "kmer_ssgnn_miniBatch"
        ):
            assert (
                len(args.representation_stride.split(",")) == 1
            ), "Multiple strides only supported for ssgnn."

    if hasattr(args, "representation_ss_hidden_channels"):
        if args.representation_ss_hidden_channels != "":
            if len(args.representation_ss_hidden_channels[0].split("_")) > 1:
                assert (
                    max(
                        [
                            int(x.split("_")[1][-1])
                            for x in args.representation_ss_hidden_channels.split(",")
                            if x.split("_")[1][0:2] == "KF"
                        ]
                    )
                    + 1
                ) <= len(
                    args.representation_small_k.split(",")
                ), "Number of hidden channels for kmer frequency must be equal or smaller to number of small_k values."


def convert_string_args_to_list(args):
    """
    Converts specific string attributes of the given argument object into lists.

    The function inspects certain attributes of the `args` object which are expected
    to be comma-separated strings. These attributes, if present and non-empty, are
    split and converted to lists.

    Parameters:
    - args (object): An object, typically an instance of `argparse.Namespace`,
      containing attributes whose string values need to be converted into lists.

    Returns:
    - object: The modified `args` object with the relevant attributes converted
      from strings to lists.
    """
    if hasattr(args, "channels"):
        if args.channels == "":
            args.channels = []
        else:
            args.channels = [int(x) for x in args.channels.split(",")]
    if hasattr(args, "network_layers_before_cat"):
        if args.network_layers_before_cat == "":
            args.network_layers_before_cat = []
        else:
            args.network_layers_before_cat = [
                int(x) for x in args.network_layers_before_cat.split(",")
            ]
    if hasattr(args, "network_layers_after_cat"):
        if args.network_layers_after_cat == "":
            args.network_layers_after_cat = []
        else:
            args.network_layers_after_cat = [
                int(x) for x in args.network_layers_after_cat.split(",")
            ]
    if hasattr(args, "representation_ss_hidden_channels"):
        if args.representation_ss_hidden_channels == "":
            args.representation_ss_hidden_channels = []
        else:
            args.representation_ss_hidden_channels = [
                str(x) for x in args.representation_ss_hidden_channels.split(",")
            ]
    if hasattr(args, "representation_stride"):
        if args.representation_stride == "":
            args.representation_stride = []
        else:
            args.representation_stride = [
                int(x) for x in args.representation_stride.split(",")
            ]

    return args
