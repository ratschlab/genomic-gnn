from src.representations.kmer_onehot import kmer_onehot_fit, kmer_onehot_transform
from src.representations.kmer_word2vec import kmer_word2vec_fit, kmer_word2vec_transform
from src.representations.kmer_node2vec import kmer_node2vec_fit, kmer_node2vec_transform
from src.representations.kmer_ssgnn import KmerSsgnn
from src.representations.kmer_ssgnn_miniBatch import KmerSsgnnMb


class KmerOnehotWrapper:
    def __init__(self):
        self.fit = kmer_onehot_fit
        self.transform = kmer_onehot_transform


class KmerWord2vecWrapper:
    def __init__(self):
        self.fit = kmer_word2vec_fit
        self.transform = kmer_word2vec_transform


class KmerNode2vecWrapper:
    def __init__(self):
        self.fit = kmer_node2vec_fit
        self.transform = kmer_node2vec_transform


# The dictionary now points to classes
AVAILABLE_REPRESENTATIONS = {
    "kmer_onehot": KmerOnehotWrapper,
    "kmer_word2vec": KmerWord2vecWrapper,
    "kmer_node2vec": KmerNode2vecWrapper,
    "kmer_ssgnn": KmerSsgnn,
    "kmer_ssgnn_miniBatch": KmerSsgnnMb,
}


class RepresentationFactory:
    """
    A factory class to generate desired k-mer representation methods.

    Attributes:
    - name: The name of the representation method.
    - class_: The actual class or wrapper corresponding to the chosen representation.
    - layers_list: A list of pytorch EMBEDDING layers generated during fitting, specific to some methods.
    - index_dict_list: List of dictionaries mapping k-mers to indices, specific to some methods.
    - instances: Dictionary holding instances for each k in k-mer.
    """

    def __init__(self, method: str = "kmer_node2vec"):
        """
        Initializes the RepresentationFactory with a specified method.

        Parameters:
        - method (str): The name of the desired k-mer representation method. Default is "kmer_node2vec".
        """
        assert method in AVAILABLE_REPRESENTATIONS

        self.name = method
        self.class_ = AVAILABLE_REPRESENTATIONS[self.name]

    def fit(self, dfs_list: list, args: dict):
        """
        Fits the specified representation on provided data.

        Parameters:
        - dfs_list (list): List of DataFrames containing sequences for which k-mer representations need to be computed.
        - args (dict): Dictionary of arguments required by the specific representation method.
        """
        self.layers_list = []
        self.index_dict_list = []
        self.instances = {}  # Dictionary to hold instances
        for k in args["representation_k"].split(","):
            self.instances[int(k)] = self.class_()  # Save instance to dictionary
            index_dict, layer = self.instances[int(k)].fit(
                dfs_list, k=int(k), args=args
            )
            self.layers_list.append(layer)
            self.index_dict_list.append(index_dict)

    def transform(self, dfs_list: list, args: dict):
        """
        Transforms the provided data using the previously fitted representation.

        Parameters:
        - dfs_list (list): List of DataFrames containing sequences to be transformed.
        - args (dict): Dictionary of arguments required by the specific representation method.

        Returns:
        - list: List of transformed DataFrames.
        """
        # if new layers list is created, then use it
        for i, k in enumerate(args["representation_k"].split(",")):
            self.instances[int(k)].transform(
                dfs_list, self.index_dict_list[i], k=int(k), args=args
            )
            if (
                self.name == "kmer_ssgnn" or self.name == "kmer_ssgnn_miniBatch"
            ) and hasattr(self.instances[int(k)], "new_layer"):
                self.layers_list[i] = self.instances[int(k)].new_layer

        return dfs_list

    def fit_transform(self, dfs_list: list, args: dict):
        """
        Fits the specified representation on provided data and then transforms the data.

        Parameters:
        - dfs_list (list): List of DataFrames containing sequences for which k-mer representations need to be computed and transformed.
        - args (dict): Dictionary of arguments required by the specific representation method.

        Returns:
        - list: List of transformed DataFrames.
        """
        self.fit(dfs_list, args=args)
        return self.transform(dfs_list, args=args)
