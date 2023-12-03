import torch
import numpy as np
from itertools import product
import random
from tqdm import tqdm
from torch_geometric.utils import from_networkx, to_networkx
import wandb
import faiss
from scipy.sparse import lil_matrix
import gc

ALPHABET = ["A", "C", "G", "T"]


def compute_dirichlet_energy(graph, embeddings):
    """
    Computes the Dirichlet energy for a given graph and embeddings.

    Args:
        graph (dict): A dictionary containing the graph's structure and edge weights.
        embeddings (Tensor): Node embeddings of the graph.

    Returns:
        dict: A dictionary with keys indicating the edge type and values representing the computed Dirichlet energy.
    """
    dirichlet_energy = {}

    for edges in graph.items():
        if edges[0].startswith("edge_index"):
            if edges[0] == "edge_index":
                edge_index = graph["edge_index"]
                edge_weight = graph["edge_weight"]
                edge_type = "default"
            else:
                edge_type = edges[0].split("_")[2]
                edge_index = graph["edge_index_" + edge_type]
                edge_weight = graph["edge_weight_" + edge_type]

            row, col = edge_index
            edge_feature_diff = embeddings[row] - embeddings[col]

            if edge_weight is not None:
                # Incorporate edge weights if provided
                edge_feature_diff *= edge_weight.view(-1, 1)

            num_edges = edge_index.size(1)
            dirichlet_energy["dirichlet_energy_" + edge_type] = (
                ((edge_feature_diff.norm(dim=1) ** 2).sum().item() / num_edges)
                if num_edges > 0
                else 0
            )

    wandb.log(dirichlet_energy)
    print("dirichlet_energy: ", dirichlet_energy)

    return dirichlet_energy


def setup_faiss_index(
    X,
    index_type="IVFFlat",
    distance="L2",
    nlist=None,
    nprobe=None,
    m=None,
    nbits=None,
):
    """
    Set up a FAISS index with configurable types, distance metrics, and ANN search options.

    Parameters:
    - X (np.array): Dataset of vectors to index.
    - index_type (str): Type of index, can be "IVFFlat", "IVFPQ", or "IVFSpectralHash".
    - distance (str): Distance metric, can be "L2" or "IP".
    - nlist (int): Number of clusters for quantization (for ANN search).
    - nprobe (int): Number of clusters to search (for ANN search, only for IVFFlat).
    - m (int): Number of centroid subvectors for IVFPQ.
    - nbits (int): Number of bits per component for IVFSpectralHash.

    Returns:
    - index (faiss.Index): Configured FAISS index.
    """
    d = X.shape[1]  # Vector dimensionality
    metric = faiss.METRIC_L2 if distance == "L2" else faiss.METRIC_INNER_PRODUCT

    # Set nlist to a default value if not provided
    if nlist is None:
        nlist = int(np.sqrt(X.shape[0]))
    if nprobe is None:
        nprobe = int(np.sqrt(nlist))
    if nbits is None:
        nbits = 8

    quantizer = faiss.IndexFlatL2(d) if distance == "L2" else faiss.IndexFlatIP(d)

    # Select the index type
    if index_type == "IVFFlat":
        print("Using IVFFlat index.")
        index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
        index.nprobe = nprobe
    elif index_type == "IVFPQ":
        print("Using IVFPQ index.")
        if m is None:
            m = min(d // 2, 32)  # A simple heuristic, with a max of 64
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    elif index_type == "IVFSpectralHash":
        print("Using IVFSpectralHash index.")
        index = faiss.IndexIVFSpectralHash(quantizer, d, nlist, nbits, 1.0)

    if distance == "IP":
        faiss.normalize_L2(X)
    # Train the index if necessary
    if not index.is_trained:
        index.train(X)

    # Add vectors to the index
    index.add(X)

    return index


def kf_faiss_to_edge_index_weight(
    X,
    k,
    index_type="IVFFlat",
    distance="L2",
    nlist=None,
    nprobe=None,
    m=None,
    nbits=None,
    useFloat16=True,
):
    """
    Perform k-nearest neighbors (kNN) search using FAISS and return the edge index and corresponding weights.

    This function initializes a FAISS index based on the specified parameters, optionally moves the index to GPU
    for acceleration, and performs a kNN search on the input dataset X. The function handles both L2 and inner
    product (IP) distances, and post-processes the results to convert distances to similarity scores if needed.
    Self-edges are removed, and zero similarity edges are filtered out.

    Parameters:
        X (np.ndarray): A 2D numpy array representing the dataset on which kNN search is performed.
        k (int): The number of nearest neighbors to search for each vector in X.
        index_type (str, optional): The type of FAISS index to use. Defaults to 'IVFFlat'.
        distance (str, optional): The distance metric to use ('L2' for Euclidean or 'IP' for inner product). Defaults to 'L2'.
        nlist (int, optional): The number of cells (for IVF index). Required for some index types.
        nprobe (int, optional): The number of cells to visit during search (for IVF index).
        m (int, optional): The number of bytes per vector (for PQ index).
        nbits (int, optional): The number of bits per sub-vector (for PQ index).
        useFloat16 (bool, optional): Whether to use Float16 in GPU computation. Defaults to True.

    Returns:
        tuple: A tuple containing two elements:
            - edge_index (np.ndarray): A 2D array of shape (2, num_edges) representing the edge indices in the format [source, target].
            - edge_weight (np.ndarray): A 1D array representing the weight (or similarity) of each edge.

    Notes:
        - If 'IP' is chosen as the distance, vectors in X are normalized before the search.
        - For 'L2' distance, distances are converted to similarity scores.
        - The function automatically detects GPU availability and uses it for acceleration if possible.
        - Self-edges and zero similarity edges are removed from the final results.
    """
    print(
        f"Starting kNN search with {X.shape[0]} vectors of dimension {X.shape[1]} using {distance} distance."
    )

    # Normalize the vectors if using inner product (cosine similarity)

    X = X.astype(np.float32)
    if distance == "IP":
        print("Normalizing vectors for inner product.")
        faiss.normalize_L2(X)

    # Create the FAISS index
    index = setup_faiss_index(
        X,
        index_type,
        distance,
        nlist,
        nprobe,
        m,
        nbits,
    )

    # Check if a GPU is available and move the index to GPU
    ngpus = faiss.get_num_gpus()
    # disable GPU for SpectralHash
    if index_type == "IVFSpectralHash":
        ngpus = 0
    if ngpus > 0:
        print(f"Detected {ngpus} GPUs. Moving index to GPU.")
        gpu_resources = faiss.StandardGpuResources()
        if useFloat16:
            co = faiss.GpuClonerOptions()
            co.useFloat16LookupTables = True
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index, co)
        else:
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

    # Perform the search
    print(f"Performing the search for the {k} nearest neighbors.")
    D, I = index.search(
        X, k + 1
    )  # k+1 because the search includes the query vector itself

    if ngpus > 0:
        print("Moving index back to CPU.")
        index = faiss.index_gpu_to_cpu(index)

    print("Search completed, post-processing the results.")
    # Remove self-edges and reshape the results into two arrays
    mask = np.ones(I.shape, dtype=bool)
    mask[:, 0] = False
    I = I[mask].reshape(X.shape[0], k)
    D = D[mask].reshape(X.shape[0], k)

    # Normalize distances if using L2 distance and convert to similarity
    if distance == "L2":
        max_distance = np.max(D)
        D = 1 - (D / max_distance)
    D = D.clip(max=1.0)

    # Filter out zero similarity (or maximum distance) edges
    print("Filtering out zero similarity edges.")
    nonzero_mask = D.flatten() > 0.0001
    source = np.repeat(np.arange(X.shape[0]), k)[nonzero_mask]
    target = I.flatten()[nonzero_mask]
    edge_weight = D.flatten()[nonzero_mask]

    # Create edge_index array with only nonzero edges
    edge_index = np.stack([source, target], axis=0)
    print("Faiss Edge index shape: ", edge_index.shape)

    del index
    gc.collect()
    print("kNN search completed, returning results.")
    return edge_index, edge_weight


def cosine_similarity_to_edge_index_weight(
    X, chunk_size=1024, threshold=0.0, keep_top_k=None
):
    """
    Computes cosine similarity for the rows of a matrix X and converts it into edge indices and weights.

    Args:
        X (np.array): Input array of shape (N, D).
        chunk_size (int, optional): Size of chunks for computation. Default is 1024.
        threshold (float, optional): Threshold for considering an edge based on similarity. Default is 0.0.
        keep_top_k (float, optional): Proportion of top edges to keep based on weight.

    Returns:
        tuple: Edge indices and associated weights.
    """
    N = X.shape[0]
    num_chunks = (
        N + chunk_size - 1
    ) // chunk_size  # Ceiling division to include the last smaller chunk

    source, target, edge_weight = [], [], []

    for i in tqdm(range(num_chunks), desc="Computing cosine similarity..."):
        start_i = i * chunk_size
        end_i = min(start_i + chunk_size, N)

        X_chunk = X[start_i:end_i]

        for j in range(i, num_chunks):
            start_j = j * chunk_size
            end_j = min(start_j + chunk_size, N)

            X_chunk2 = X[start_j:end_j]

            chunk_sim = X_chunk @ X_chunk2.T

            norm_chunk1 = np.linalg.norm(X_chunk, axis=1).reshape(-1, 1)
            norm_chunk2 = np.linalg.norm(X_chunk2, axis=1).reshape(1, -1)

            chunk_sim = chunk_sim / (norm_chunk1 * norm_chunk2)

            # Replace NaNs with zero
            np.nan_to_num(chunk_sim, copy=False)

            # Save only the upper triangular part if it's not a diagonal block
            # Set diagonal elements to zero if it's a diagonal block
            if i == j:
                chunk_sim = np.triu(chunk_sim, k=1)
            else:
                chunk_sim = np.triu(chunk_sim, k=-X_chunk.shape[0])

            chunk_sim = np.triu(chunk_sim, k=-X_chunk.shape[0])

            # Find the indices of the elements that meet the threshold
            s, t = np.nonzero(chunk_sim > threshold)

            s = s + start_i  # Adjust the indices based on the chunk's position
            t = t + start_j

            source.append(s)
            target.append(t)
            edge_weight.append(chunk_sim[s - start_i, t - start_j])

    # Concatenate all arrays
    source = np.concatenate(source)
    target = np.concatenate(target)
    edge_weight = np.concatenate(edge_weight)

    # Create edge_index array
    edge_index = np.stack([source, target], axis=0)

    if keep_top_k is not None:
        keep_top_n = min(int(N**2 * keep_top_k), edge_weight.shape[0])
        indices = np.argpartition(edge_weight, -keep_top_n)[-keep_top_n:]

        source = source[indices]
        target = target[indices]
        edge_weight = edge_weight[indices]

    edge_index = np.stack([source, target], axis=0).astype(np.int64)
    print("Faiss Edge index shape: ", edge_index.shape)

    return edge_index, edge_weight


def levenshtein(s1, s2):
    """
    Calculates the Levenshtein distance between two strings.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def edit_distance_to_edge_index_weight(
    k, chunk_size=1024, threshold=0.0, keep_top_k=None
):
    """
    Computes edit distance for k-mers and converts it into edge indices and weights.

    Args:
        k (int): Length of k-mers.
        chunk_size (int, optional): Size of chunks for computation. Default is 1024.
        threshold (float, optional): Threshold for considering an edge based on distance. Default is 0.0.
        keep_top_k (float, optional): Proportion of top edges to keep based on weight.

    Returns:
        tuple: Edge indices and associated weights.
    """
    all_kmers = ["".join(i) for i in product(ALPHABET, repeat=k)]
    num_kmers = len(all_kmers)
    num_chunks = (num_kmers + chunk_size - 1) // chunk_size

    source, target, edge_weight = [], [], []

    for i in tqdm(range(num_chunks), desc="Computing edit distance"):
        start_i = i * chunk_size
        end_i = min(start_i + chunk_size, num_kmers)

        for j in range(i, num_chunks):
            start_j = j * chunk_size
            end_j = min(start_j + chunk_size, num_kmers)

            chunk = np.ones((end_i - start_i, end_j - start_j)) * k

            for ii, kmer1 in enumerate(all_kmers[start_i:end_i]):
                for jj, kmer2 in enumerate(all_kmers[start_j:end_j]):
                    if ii >= jj:
                        continue
                    distance = levenshtein(kmer1, kmer2)
                    chunk[ii, jj] = distance

            # Normalize chunk and invert it (so that higher distance means lower similarity)
            chunk = 1 - chunk / k

            # Find the indices of the elements that meet the threshold
            s, t = np.nonzero(chunk > threshold)

            s = s + start_i  # Adjust the indices based on the chunk's position
            t = t + start_j

            source.append(s)
            target.append(t)

            edge_weight.append(chunk[s - start_i, t - start_j])

    # Concatenate all arrays
    source = np.concatenate(source)
    target = np.concatenate(target)
    edge_weight = np.concatenate(edge_weight)

    # Create edge_index array
    edge_index = np.stack([source, target], axis=0)

    if keep_top_k is not None:
        keep_top_n = min(int(num_kmers**2 * keep_top_k), edge_weight.shape[0])
        indices = np.argpartition(edge_weight, -keep_top_n)[-keep_top_n:]
        source = source[indices]
        target = target[indices]
        edge_weight = edge_weight[indices]

    edge_index = np.stack([source, target], axis=0).astype(np.int64)

    return edge_index, edge_weight


def node_embedding_initial(
    graph, method: str = "onehot", k: int = 3, small_k: int = None, random_size: int = 4
):
    """
    Initializes node embeddings based on the specified method.

    Args:
        graph (Graph): Input graph.
        method (str, optional): Method to initialize embeddings. Options are ["onehot", "random", "kmer_frequency"]. Default is "onehot".
        k (int, optional): Length for k-mers. Default is 3.
        small_k (int, optional): Sub k for kmer_frequency method.
        random_size (int, optional): Size for random initialization. Default is 4.

    Returns:
        list: List of node embeddings.
    """
    graph_kmers = list(graph.nodes())  # Get the k-mers that are already in the graph

    if method == "onehot":
        index_dict = {kmer: idx for idx, kmer in enumerate(graph_kmers)}
        w = np.eye(len(graph_kmers))
        labels = [w[index_dict[label]] for label in graph_kmers]

    elif method == "random":
        labels = np.random.rand(len(graph_kmers), random_size)

    elif method == "kmer_frequency":
        if small_k is None:
            raise ValueError("small_k must be provided for kmer_frequency method")

        subkmers = ["".join(i) for i in product(ALPHABET, repeat=small_k)]
        subkmer_to_index = {subkmer: i for i, subkmer in enumerate(subkmers)}
        w = lil_matrix((len(graph_kmers), len(subkmers)), dtype=int)

        for i, kmer in enumerate(
            tqdm(
                graph_kmers,
                desc="Computing Occurances of Sub-k-mers...",
                mininterval=30,
            )
        ):
            for j in range(len(kmer) - small_k + 1):
                sub_kmer = kmer[j : j + small_k]
                w[i, subkmer_to_index[sub_kmer]] += 1

        # Convert to CSR format and sum across rows to get total counts for each sub-k-mer
        w_csr = w.tocsr()
        sub_kmer_counts = w_csr.sum(axis=0)

        # Find indices of sub-k-mers that occur at least once
        nonzero_indices = np.where(sub_kmer_counts > 0)[1]

        # Filter the columns of the matrix to include only sub-k-mers that occur
        w_filtered = w_csr[:, nonzero_indices]

        # Normalize the frequency vectors
        normalization_factor = len(kmer) - small_k + 1
        w_normalized = w_filtered / normalization_factor

        # Convert the filtered and normalized matrix to a dense NumPy array
        return w_normalized.toarray()

    else:
        raise ValueError("Invalid method for node embedding initialisation")

    return labels


def attach_higher_strides_to_torch_graph(new_graph, torch_graph, stride_number):
    """
    Attaches higher stride edge indices and weights to a torch graph.

    Args:
        new_graph (Graph): Graph containing higher stride information.
        torch_graph (Data): PyTorch Geometric graph data.
        stride_number (int): Stride number for the new data.

    Returns:
        Data: Updated PyTorch Geometric graph data.
    """
    data = from_networkx(new_graph)
    torch_graph["edge_index_" + str(stride_number)] = data.edge_index
    torch_graph["edge_weight_" + str(stride_number)] = data.weight
    return torch_graph


def sample_biased_random_walks(
    data,
    num_nodes_to_sample: int,
    num_walks: int = 5,
    walk_length: int = 2,
    window_size: int = 3,
    p: float = 1.0,
    q: float = 1.0,
):
    """
    Generates biased random walks from a graph.

    Args:
        data (Data): PyTorch Geometric graph data.
        num_nodes_to_sample (int): Number of nodes to sample for the random walks.
        num_walks (int, optional): Number of walks per node. Default is 5.
        walk_length (int, optional): Length of each walk. Default is 2.
        window_size (int, optional): Window size for positive pair generation. Default is 3.
        p (float, optional): Return hyperparameter for biased random walks. Default is 1.0.
        q (float, optional): In-out hyperparameter for biased random walks. Default is 1.0.

    Returns:
        Tensor: Tensor containing positive node pairs.
    """
    G = to_networkx(data, edge_attrs=["weight"], to_undirected=False)

    # Sample nodes
    all_nodes = list(G.nodes)
    if num_nodes_to_sample < len(all_nodes):
        nodes_to_walk = random.sample(all_nodes, num_nodes_to_sample)
    else:
        nodes_to_walk = all_nodes
    random.shuffle(nodes_to_walk)

    positive_pairs = []
    for _ in range(num_walks):
        for node in nodes_to_walk:
            walk = [node]
            previous_node = None
            for _ in range(walk_length):
                current_node = walk[-1]
                neighbors = list(G.neighbors(current_node))
                weights = [
                    G[current_node][neighbor]["weight"] for neighbor in neighbors
                ]

                if len(neighbors) > 0:
                    if previous_node is None:
                        next_node = random.choices(neighbors, weights=weights, k=1)[0]
                    else:
                        # Otherwise, perform a biased random walk step
                        transition_probabilities = []
                        for neighbor in neighbors:
                            if neighbor == previous_node:
                                transition_probabilities.append(
                                    weights[neighbors.index(neighbor)] / p
                                )
                            elif G.has_edge(neighbor, previous_node):
                                transition_probabilities.append(
                                    weights[neighbors.index(neighbor)]
                                )
                            else:
                                transition_probabilities.append(
                                    weights[neighbors.index(neighbor)] / q
                                )
                        # Normalize the probabilities
                        transition_probabilities = [
                            prob / sum(transition_probabilities)
                            for prob in transition_probabilities
                        ]
                        next_node = random.choices(
                            neighbors, weights=transition_probabilities, k=1
                        )[0]

                    walk.append(next_node)
                    previous_node = current_node
                else:
                    break

            # Create positive pairs
            for i in range(len(walk)):
                shrink_window_size = random.randint(1, window_size)
                for j in range(i - shrink_window_size, i + shrink_window_size + 1):
                    if j >= 0 and j < len(walk) and j != i:
                        positive_pairs.append((walk[i], walk[j]))

    return torch.tensor(positive_pairs, dtype=torch.long).t()
