"""
Provides utilities for hyperbolic distances,
and their respective matrix forms to compute pairwise distances.

Functions:
----------
- hyperbolic_distance: Calculate the hyperbolic distance between two embeddings.
- hyperbolic_matrix: Calculate the pairwise hyperbolic distances between two sets of embeddings.
- hyperbolic_matrix_symmetric: Compute the pairwise hyperbolic distances within the same set of embeddings.

Constants:
----------
- DISTANCES_FACTORY: A dictionary mapping distance names to their respective function implementations.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def hyperbolic_distance(u, v, epsilon=1e-7):
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u**2, dim=-1)
    sqvnorm = torch.sum(v**2, dim=-1)

    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = torch.sqrt(x**2 - 1)

    return torch.log(x + z)


def hyperbolic_matrix_symmetric(enc_reference, batch_size=16):
    N, D = enc_reference.shape

    d = torch.zeros((N, N), device=enc_reference.device)

    for i in tqdm(range(0, N, batch_size), desc="Hyperbolic distance"):
        end_i = min(i + batch_size, N)
        enc_ref_batch_i = enc_reference[i:end_i].unsqueeze(1)
        for j in range(i, N, batch_size):
            end_j = min(j + batch_size, N)
            enc_ref_batch_j = enc_reference[j:end_j].unsqueeze(0)
            d[i:end_i, j:end_j] = hyperbolic_distance(enc_ref_batch_i, enc_ref_batch_j)
            if i != j:
                d[j:end_j, i:end_i] = d[i:end_i, j:end_j].T

    return d


def hyperbolic_matrix(enc_reference, enc_query):
    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    d = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        d[:, j] = hyperbolic_distance(enc_reference, enc_query[j : j + 1].repeat(N, 1))
    return d


DISTANCES_FACTORY = {
    "hyperbolic": hyperbolic_distance,
    "hyperbolic_matrix": hyperbolic_matrix,
    "hyperbolic_matrix_symmetric": hyperbolic_matrix_symmetric,
}
