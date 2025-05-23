import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def build_graph_from_edges(edges, directed=False):
    """
    Generate a NetworkX graph from a list of edges.
    
    Parameters:
    -----------
    edges : list of tuples
        Edge list where each element is (u, v) or (u, v, w) for weighted edges.
    directed : bool
        If True, create a directed graph (DiGraph); otherwise undirected (Graph).
    
    Returns:
    --------
    G : networkx.Graph or networkx.DiGraph
        The constructed graph with edges added.
    """
    # Initialize directed or undirected graph
    G = nx.DiGraph() if directed else nx.Graph()
    
    # Add weighted edges if weight is provided, else simple edges
    if edges and len(edges[0]) == 3:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
    
    return G


def laplacian_eigenvectors(G, k=2, normed=True):
    """
    Compute the first k non-trivial Laplacian eigenvectors for node embeddings.

    Parameters:
    ----------
    G : networkx.Graph
        Input graph (should be connected for meaningful embeddings).
    k : int
        Number of embedding dimensions (non-trivial eigenvectors) to return.
    normed : bool
        If True, compute the normalized Laplacian; otherwise unnormalized.

    Returns:
    -------
    eig_vals : ndarray, shape (k,)
        The k smallest non-zero eigenvalues of the Laplacian.
    eig_vecs : ndarray, shape (n_nodes, k)
        Corresponding eigenvectors; each column is an embedding dimension.
    """
    # Convert graph to sparse adjacency matrix
    A = nx.to_scipy_sparse_array(G, format='csr')

    # Compute Laplacian (normalized or unnormalized)
    L = csgraph.laplacian(A, normed=normed)

    # Compute the smallest k+1 eigenvalues/vectors (including the trivial 0)
    eig_vals, eig_vecs = eigsh(L, k=k+1, which='SM')

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eig_vals)
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Exclude the first trivial eigenvector (corresponding to eigenvalue ~0)
    return eig_vals[1:k+1], eig_vecs[:, 1:k+1]


if __name__ == "__main__":
    # Example usage for build_graph_from_edges
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    G = build_graph_from_edges(edges, directed=False)
    print("Nodes:", G.nodes())
    print("Edges:", G.edges())

    # Example usage for laplacian_eigenvectors
    vals, vecs = laplacian_eigenvectors(G, k=2)
    embeddings = {node: vecs[i] for i, node in enumerate(G.nodes())}
    print("Eigenvalues:", vals)
    print("Embeddings:", embeddings)