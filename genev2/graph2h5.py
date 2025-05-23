import h5py
import pandas as pd
import numpy as np
import networkx as nx
import time
import faiss

edges_path = "/orange/sai.zhang/khoa/repos/prophage/data/ecoli_ppi_edges.txt"
df_edges = pd.read_csv(edges_path, sep = " ")
ecoli_embeddings = None
with h5py.File("ecoli_embeddings.h5") as f:
    ecoli_embeddings = f["mean_middle"][...]
    samples = f["samples"][...]
all_embeddings = proteinEmbeddings.get_all_embeddings()["mean_middle"]
G = build_graph_from_edges(list((df_edges[["protein1", "protein2"]].values)))
vals, vecs = laplacian_eigenvectors(G, k=16)
graph_embeddings = {node: vecs[i] for i, node in enumerate(G.nodes())}
# Build FAISS index
index = faiss.IndexFlatL2(ecoli_embeddings.shape[1])
index.add(ecoli_embeddings)
tic = time.time()
distances, indices = index.search(all_embeddings, 5)
toc = time.time()
print(toc - tic)

cluster_ids = proteinEmbeddings.get_all_cluster_ids()
graph_tokens = dict(zip(cluster_ids, indices[:,0]))
graph_protein_tokens = {}
for key, val in graph_tokens.items():
    graph_protein_tokens[key] = samples[val]
graph_pes = np.zeros((len(samples), 16))
for i, sample in enumerate(samples):
    pe = graph_embeddings[sample.decode()]
    graph_pes[i] = pe
with h5py.File("graph_embeddings.h5", "w") as f:
    f.create_dataset('graph_embeddings', data=graph_pes)
    dt = h5py.string_dtype(encoding='utf-8')
    f.create_dataset('protein_ids',
                     data=samples,
                     dtype=dt)
np.save("protein_cluster_graph_map.npy", graph_tokens)