import h5py
import pandas as pd

class GraphEmbeddings:
    def __init__(self, embeddings_path, cluster_map_path, id_embedding_map):
        self.embeddings_path = embeddings_path
        self.cluster_map_path = cluster_map_path
        self.protein_ids = None
        self.cluster_map = self.load_map(self.cluster_map_path)
        self.mean_embeddings = None
        self.max_embeddings = None
        self.mean_middle_embeddings = None
        self.max_middle_embeddings = None
        self.ref_protein_ids = None
        self.id_embedding_map = id_embedding_map
        self.load_embeddings(self.embeddings_path)
        self.embedding_dim = self.graph_embeddings.shape[1]

    def load_map(self, protein_ids_map):
        df_map = pd.read_csv(protein_ids_map, sep="\t", header=None)
        df_map.columns = ["protein_cluster", "protein_id"]
        cluster_map = df_map.set_index("protein_id").to_dict()["protein_cluster"]
        self.protein_ids = df_map["protein_id"].tolist()
        return cluster_map
    
    def load_embeddings(self, embeddings_path):
        with h5py.File(embeddings_path, "r") as hf:
            self.graph_embeddings = hf["graph_embeddings"][:]
            self.ref_protein_ids = hf["protein_ids"].asstr()[:]

    def get_protein_indices(self, protein_ids: list[str]):
        if isinstance(protein_ids, str):
            protein_ids = [protein_ids]
        cluster_ids = [self.cluster_map[protein_id] for protein_id in protein_ids]
        protein_indices = [self.id_embedding_map[cluster_id] for cluster_id in cluster_ids]
        return protein_indices
    
    def get_embeddings(self, protein_ids):
        if isinstance(protein_ids, str):
            protein_ids = [protein_ids]
        indices = self.get_protein_indices(protein_ids)
        return self.graph_embeddings[indices]
        
    def get_all_embeddings(self):
        return self.graph_embeddings[:]
    
    def get_all_ref_protein_ids(self):
        return self.ref_protein_ids
    
    def get_all_protein_ids(self):
        return self.protein_ids
    
    