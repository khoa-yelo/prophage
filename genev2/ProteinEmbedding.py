"""
This module provides a class for loading
and managing protein embeddings from an HDF5 file.
It includes methods for loading embeddings,
retrieving embeddings for specific proteins,
and accessing all embeddings and protein IDs.

Author: Khoa Hoang
Date: 05/04/2025
"""
import h5py
import pandas as pd

class ProteinEmbedding:
    def __init__(self, embeddings_path, cluster_map_path):
        self.embeddings_path = embeddings_path
        self.cluster_map_path = cluster_map_path
        self.cluster_map = self.load_map(self.cluster_map_path)
        self.mean_embeddings = None
        self.max_embeddings = None
        self.mean_middle_embeddings = None
        self.max_middle_embeddings = None
        self.protein_cluster_ids = None
        self.protein_ids = None
        self.id_embedding_map = None
        self.load_embeddings(self.embeddings_path)

    def load_map(self, protein_ids_map):
        df_map = pd.read_csv(protein_ids_map, sep="\t", header=None)
        df_map.columns = ["protein_cluster", "protein_id"]
        map = df_map.set_index("protein_id").to_dict()["protein_cluster"]
        self.protein_ids = df_map["protein_id"].tolist()
        return map
    
    def load_embeddings(self, embeddings_path):
        with h5py.File(embeddings_path, "r") as hf:
            self.mean_embeddings = hf["mean"][:]
            self.max_embeddings = hf["max"][:]
            self.mean_middle_embeddings = hf["mean_middle_layer_12"][:]
            self.max_middle_embeddings = hf["max_middle_layer_12"][:]
            self.protein_cluster_ids = hf["protein_ids"].asstr()[:]
            self.id_embedding_map = {protein_id: idx for idx, protein_id in enumerate(self.protein_cluster_ids)}


    def get_protein_indices(self, protein_ids: list[str]):
        if isinstance(protein_ids, str):
            protein_ids = [protein_ids]
        cluster_ids = [self.cluster_map[protein_id] for protein_id in protein_ids]
        protein_indices = [self.id_embedding_map[cluster_id] for cluster_id in cluster_ids]
        return protein_indices
    

    def get_embeddings(self, protein_ids, metric = "mean"):
        if isinstance(protein_ids, str):
            protein_ids = [protein_ids]
        indices = self.get_protein_indices(protein_ids)
        if metric == "mean":
            return self.mean_embeddings[indices]
        elif metric == "max":
            return self.max_embeddings[indices]
        elif metric == "mean_middle":
            return self.mean_middle_embeddings[indices]
        elif metric == "max_middle":
            return self.max_middle_embeddings[indices]
        else:
            raise ValueError("Invalid metric. Choose from 'mean', 'max', 'mean_middle', or 'max_middle'.")
        
    def get_all_embeddings(self):
        all_embeddings = {
            "mean": self.mean_embeddings,
            "max": self.max_embeddings,
            "mean_middle": self.mean_middle_embeddings,
            "max_middle": self.max_middle_embeddings
        }
        return all_embeddings
    
    def get_all_cluster_ids(self):
        return self.protein_cluster_ids
    
    def get_all_protein_ids(self):
        return self.protein_ids
    
    