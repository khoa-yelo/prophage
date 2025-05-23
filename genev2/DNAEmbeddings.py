import h5py
import numpy as np
from typing import List, Tuple, Dict

class DNAEmbeddings:
    def __init__(self, embeddings_path: str):
        """
        Load DNA embeddings and associated IDs from an HDF5 file.

        embeddings_path: path to HDF5 file containing datasets:
            - 'dnafeat': float array of shape (N, D)
            - 'ids': string array of shape (N,) where each entry follows the pattern
               'lcl|{record_id}_cds_{protein_id}_...'
            - 'locus_tag': string array of shape (N,) of locus tags matching features
        """
        self.embeddings, raw_ids, self.locus_tags = self.load_embeddings(embeddings_path)
        self.embedding_dim = self.embeddings.shape[1]

        # Parse raw_ids into (record_id, protein_id) pairs
        id_pairs: List[Tuple[str, str]] = []
        for raw in raw_ids:
            s = raw
            # remove prefix if present
            if '|' in s:
                _, s = s.split('|', 1)
            # split at '_cds_' to separate record and protein portion
            if '_cds_' in s:
                rec, rest = s.split('_cds_', 1)
                # protein_id is the first segment before any further underscores
                parts = rest.split('_')
                prot = parts[0]
            else:
                # fallback: entire string is record, no protein
                rec, prot = s, ''
            id_pairs.append((rec, prot))

        # store as numpy array of shape (N,2)
        self.ids: np.ndarray = np.array(id_pairs, dtype='<U')

        # build lookup maps for fast indexing by (record_id, protein_id)
        self._index: Dict[Tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(id_pairs)
        }
        # build lookup by locus_tag
        self._locus_index: Dict[str, int] = {
            tag: idx for idx, tag in enumerate(self.locus_tags)
        }

    def load_embeddings(self, embeddings_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(embeddings_path, 'r') as f:
            embeddings = f['dnafeat'][:]           # shape: (N, D)
            raw_ids     = f['id'].asstr()[:]      # shape: (N,)
            locus_tags  = f['locus_tag'].asstr()[:]  # shape: (N,)
        return embeddings, raw_ids, locus_tags

    def get_embeddings(
        self,
        locus_tags: List[str],
        record_ids: List[str],
        protein_ids: List[str]
    ) -> np.ndarray:
        """
        Given parallel lists of locus_tags, record_ids, and protein_ids, return the corresponding
        embeddings array of shape (len(locus_tags), D).

        Tries to match on locus_tag first; if a locus_tag is missing, falls back to (record_id, protein_id).
        Returns a zero vector for entries not found.
        """
        if not (len(locus_tags) == len(record_ids) == len(protein_ids)):
            raise ValueError("locus_tags, record_ids, and protein_ids must have the same length")

        embs: List[np.ndarray] = []
        for tag, rec, prot in zip(locus_tags, record_ids, protein_ids):
            # Attempt locus_tag lookup
            if tag in self._locus_index:
                idx = self._locus_index[tag]
                embs.append(self.embeddings[idx])
            else:
                # Fallback to record/protein lookup
                key = (rec, prot)
                if key in self._index:
                    idx = self._index[key]
                    embs.append(self.embeddings[idx])
                else:
                    # return zero vector if not found
                    embs.append(np.zeros(self.embedding_dim, dtype=self.embeddings.dtype))

        return np.stack(embs, axis=0)

    def get_all_ids(self) -> List[Tuple[str, str]]:
        """
        Return a list of all (record_id, protein_id) pairs in the same order as embeddings.
        """
        return [tuple(pair) for pair in self.ids.tolist()]
