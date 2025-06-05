import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, average_precision_score

class NoisySubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        prev_flag = self.base.training
        self.base.training = True               # turn on noise
        sample = self.base[real_idx]
        self.base.training = prev_flag          # restore original flag
        return sample


class ViromeDataset(Dataset):
    def __init__(
        self,
        db,
        protein_embeddings_source,
        dna_embeddings_source,
        graph_embeddings_source,
        max_sequence_length: int,
        padding_value: float = 0.0,
        training:bool =False
    ):
        """
        Preload and pad/truncate all samples to a fixed length for fast indexing.

        Args:
            db: IsolateDB instance
            protein_embeddings_source: ProteinEmbeddings instance
            dna_embeddings_source: DNAEmbeddings instance
            graph_embeddings_source: GraphEmbeddings instance
            max_sequence_length: fixed length to pad/truncate sequences
            padding_value: value to use for padding
        """
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        self.training = training
        # Store references
        self.db = db
        self.protein_source = protein_embeddings_source
        self.dna_source = dna_embeddings_source
        self.graph_source = graph_embeddings_source

        # Load vocabulary mappings
        self.record_ids = db.get_all_record_ids()
        self.type_categories = db.get_all_types()
        self.biotype_categories = db.get_all_biotypes()
        self.type_to_index = {cat: idx for idx, cat in enumerate(self.type_categories)}
        self.biotype_to_index = {cat: idx for idx, cat in enumerate(self.biotype_categories)}

        # Preload and pad all samples
        self.samples = []
        for record_id in self.record_ids:
            features = db.get_record_feature(record_id)
            seq_length = len(features['type'])
            # filter out seq_len < 2
            if seq_length < 2:
                continue
            # 1D categorical tracks
            type_indices = torch.tensor(
                [self.type_to_index.get(t, 0) for t in features['type']],
                dtype=torch.long
            )
            biotype_indices = torch.tensor(
                [self.biotype_to_index.get(b, 0) for b in features['biotype']],
                dtype=torch.long
            )
            raw_strand = torch.tensor(features['strand'], dtype=torch.long)
            strand_indices = ((raw_strand + 1) // 2).clamp(0, 1)

            # Homolog count track (float), apply log-scaling
            homologs_raw = torch.tensor(features['homologs'], dtype=torch.float32)
            homologs_scaled = torch.log1p(homologs_raw)

            # Label track
            label_tensor = torch.full((seq_length,), -1, dtype=torch.long)
            pos_mask = torch.tensor(features['prophage']) == 1
            neg_mask_type = torch.tensor(
                [b in ("tRNA", "rRNA") for b in features['biotype']],
                dtype=torch.bool
            )
            neg_mask = torch.tensor(features['species_prev']) > 250
            label_tensor[neg_mask] = 0
            for idx, product in enumerate(features['product']):
                if 'hypothetical protein' in str(product).lower():
                    label_tensor[idx] = -1
            label_tensor[pos_mask] = 1
            label_tensor[neg_mask_type] = 0
            
            # Embedding tracks
            # Protein embeddings
            protein_dim = self.protein_source.embedding_dim
            protein_matrix = np.zeros((seq_length, protein_dim), dtype=np.float32)
            valid_positions = [i for i, pid in enumerate(features['protein_id']) if pid]
            if valid_positions:
                valid_pids = [features['protein_id'][i] for i in valid_positions]
                batch_embs = self.protein_source.get_embeddings(valid_pids)
                for i, emb in zip(valid_positions, batch_embs):
                    protein_matrix[i] = emb
            protein_embeddings = torch.tensor(protein_matrix, dtype=torch.float32)

            # DNA embeddings
            dna_matrix = self.dna_source.get_embeddings(
                features['locus_tag'],
                [record_id] * seq_length,
                features['protein_id']
            )
            dna_embeddings = torch.tensor(dna_matrix, dtype=torch.float32)

            # Graph embeddings
            graph_dim = self.graph_source.embedding_dim
            graph_matrix = np.zeros((seq_length, graph_dim), dtype=np.float32)
            if valid_positions:
                graph_batch = self.graph_source.get_embeddings(valid_pids)
                for i, emb in zip(valid_positions, graph_batch):
                    graph_matrix[i] = emb
            graph_embeddings = torch.tensor(graph_matrix, dtype=torch.float32)

            # Padding/truncation helpers
            def pad_1d(tensor_1d: torch.Tensor) -> torch.Tensor:
                length = tensor_1d.size(0)
                if length < self.max_sequence_length:
                    return F.pad(
                        tensor_1d,
                        (0, self.max_sequence_length - length),
                        value=self.padding_value
                    )
                return tensor_1d[: self.max_sequence_length]

            def pad_2d(tensor_2d: torch.Tensor) -> torch.Tensor:
                length, dim = tensor_2d.size()
                if length < self.max_sequence_length:
                    return F.pad(
                        tensor_2d,
                        (0, 0, 0, self.max_sequence_length - length),
                        value=self.padding_value
                    )
                return tensor_2d[: self.max_sequence_length, :]

            # Apply padding/truncation
            type_track      = pad_1d(type_indices)
            biotype_track   = pad_1d(biotype_indices)
            strand_track    = pad_1d(strand_indices)
            homologs_track  = pad_1d(homologs_scaled)
            label_track     = pad_1d(label_tensor)
            protein_track   = pad_2d(protein_embeddings)
            dna_track       = pad_2d(dna_embeddings)
            graph_track     = pad_2d(graph_embeddings)

            # Padding mask: True for padded positions
            actual_len = min(seq_length, self.max_sequence_length)
            padding_mask = torch.arange(self.max_sequence_length) >= actual_len

            self.samples.append({
                'type_track': type_track,
                'biotype_track': biotype_track,
                'strand_track': strand_track,
                'homologs_track': homologs_track,
                'protein_embeddings': protein_track,
                'dna_embeddings': dna_track,
                'graph_embeddings': graph_track,
                'labels': label_track,
                'padding_mask': padding_mask,
                'record_id': record_id
            })
    
    def get_test_index(self):
        """
        Returns indices of samples that have NO positive prophage labels (i.e., label==1).
        """
        return [
            idx
            for idx, sample in enumerate(self.samples)
            # `.any().item()` checks if any position == 1
            if not (sample['labels'] == 1).any().item()
        ]

    def get_train_index(self):
        """
        Returns indices of samples that have AT LEAST one positive prophage label (i.e., label==1).
        """
        return [
            idx
            for idx, sample in enumerate(self.samples)
            if (sample['labels'] == 1).any().item()
        ]
    
    def get_record_ids(self):
        record_ids = []
        for sample in self.samples:
            record_ids.append(sample["record_id"])
            
        return record_ids
        
    def _add_label_noise(self, labels):
        labels = labels.clone()
        for val, p_zero, p_neg1 in [
            (-1, (0.3, 0.5), (0.0, 0.0)),  # unknown → 0 or 1
            (1, (0.15, 0.25), (0.05, 0.15))    # 1 → 0 or -1
        ]:
            idxs = (labels == val).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            n = len(idxs)
            n_zero = int(n * random.uniform(*p_zero))
            n_neg1 = int(n * random.uniform(*p_neg1))
            perm = idxs[torch.randperm(n)]
            if val == -1:
                labels[perm[:n_zero]] = 0
                labels[perm[n_zero:n_zero + n_neg1]] = 1
            elif val == 1:
                labels[perm[:n_zero]] = 0
                labels[perm[n_zero:n_zero + n_neg1]] = -1
        return labels
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index].copy()
        if self.training:
            sample['labels'] = self._add_label_noise(sample['labels'])
        return sample