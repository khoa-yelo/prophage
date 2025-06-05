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

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    Adds a (batch, seq_len, d_model) pe tensor to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerClassifier(nn.Module):
    """
    Transformer encoder with sinusoidal positional encoding + token classification head.
    Embeddings for each of the protein, graph, and dna tracks are normalized
    individually before being concatenated with the other track features.
    All Linear layers are initialized with Kaiming uniform (suitable for ReLU),
    and Embedding layers are initialized with a normal distribution (std=0.02).
    """
    def __init__(
        self,
        type_vocab_size: int,
        biotype_vocab_size: int,
        strand_vocab_size: int,
        track_embed_dim: int,
        strand_embed_dim: int,
        homologs_embed_dim: int,
        protein_embed_dim: int,
        dna_embed_dim: int,
        graph_embed_dim: int,
        protein_embed_dim_shrink: int,
        dna_embed_dim_shrink: int,
        graph_embed_dim_shrink: int,
        embed_dim: int = 300,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 10000
    ):
        super().__init__()

        # 1D track embeddings
        self.type_emb    = nn.Embedding(type_vocab_size,    track_embed_dim)
        self.biotype_emb = nn.Embedding(biotype_vocab_size, track_embed_dim)
        self.strand_emb  = nn.Embedding(strand_vocab_size,  strand_embed_dim)
        self.homologs_proj = nn.Linear(1, homologs_embed_dim)

        # raw embedding dimensions (for finetuning)
        self.protein_embed_dim = protein_embed_dim
        self.dna_embed_dim     = dna_embed_dim
        self.graph_embed_dim   = graph_embed_dim

        # shrunk dimensions after finetuning
        self.protein_embed_dim_shrink = protein_embed_dim_shrink
        self.dna_embed_dim_shrink     = dna_embed_dim_shrink
        self.graph_embed_dim_shrink   = graph_embed_dim_shrink

        # Linear layers to project down pretrained embeddings to smaller dims
        self.protein_finetune = nn.Linear(self.protein_embed_dim, self.protein_embed_dim_shrink)
        self.dna_finetune     = nn.Linear(self.dna_embed_dim,     self.dna_embed_dim_shrink)
        self.graph_finetune   = nn.Linear(self.graph_embed_dim,   self.graph_embed_dim_shrink)

        # LayerNorm layers to normalize each shrunk embedding track
        self.protein_norm = nn.LayerNorm(self.protein_embed_dim_shrink)
        self.dna_norm     = nn.LayerNorm(self.dna_embed_dim_shrink)
        self.graph_norm   = nn.LayerNorm(self.graph_embed_dim_shrink)

        # Compute total feature dimension after concatenation
        total_feat_dim = (
              track_embed_dim   # type
            + track_embed_dim   # biotype
            + strand_embed_dim  # strand
            + homologs_embed_dim# homologs
            + protein_embed_dim_shrink
            + dna_embed_dim_shrink
            + graph_embed_dim_shrink
        )

        # Normalize the concatenated feature vector
        self.input_norm = nn.LayerNorm(total_feat_dim)
        self.input_proj = nn.Linear(total_feat_dim, embed_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2)
        )

        # Initialize weights as suggested
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for Linear and Embedding layers:
          - Linear: Kaiming Uniform (for ReLU activations)
          - Embedding: Normal(mean=0, std=0.02)
          - Bias terms: uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)]
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming uniform (for ReLU)
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(module.bias, -bound, bound)

            elif isinstance(module, nn.Embedding):
                # Initialize embeddings with small normal noise
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        type_ids: torch.LongTensor,
        biotype_ids: torch.LongTensor,
        strand_ids: torch.LongTensor,
        homologs: torch.FloatTensor,
        protein_emb: torch.FloatTensor,
        dna_emb: torch.FloatTensor,
        graph_emb: torch.FloatTensor,
        mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        # Embed 1D categorical tracks
        te = self.type_emb(type_ids)      # (B, L, track_embed_dim)
        be = self.biotype_emb(biotype_ids) # (B, L, track_embed_dim)

        # Remap strand {–1, 0, +1} to {0, 1, 2,...} indices
        strand_idx = ((strand_ids + 1) // 2).clamp(0, self.strand_emb.num_embeddings - 1)
        se = self.strand_emb(strand_idx)  # (B, L, strand_embed_dim)

        # Project homologs (float) into embedding space
        he = self.homologs_proj(homologs.unsqueeze(-1))  # (B, L, homologs_embed_dim)

        # Finetune each pretrained embedding track down to smaller dimension
        protein_emb = self.protein_finetune(protein_emb)  # (B, L, protein_embed_dim_shrink)
        dna_emb     = self.dna_finetune(dna_emb)          # (B, L, dna_embed_dim_shrink)
        graph_emb   = self.graph_finetune(graph_emb)      # (B, L, graph_embed_dim_shrink)

        # Normalize each finetuned embedding track separately
        protein_emb = self.protein_norm(protein_emb)
        dna_emb     = self.dna_norm(dna_emb)
        graph_emb   = self.graph_norm(graph_emb)

        # Concatenate all features along the last dimension
        x = torch.cat([te, be, se, he, protein_emb, dna_emb, graph_emb], dim=-1)  # (B, L, total_feat_dim)

        # Normalize the concatenated features, then project into model dimension
        x = self.input_norm(x)
        x = self.input_proj(x)             # (B, L, embed_dim)

        # Add sinusoidal positional encodings
        x = self.pos_encoder(x)            # (B, L, embed_dim)

        # Pass through Transformer encoder
        out = self.transformer(x, src_key_padding_mask=mask)  # (B, L, embed_dim)

        # Compute per-token logits
        logits = self.classifier(out)      # (B, L, 2)
        return logits