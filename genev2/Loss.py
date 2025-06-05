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

class MaskedCrossEntropyLoss(nn.Module):
    """
    Standard CE over two classes, but:
     - ignores padding (mask == True)
     - treats label == -1 (unknown) by biasing predictions toward a specified ratio
    """
    def __init__(
        self,
        unk_bias_ratio: float = 0.5,
        unk_weight: float = 1.0
    ):
        super().__init__()
        # ignore_index = -2 to skip padded positions
        self.ce = nn.CrossEntropyLoss(ignore_index=-2, reduction='mean')
        self.unk_bias_ratio = unk_bias_ratio
        self.unk_weight = unk_weight

    def forward(
        self,
        logits: torch.Tensor,         # (B, L, 2)
        labels: torch.Tensor,         # (B, L), unknown = -1
        padding_mask: torch.Tensor = None  # (B, L) bool
    ) -> torch.Tensor:
        B, L, C = logits.shape
        flat_logits = logits.view(-1, C)
        flat_labels = labels.view(-1)

        # apply padding mask
        if padding_mask is not None:
            flat_mask = padding_mask.view(-1)
            flat_labels = flat_labels.masked_fill(flat_mask, -2)

        # mask unknown (-1) as ignore_index too
        flat_labels = flat_labels.masked_fill(flat_labels == -1, -2)
        valid_known = (flat_labels != -2)
        if valid_known.sum() == 0:
            loss_main = 0.0
        else:
            # main CE over known labels
            loss_main = self.ce(flat_logits, flat_labels)

        # handle unknown positions (original labels == -1 before masking)
        unk_mask = (labels.view(-1) == -1)
        if padding_mask is not None:
            unk_mask &= ~padding_mask.view(-1)

        if unk_mask.any():
            # gather logits for unknown positions
            unk_logits = flat_logits[unk_mask]  # (N_unk, 2)
            # compute log-probs
            logp = F.log_softmax(unk_logits, dim=-1)  # (N_unk, 2)
            # target distribution [ratio for class0, 1-ratio for class1]
            t0 = self.unk_bias_ratio
            t1 = 1.0 - self.unk_bias_ratio
            # soft-cross-entropy: -sum(target * logp)
            loss_unk = -(t0 * logp[:,0] + t1 * logp[:,1]).mean()
            return loss_main + self.unk_weight * loss_unk
        return loss_main
    


class MaskedCrossEntropyLoss(nn.Module):
    """
    Standard CE over two classes, but:
     - ignores padding (mask == True)
     - treats label == -1 (unknown) by biasing predictions toward a specified ratio
     - uses SUM reduction (not mean) so that token counts in known vs unknown classes matter
    """
    def __init__(
        self,
        unk_bias_ratio: float = 0.5,
        unk_weight: float = 1.0
    ):
        super().__init__()
        # ignore_index = -2 to skip padded and unknown‐as‐ignored positions
        # reduction='sum' so we accumulate over tokens instead of averaging
        self.ce = nn.CrossEntropyLoss(ignore_index=-2, reduction='sum')
        self.unk_bias_ratio = unk_bias_ratio
        self.unk_weight = unk_weight

    def forward(
        self,
        logits: torch.Tensor,         # (B, L, 2)
        labels: torch.Tensor,         # (B, L), unknown = -1
        padding_mask: torch.Tensor = None  # (B, L) bool (True = pad)
    ) -> torch.Tensor:
        B, L, C = logits.shape
        device = logits.device

        # flatten logits & labels to shape (B*L, 2) and (B*L,)
        flat_logits = logits.view(-1, C)
        flat_labels = labels.view(-1)

        # 1) Apply padding mask: mark padded positions as ignore_index = -2
        if padding_mask is not None:
            flat_pad_mask = padding_mask.view(-1)                  # (B*L,)
            flat_labels = flat_labels.masked_fill(flat_pad_mask, -2)

        # 2) Also mask unknown labels (label == -1) as ignore_index for the 'main' CE
        #    We'll handle unknown tokens separately below.
        unknown_positions = (flat_labels == -1)
        flat_labels = flat_labels.masked_fill(unknown_positions, -2)

        # 3) Compute CE‐sum over all known tokens
        valid_known = (flat_labels != -2)  # these have 0/1 labels
        if valid_known.sum() == 0:
            # If no known tokens remain, set loss_main = 0 (as a tensor)
            loss_main = torch.tensor(0.0, device=device)
        else:
            # CrossEntropyLoss with reduction='sum' automatically sums over all non‐ignored tokens
            loss_main = self.ce(flat_logits, flat_labels)

        # 4) Handle unknown positions (where original label == -1 and not padding)
        unk_mask = (labels.view(-1) == -1)  # shape (B*L,)
        if padding_mask is not None:
            unk_mask &= ~padding_mask.view(-1)

        if unk_mask.any():
            # Gather logits only at unknown positions
            unk_logits = flat_logits[unk_mask]  # (N_unk, 2)
            # Log‐probabilities over the two classes
            logp = F.log_softmax(unk_logits, dim=-1)  # (N_unk, 2)
            # Desired target distribution: [t0, t1]
            t0 = self.unk_bias_ratio
            t1 = 1.0 - self.unk_bias_ratio
            # Compute sum of soft‐cross‐entropy over unknown tokens
            #   i.e. sum_{i in unk} [ −(t0 * logp[i,0] + t1 * logp[i,1]) ]
            loss_unk = -(t0 * logp[:, 0] + t1 * logp[:, 1]).sum()

            return loss_main + self.unk_weight * loss_unk

        return loss_main
        
class IntegratedMaskedCrossEntropyLoss(nn.Module):
    def __init__(self, unk_bias_ratio: float = 0.5, scale_factor: float = 0.001):
        super().__init__()
        assert 0.0 <= unk_bias_ratio <= 1.0, "unk_bias_ratio must be in [0,1]"
        self.unk_bias_ratio = unk_bias_ratio
        # A user‐provided constant to scale the sum of losses.
        # Typical usage: if L_raw can become very large, choose scale_factor < 1.0.
        self.scale_factor = scale_factor

    def forward(
        self,
        logits: torch.Tensor,        # (B, L, 2)
        labels: torch.Tensor,        # (B, L), in {0,1} or -1 for unknown
        padding_mask: torch.BoolTensor = None  # (B, L), True = pad
    ) -> torch.Tensor:
        B, L, C = logits.shape
        device = logits.device

        # Flatten to (B*L, 2) and (B*L,)
        flat_logits = logits.view(-1, C)   # (B·L, 2)
        flat_labels = labels.view(-1)      # (B·L,)

        # Build a pad mask (False if not provided)
        if padding_mask is not None:
            flat_pad = padding_mask.view(-1)  # (B·L,)
        else:
            flat_pad = torch.zeros(B * L, dtype=torch.bool, device=device)

        # Compute log‐probabilities once for all tokens
        logp = F.log_softmax(flat_logits, dim=-1)  # (B·L, 2)

        # Prepare a “soft target” distribution for every token
        #   - Known tokens: one‐hot [1,0] or [0,1]
        #   - Unknown tokens: [t0, t1] = [unk_bias_ratio, 1 - unk_bias_ratio]
        #   - Padded tokens: leave [0,0], will be masked out
        target = torch.zeros_like(logp)           # (B·L, 2)

        # 1) Known tokens (labels == 0 or 1)
        known_mask = (flat_labels == 0) | (flat_labels == 1)
        if known_mask.any():
            target[flat_labels == 0, 0] = 1.0
            target[flat_labels == 1, 1] = 1.0

        # 2) Unknown tokens (label == -1) that are not padding
        unk_mask = (flat_labels == -1) & (~flat_pad)
        if unk_mask.any():
            t0 = self.unk_bias_ratio
            t1 = 1.0 - self.unk_bias_ratio
            target[unk_mask, 0] = t0
            target[unk_mask, 1] = t1

        # 3) Compute per‐token KL divergence: KL(target ‖ p), summing over classes → (B·L,)
        per_token_kl = F.kl_div(logp, target, reduction='none').sum(dim=-1)

        # 4) Zero out loss for padded positions
        per_token_kl = per_token_kl.masked_fill(flat_pad, 0.0)

        # 5) Sum over all non‐padded tokens
        raw_sum = per_token_kl.sum()

        # 6) Scale by a constant for numerical stability
        loss = raw_sum * self.scale_factor

        return loss
