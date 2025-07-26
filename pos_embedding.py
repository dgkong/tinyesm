import torch
import torch.nn as nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1) # (B,n_head,T,head_dim/2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE) proposed in https://arxiv.org/abs/2104.09864.
    Uses Huggingface implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py#L76
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim)) # (dim/2,)
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dim=2):
        seq_len = x.shape[seq_dim]
        
        # Reset the tables if the sequence length has changed
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            # Create a tensor representing positions: [0, 1, ..., seq_len - 1] (seq_len,)
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq) # (seq_len,dim/2)
            # Concatenate freqs with itself to get pairs for complex number representation (seq_len,dim)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :] # (1,1,seq_len,dim)
            self._sin_cached = emb.sin()[None, None, :, :] # (1,1,seq_len,dim)

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=-2)

        q_rot = apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k_rot = apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        return q_rot, k_rot


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        num_embeddings_ = num_embeddings + padding_idx + 1
        super().__init__(num_embeddings_, embedding_dim, padding_idx)

    def forward(self, mask: torch.Tensor):
        # padding_idx row of embedding table is for padded positions
        # <cls>: row 2 | pos 1: row 3 | pos 2: row 4...
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return super().forward(positions)
