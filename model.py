from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        num_embeddings_ = num_embeddings + padding_idx + 1
        super().__init__(num_embeddings_, embedding_dim, padding_idx)

    def forward(self, mask: torch.Tensor):
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx
        )


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.position_embeddings = LearnedPositionalEmbedding(config.max_seq_len+1, config.n_embd, 1)

    def forward(self, input, mask):
        tok_emb = self.word_embeddings(input)
        pos_emb = self.position_embeddings(mask)
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.layer_norm(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.self = nn.ModuleDict(dict(
            query = nn.Linear(config.n_embd, config.n_embd),
            key = nn.Linear(config.n_embd, config.n_embd),
            value = nn.Linear(config.n_embd, config.n_embd),
        ))
        self.output = nn.ModuleDict(dict(
            dense = nn.Linear(config.n_embd, config.n_embd)
        ))
        self.LayerNorm = nn.LayerNorm(config.n_embd)

    # ------------------------------------------------------------------
    # x: (T, B, E)   – ESM & fairseq use sequence‑first
    # attn_mask: (B, T)  with 0 = real token, 1 = pad   (optional)
    # need_head_weights: return per‑head attention map
    # ------------------------------------------------------------------
    def forward(self, x, attn_mask=None, need_head_weights=False):
        B, T, E = x.shape

        # ① pre‑norm (ESM‑1b is a pre‑norm transformer)
        x = self.LayerNorm(x)                      # (B, T, E)

        # ② Q, K, V projections
        q = self.self.query(x)                  # (B, T, E)
        k = self.self.key(x)
        v = self.self.value(x)

        # ③ reshape to heads
        #    (B, T, E) -> (B, n_head, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = split_heads(q) 
        k = split_heads(k)
        v = split_heads(v)

        # ④ raw attention scores
        # (B, n_head, T, d) @ (B, n_head, d, T) -> (B, n_head, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim ** -0.5  # q·kᵀ / √d_k

        # ⑤ add mask  (convert 1/0 padding mask to -inf on pads)
        if attn_mask is not None:
            # pad positions are where attn_mask == 0
            pad_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            attn_scores = attn_scores.masked_fill(pad_mask, float('-inf'))

        # ⑥ softmax → attention probs
        attn_probs = F.softmax(attn_scores, dim=-1) # (B, h, T, T)

        # ⑦ attention drop out could be inserted here if training

        # ⑧ weighted sum
        context = torch.matmul(attn_probs, v)   # (B, n_head, T, d)

        # merge heads back -> (B, T, E)
        context = context.transpose(1, 2).contiguous().view(B, T, E)

        # 9. output proj + residual
        x = self.output.dense(context)

        if need_head_weights:
            return x, attn_probs          # (B,T,E), (B, n_head, T, T)
        return x


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiheadAttention(config)
        self.intermediate = nn.ModuleDict(dict(
            dense = nn.Linear(config.n_embd, 4*config.n_embd)
        ))
        self.output = nn.ModuleDict(dict(
            dense = nn.Linear(4*config.n_embd, config.n_embd)
        ))
        self.LayerNorm = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.output.dense(F.gelu(self.intermediate.dense(self.LayerNorm(x))))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.n_layer)])
        self.emb_layer_norm_after = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        x = self.emb_layer_norm_after(x)
        return x


class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, x):                          # x : (B, T, E)
        x = F.gelu(self.dense(x))                  # hidden → GELU
        x = self.layer_norm(x)                     # LN
        logits = self.decoder(x) + self.bias       # projection + tied‑weight bias
        return logits 


@dataclass
class ESMConfig:
    max_seq_len: int = 255 # 256 with cls token
    toks_per_batch: int = 2048
    vocab_size: int = 33
    n_layer: int = 6 # number of layers
    n_head: int = 6 # number of heads
    n_embd: int = 768 # embedding dimension


class ESM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.esm = nn.ModuleDict(dict(
            embeddings = Embedding(config),
            encoder = Encoder(config),
        ))
        self.lm_head = LMHead(config)

        # weight sharing scheme
        self.lm_head.decoder.weight = self.esm.embeddings.word_embeddings.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, toks, mask, targets=None):
        # forward the embeddings
        x = self.esm.embeddings(toks, mask)
        x = x * mask.unsqueeze(-1).type_as(x) # x is (B,T,n_embd) but mask is (B,T), so have to unsqueeze to (B,T,1)

        # forward the transformer layers
        x = self.esm.encoder(x)

        # forward the classifier
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
