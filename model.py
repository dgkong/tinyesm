from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from pos_embedding import LearnedPositionalEmbedding, RotaryPositionalEmbedding


@dataclass
class ESMConfig:
    max_seq_len: int = 256                  # 254 residues w/o cls and eos tokens
    vocab_size: int = 33                    # number of tokens
    pad_token_id: int = 1
    mask_token_id: int = 32
    n_layer: int = 6                        # number of layers
    n_head: int = 20                        # number of heads
    n_embd: int = 320                       # embedding dimension
    ffwd_dim: int = 1280                    # 4 * n_embd
    position_embedding_type: str = "rotary" # "absolute" or "rotary"
    token_dropout: bool = True              # dropout <mask> token


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = None

        if config.position_embedding_type == "absolute":
            self.position_embeddings = LearnedPositionalEmbedding(
                config.max_seq_len, config.n_embd, padding_idx=config.pad_token_id
            )

        self.mask_token_id = config.mask_token_id
        self.token_dropout = config.token_dropout
        self.position_embedding_type = config.position_embedding_type

    def forward(self, input, mask):
        # input and mask are (B,T)
        emb = self.word_embeddings(input) # (B,T,n_embd)

        if self.token_dropout:
            emb = emb.masked_fill((input == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs
            src_lengths = mask.sum(-1) # (B)
            mask_ratio_observed = (input == self.mask_token_id).sum(-1).float() / src_lengths
            scale = ((1 - mask_ratio_train) / (1 - mask_ratio_observed)).clamp(max=4.0)
            # scaling is necessary so different sequences don't have drastically different signal magnitudes
            emb = (emb * scale[:, None, None]).to(emb.dtype)

        if self.position_embeddings is not None:
            emb += self.position_embeddings(mask) # (B,T,n_embd)

        emb = emb * mask.unsqueeze(-1).type_as(emb) # mask is (B,T), unsqueeze to (B,T,1)

        return emb
        

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
        self.output.dense.RES_SCALE_INIT = 1
        self.LayerNorm = nn.LayerNorm(config.n_embd)

        self.rotary_embeddings = None
        if config.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryPositionalEmbedding(dim=self.head_dim)

    def forward(self, x, mask, need_head_weights=False):
        B, T, C = x.shape

        # Pre-Norm
        x = self.LayerNorm(x)

        q = self.self.query(x)
        k = self.self.key(x)
        v = self.self.value(x)

        # (B,T,C) -> (B,n_head,T,head_dim)
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = split_heads(q) * self.head_dim**-0.5 # scaled query before dot product
        k = split_heads(k)
        v = split_heads(v)

        if self.rotary_embeddings is not None:
            q, k = self.rotary_embeddings(q, k)

        # (B,n_head,T,head_dim) @ (B,n_head,head_dim,T) -> (B,n_head,T,T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            # pad positions are where mask == 0
            pad_idxs = (mask == 0)[:,None,None,:] # (B,1,1,T)
            attn_scores = attn_scores.masked_fill(pad_idxs, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1) # (B,n_head,T,T)

        context = torch.matmul(attn_probs, v) # (B,n_head,T,head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, C) # merge heads back -> (B,T,C)
        x = self.output.dense(context)

        if need_head_weights:
            return x, attn_probs # (B,T,C), (B,n_head,T,T)
        return x


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiheadAttention(config)
        self.intermediate = nn.ModuleDict(dict(
            dense = nn.Linear(config.n_embd, config.ffwd_dim)
        ))
        self.output = nn.ModuleDict(dict(
            dense = nn.Linear(config.ffwd_dim, config.n_embd)
        ))
        self.output.dense.RES_SCALE_INIT = 1
        self.LayerNorm = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, mask):
        x = x + self.attention(x, mask)
        x = x + self.output.dense(F.gelu(self.intermediate.dense(self.LayerNorm(x))))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.n_layer)])
        self.emb_layer_norm_after = nn.LayerNorm(config.n_embd)

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        x = self.emb_layer_norm_after(x)
        return x


class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, x):                         
        x = F.gelu(self.dense(x))
        x = self.layer_norm(x)
        logits = self.decoder(x) + self.bias
        return logits 


class ESM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.esm = nn.ModuleDict(dict(
            embeddings = Embedding(config),
            encoder = Encoder(config),
        ))
        self.lm_head = LMHead(config)

        # tied decoder and embedding weights
        self.lm_head.decoder.weight = self.esm.embeddings.word_embeddings.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RES_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        # PyTorch LayerNorm default init is mean = 1.0, bias = 0
        elif isinstance(module, LMHead):
            nn.init.zeros_(module.bias)

    def forward(self, toks, mask=None, targets=None):
        if mask is None:
            mask = torch.ones_like(toks)

        x = self.esm.embeddings(toks, mask) # (B,T,n_embd)
 
        x = self.esm.encoder(x, mask) # (B,T,n_embd)

        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.98), eps=1e-8)
        return optimizer
