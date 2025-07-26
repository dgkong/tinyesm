# TinyESM

Reimplementation of ESM-2 8M parameter model for learning purposes. 

Goal is to beat ESM-2 8M parameter Validation Perplexity @ 270K steps: 10.45 ~2.3466

References:
1) ESM-2 implementation released by facebookresearch:
https://github.com/facebookresearch/esm
2) HuggingFace transformers ESM implementation:
https://github.com/huggingface/transformers/tree/v4.23.1/src/transformers/models/esm

step 4400/200000 | validation loss: 2.560618

There seemed to be a noticeable increase in memory usage and time per step potentially meaning memory leak.
- training run ended through early stopping (albeit very low patience) at 4400 steps
- on current configuration and device, meaningful number of steps seem infeasible
