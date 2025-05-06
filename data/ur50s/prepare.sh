#!/usr/bin/env bash
# --------------------------------------------------------------
# Split UniRef50 into train / validation for TinyESM
# --------------------------------------------------------------
set -euo pipefail

################################################################
FASTA="uniref50.fasta"   # input FASTA
CONTEXT=255              # TinyESM max context length
VAL_FRAC=0.10            # fraction of clusters → validation
SEED=42                  # RNG seed for reproducibility
SHARD_SIZE=100000        # Number of sequences per shard
################################################################

echo ">>> 1) Sampling ${VAL_FRAC} of clusters for validation..."
# Sample 10 % of records; keep it reproducible with --rand-seed
seqkit sample -p "${VAL_FRAC}" --rand-seed "${SEED}" "${FASTA}" \
  | seqkit fx2tab -n \
  > val_clusters.txt
echo "    -> wrote val_clusters.txt  ($(wc -l < val_clusters.txt) IDs)"

echo ">>> 2) Splitting FASTA into train / val ..."
# Validation FASTA (all lengths)
seqkit grep -n -f val_clusters.txt  "${FASTA}" -o val_raw.fasta
echo "    -> val_raw.fasta  ($(seqkit stats -T val_raw.fasta | awk 'NR==2{print $4}') seqs)"
# Training FASTA = everything not in val_clusters.txt
seqkit grep -n -v -f val_clusters.txt "${FASTA}" -o train.fasta
echo "    -> train.fasta ($(seqkit stats -T train.fasta | awk 'NR==2{print $4}') seqs)"

echo ">>> 3) Shard training dataset with 100k seqs/shard ..."
# Shard into 100k seqs / shard, and gzip each shard to keep I/O down
seqkit split -s "${SHARD_SIZE}" train.fasta -O train_shards
parallel gzip ::: train_shards/*.fasta

echo ">>> 4) Creating length-filtered validation set (≤${CONTEXT}) ..."
seqkit seq -g -M "${CONTEXT}" val_raw.fasta -o val.fasta
echo "    -> val.fasta       ($(seqkit stats -T val.fasta | awk 'NR==2{print $4}') seqs ≤${CONTEXT} aa)"

echo "Done!"
