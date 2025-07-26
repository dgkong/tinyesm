#!/usr/bin/env bash
set -euo pipefail

FASTA="uniref50.fasta"   # input FASTA
RESIDUES=510             # max residue length
VAL_FRAC=0.01            # 1% validation split
SEED=42                  # RNG seed for reproducibility
SHARD_SIZE=200_000       # Number of sequences per shard
# --------------------------------------------------------------

echo ">>> 1) Sampling ${VAL_FRAC} of clusters for validation..."
seqkit sample -p "${VAL_FRAC}" --rand-seed "${SEED}" "${FASTA}" \
  | seqkit fx2tab -n \
  > val_ids.txt
echo "    -> val_ids.txt  ($(wc -l < val_ids.txt) seqs)"

echo ">>> 2) Splitting FASTA into train / val ..."
# Validation FASTA (all lengths)
seqkit grep -n -f val_ids.txt  "${FASTA}" -o val_raw.fasta
echo "    -> val_raw.fasta  ($(seqkit stats -T val_raw.fasta | awk 'NR==2{print $4}') seqs)"
# Training FASTA = everything not in val_ids.txt
seqkit grep -n -v -f val_ids.txt "${FASTA}" -o train.fasta
echo "    -> train.fasta ($(seqkit stats -T train.fasta | awk 'NR==2{print $4}') seqs)"

echo ">>> 3) Shard training dataset with 200k seqs/shard ..."
# Shard into 200k seqs / shard, and gzip each shard to keep I/O down
seqkit split -s "${SHARD_SIZE}" train.fasta -O train_shards
parallel gzip ::: train_shards/*.fasta

echo ">>> 4) Creating length-filtered validation set (≤${RESIDUES}) ..."
seqkit seq -g -M "${RESIDUES}" val_raw.fasta -o val.fasta
echo "    -> val.fasta       ($(seqkit stats -T val.fasta | awk 'NR==2{print $4}') seqs ≤${RESIDUES} aa)"

echo "Done!"
