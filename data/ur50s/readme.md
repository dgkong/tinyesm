# UniRef50

"UniRef50 is built by clustering UniRef90 seed sequences that have at least 50% sequence identity to and 80% overlap with the longest sequence in the cluster"

"UniRef50 yields a database size reduction of approximately 79%"

After running 'prepare.sh':

- train.fasta has 62,538,836 seqs
    - sharded into 626 files ~100k seqs each
- val.fasta has 4,329,087 seqs ≤256 aa
