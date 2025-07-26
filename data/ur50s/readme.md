# UniRef50

"UniRef50 is built by clustering UniRef90 seed sequences that have at least 50% sequence identity to and 80% overlap with the longest sequence in the cluster"

"UniRef50 yields a database size reduction of approximately 79%"

After running 'prepare.sh':

- train.fasta has 68,792,632 seqs
    - sharded into 344 files ~200k seqs each
- val.fasta has 607,065 seqs <=510 aa residues (ESM-2 uses <cls> and <eos> tokens)

Used 1% instead of 0.5% validation split used in ESM-2 and skipped the >50% identical filtering process used in ESM-2 due to compute constraints.
