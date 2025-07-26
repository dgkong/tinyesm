import glob
import gzip
import random

from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, EsmTokenizer

TRAIN_SHARDS = "data/ur50s/train_shards/*.fasta.gz"
VAL_DATA = "data/ur50s/val.fasta"

class FastaSequences:
    def __init__(self, sequence_strs):
        self.sequence_strs = list(sequence_strs)

    @classmethod
    def from_file(cls, fasta_file):
        sequence_strs = []
        buf = []

        def _flush_current_seq():
            nonlocal buf
            if not buf:
                return
            sequence_strs.append("".join(buf))
            buf = []

        open_fn = gzip.open if str(fasta_file).endswith(".gz") else open
        with open_fn(fasta_file, "rt") as infile:
            for line in infile:
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                else: # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        return cls(sequence_strs)

    def __iter__(self):
        return iter(self.sequence_strs)

class ShardedMLMDataset(IterableDataset):
    def __init__(self, crop_len, tokens_per_batch, split):
        assert split in {'train', 'val'}
        self.split = split
        self.data = glob.glob(TRAIN_SHARDS) if split == 'train' else [VAL_DATA]
        self.crop_len = crop_len
        self.tokens_per_batch = tokens_per_batch
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer)
        
    def __iter__(self):
        if self.split == 'train':
            random.shuffle(self.data)
        buf, max_len = [], 0
        for shard in self.data:
            ds = FastaSequences.from_file(shard)
            if self.split == 'train':
                random.shuffle(ds.sequence_strs)
            for seq in ds:
                # random crop if sequence too long
                if len(seq) > self.crop_len:
                    if self.split == 'train':
                        start = random.randrange(len(seq) - self.crop_len + 1)
                    else:
                        # Deterministic center crop for validation
                        start = (len(seq) - self.crop_len) // 2
                    seq = seq[start:start + self.crop_len]
                
                prospective_max = max(max_len, len(seq) + 2) # + 2 for <cls> and <eos> tokens
                if prospective_max * (len(buf) + 1) > self.tokens_per_batch:
                    if buf:
                        yield self._make_batch(buf)
                    buf, max_len = [], 0

                buf.append(seq)
                max_len = max(max_len, len(seq) + 2)
        if buf:
            yield self._make_batch(buf)

    def _make_batch(self, sequences):
        tok = self.tokenizer(
            sequences,
            return_special_tokens_mask=True,
        )
        # Build examples: <cls> + seq + <eos>
        examples = [{"input_ids": ids, "special_tokens_mask": mask} 
                    for ids, mask in zip(tok["input_ids"], tok["special_tokens_mask"])]
        # DataCollator will dynamically pad and apply MLM masking
        batch = self.collator(examples)
        return batch["input_ids"], batch["labels"], batch["attention_mask"]

