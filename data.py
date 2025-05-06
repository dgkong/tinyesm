
import glob
import gzip
import random

from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, EsmTokenizer

TRAIN_SHARDS = "data/ur50s/train_shards/*.fasta.gz"
TOY_TRAIN_DATA = "data/ur50s/train_shards/train.part_023.fasta.gz"
VAL_DATA = "data/ur50s/val.fasta"

class FastaSequences(object):
    def __init__(self, sequence_strs):
        self.sequence_strs   = list(sequence_strs)

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

    def __len__(self):
        return len(self.sequence_strs)

    def __getitem__(self, idx):
        return self.sequence_strs[idx]

class ShardedMLMDataset(IterableDataset):
    def __init__(self, crop_len, tokens_per_batch, split):
        assert split in {'train', 'val'}
        # self.data = glob.glob(TRAIN_SHARDS) if split == 'train' else [VAL_DATA]
        self.data = [TOY_TRAIN_DATA] if split == 'train' else [VAL_DATA]
        self.crop_len = crop_len
        self.tokens_per_batch = tokens_per_batch
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        self.collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer)
        
    def __iter__(self):
        random.shuffle(self.data) # new order each epoch
        
        buf, max_len = [], 0
        for shard in self.data:
            ds = FastaSequences.from_file(shard)
            
            if len(self.data) > 1:
                random.shuffle(ds.sequence_strs)
            
            for seq in ds:
                # random crop if sequence too long
                if len(seq) > self.crop_len:
                    start = random.randrange(len(seq) - self.crop_len + 1)
                    seq   = seq[start:start + self.crop_len]
                buf.append((seq))
                # account for special token <cls>
                max_len = max(max_len, len(seq) + 1)
                if max_len * len(buf) >= self.tokens_per_batch:
                    yield self._make_batch(buf)
                    buf, max_len = [], 0
        if buf:
            yield self._make_batch(buf)

    def _make_batch(self, sequences):
        # Build examples: no <eos>, only <cls> + seq, collator will pad to longest
        examples = []
        for seq in sequences:
            # tokenize without special tokens
            enc = self.tokenizer(seq, add_special_tokens=False)
            ids = enc["input_ids"]  # a list of token IDs
            # prepend <cls>
            ids = [self.tokenizer.cls_token_id] + ids
            examples.append({"input_ids": ids})
        # DataCollator will dynamically pad and apply MLM masking
        batch = self.collator(examples)
        return batch["input_ids"], batch["labels"], batch["attention_mask"]

