import torch
from torch.utils.data import DataLoader

from data import ShardedMLMDataset
from model import ESM, ESMConfig

T = 255   # 256 with cls token
TOKENS_PER_BATCH = 2048

train_loader = DataLoader(ShardedMLMDataset(crop_len=T, tokens_per_batch=TOKENS_PER_BATCH, split='train'), batch_size=None)
val_loader = DataLoader(ShardedMLMDataset(crop_len=T, tokens_per_batch=TOKENS_PER_BATCH, split='val'), batch_size=None)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

model = ESM(ESMConfig())
model.to(device)

# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i, batch in enumerate(train_loader):
    inputs, labels, mask = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    optimizer.zero_grad()
    logits, loss = model(inputs, mask, labels)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.detach().item()}")
    if i >= 49:
        break
