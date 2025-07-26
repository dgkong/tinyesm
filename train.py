import os
import time

import torch
from torch.utils.data import DataLoader

from data import ShardedMLMDataset
from distilled_model import DistilConfig, DistilESM
from model import ESM, ESMConfig

# Configuration
VAL_INTERVAL = 400
VAL_STEPS = 100
PATIENCE = 3
MIN_DELTA = 1e-4
CHECKPOINT_PATH = "checkpoints"
DISTILLATION = True
RESUME_TRAINING = False

# Batching Configuration
total_batch_size = 131072   # 2**17 number of tokens
B, T = 16, 510              # micro batch size, max seq residues length
TOK_PER_BATCH = B * (T + 2) # T + 2 due to <cls> and <eos> tokens
assert total_batch_size % TOK_PER_BATCH == 0, "total_batch_size must be divisible by tok_per_batch"
grad_accum_steps = total_batch_size // TOK_PER_BATCH
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# LR Scheduler Configuration
max_lr = 4e-4
min_lr = max_lr * 0.1
max_steps = 200_000
warmup_steps = 500

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
log_file = os.path.join(CHECKPOINT_PATH, f"log.txt")
if not RESUME_TRAINING:
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

train_loader = DataLoader(
    ShardedMLMDataset(crop_len=T, tokens_per_batch=TOK_PER_BATCH, split='train'), 
    batch_size=None)
val_loader = DataLoader(
    ShardedMLMDataset(crop_len=T, tokens_per_batch=TOK_PER_BATCH, split='val'), 
    batch_size=None)

if DISTILLATION:
    model = DistilESM(DistilConfig())
else:
    model = ESM(ESMConfig())
model.to(device)
model = torch.compile(model, backend="aot_eager")
optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=4e-4)

def get_lr(it):
    # linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # return min learning rate post decay
    if it > max_steps:
        return min_lr
    # linear decay specified in ESM2 paper A.2.4.
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 1.0 - decay_ratio
    return min_lr + coeff * (max_lr - min_lr) 

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'iter': iteration
    }
    torch.save(checkpoint, out)

global_optimizer_step = 0
train_iter = iter(train_loader)
best_val_loss = float('inf')
patience_counter = 0

if RESUME_TRAINING:
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f"final_model.pt"), map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])
    global_optimizer_step = checkpoint['iter']

def run_validation():
    print("Running validation...")
    model.eval()
    val_loss_accum = 0.0
    steps_run = 0
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(VAL_STEPS):
            try:
                inputs, labels, mask = next(val_iter)
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
                if DISTILLATION:
                    _, _, loss = model(inputs, mask, labels)
                else:
                    _, loss = model(inputs, mask, labels)
                val_loss_accum += loss.item()
                steps_run += 1
            except StopIteration:
                print("Validation data loader exhausted before reaching VAL_STEPS.")
                break
    model.train()
    avg_loss = val_loss_accum / steps_run
    return avg_loss

print(f"Starting training for {max_steps} optimizer steps...")

while global_optimizer_step < max_steps:
    # VALIDATION AND EARLY STOPPING
    if (global_optimizer_step > 0 and global_optimizer_step % VAL_INTERVAL == 0):
        val_loss = run_validation()
        print(f"step {global_optimizer_step:4d}/{max_steps} | validation loss: {val_loss:.6f}")
        with open(log_file, "a") as f:
            f.write(f"{global_optimizer_step} val {val_loss:.6f}\n")
        if val_loss < best_val_loss - MIN_DELTA:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model.")
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, global_optimizer_step, os.path.join(CHECKPOINT_PATH, f"best_model.pt"))
        else:
            patience_counter += 1
            print(f"No significant improvement in validation loss. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {patience_counter} validation cycles with no improvement.")
            break
    
    # TRAINING
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    ce_loss_accum = 0.0
    tokens_processed = 0
    for micro_step in range(grad_accum_steps):
        try:
            inputs, labels, mask = next(train_iter)
        except StopIteration:
            print(f"Epoch finished. Restarting data loader for optimizer step {global_optimizer_step} (micro-step {micro_step}/{grad_accum_steps})")
            train_iter = iter(train_loader)
            inputs, labels, mask = next(train_iter)
        inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
        if DISTILLATION:
            _, loss, ce_loss = model(inputs, mask, labels)
            ce_loss /= grad_accum_steps
            ce_loss_accum += ce_loss.item()
        else:
            _, loss = model(inputs, mask, labels)
        loss /= grad_accum_steps
        loss_accum += loss.item()
        loss.backward()
        tokens_processed += inputs.numel()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    current_lr = get_lr(global_optimizer_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    optimizer.step()
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = tokens_processed / dt
    print(f"step {global_optimizer_step:4d}/{max_steps} | loss: {loss_accum:.6f} | ce_loss: {ce_loss_accum:.6f} | lr: {current_lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.0f}")
    with open(log_file, "a") as f:
        f.write(f"{global_optimizer_step} train {loss_accum:.6f} ce {ce_loss_accum:.6f}\n")
    global_optimizer_step += 1

print("Training finished. Saving model.")
save_checkpoint(model, optimizer, global_optimizer_step, os.path.join(CHECKPOINT_PATH, "final_model.pt"))

