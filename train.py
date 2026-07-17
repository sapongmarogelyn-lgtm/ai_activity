import os
import torch
import time
from model import MiniGPT

# =========================
# Hyperparameters
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU-optimized settings.
# Good default for AMD Ryzen 3 PRO 2200G with 32GB RAM.
if device == "cpu":
    batch_size = 4  # Lower to 8 if training becomes too slow
    block_size = 448  # Must cover the longest Question+Answer entry (was 96 - too small!)
    max_iters = 3000  # More iterations help the small model learn better
    eval_interval = 500  
    learning_rate = 3e-4
    eval_iters = 10  # Reasonable evaluation
    
    n_embd = 96  # Must be divisible by n_head
    n_head = 4
    n_layer = 3
    dropout = 0.1
    print(f"Using device: {device} (CPU-optimized but capable)")
    print(f"Settings: batch={batch_size}, context={block_size}, embd={n_embd}, layers={n_layer}, iters={max_iters}")
else:
    # GPU settings (original larger model)
    batch_size = 64
    block_size = 448
    max_iters = 15000
    eval_interval = 500  
    learning_rate = 3e-4
    eval_iters = 100
    
    n_embd = 256
    n_head = 8
    n_layer = 6
    dropout = 0.1
    print(f"Using device: {device} (GPU-optimized settings)")

torch.manual_seed(1337)

# =========================
# Load text as discrete entries (instead of one long blob)
# =========================
with open("data/train.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split on blank lines -> one entry per Question/Answer pair
entries = [e.strip() + "\n\n" for e in raw_text.split("\n\n") if e.strip()]

# =========================
# Build vocabulary (from the whole text, so nothing is missed)
# =========================
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

# Encode each entry separately so it stays intact as one unit
encoded_entries = [encode(e) for e in entries]

if len(encoded_entries) < 2:
    raise ValueError(
        "Need at least 2 Q&A entries in data/train.txt (separated by a blank line) "
        "to create a train/val split."
    )

# train / val split by ENTRY, not by raw character offset
# (an entry can no longer get sliced in half across the split)
n_entries = len(encoded_entries)
n_train = max(1, int(0.9 * n_entries))
train_entries = encoded_entries[:n_train]
val_entries = encoded_entries[n_train:]
if len(val_entries) == 0:
    # guarantee val is never empty even with a small dataset
    val_entries = encoded_entries[-2:]
    train_entries = encoded_entries[:-2] or encoded_entries[:1]

longest_entry = max(len(e) for e in encoded_entries)
if longest_entry > block_size:
    raise ValueError(
        f"block_size={block_size} is smaller than the longest entry ({longest_entry} chars). "
        f"Increase block_size to at least {longest_entry}."
    )

print(f"Loaded {n_entries} Q&A entries | train={len(train_entries)} val={len(val_entries)} | "
      f"longest entry={longest_entry} chars | block_size={block_size}")

def pad_or_truncate(tokens, size):
    if len(tokens) >= size:
        return tokens[:size]
    return tokens + [stoi["\n"]] * (size - len(tokens))  # pad with newline token

def get_batch(split):
    source = train_entries if split == "train" else val_entries
    batch = []
    for _ in range(batch_size):
        entry = source[torch.randint(len(source), (1,)).item()]
        batch.append(pad_or_truncate(entry, block_size + 1))  # +1 for the y shift
    batch = torch.tensor(batch, dtype=torch.long)
    x = batch[:, :block_size].contiguous()
    y = batch[:, 1:block_size + 1].contiguous()
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out

# =========================
# Create model
# =========================
model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# =========================
# Train loop
# =========================
print(f"\nStarting training for {max_iters} iterations...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"Will evaluate every {eval_interval} steps\n")

start_time = time.time()
for step in range(max_iters):
    # Evaluate at intervals (skip step 0 to start training immediately)
    if step > 0 and (step % eval_interval == 0 or step == max_iters - 1):
        eval_start = time.time()
        losses = estimate_loss(model)
        eval_time = time.time() - eval_start
        elapsed = time.time() - start_time
        print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | time {elapsed:.1f}s | eval {eval_time:.1f}s")

    # Show progress every 50 steps (without evaluation)
    elif step % 50 == 0:
        elapsed = time.time() - start_time
        print(f"step {step:4d} | training... | time {elapsed:.1f}s")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nTraining completed in {time.time() - start_time:.1f}s")

# =========================
# Save checkpoint
# =========================
os.makedirs("checkpoints", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "config": {
        "n_embd": n_embd,
        "block_size": block_size,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }
}

torch.save(checkpoint, "checkpoints/mini_llm.pt")
print("Saved checkpoint to checkpoints/mini_llm.pt")

# =========================
# Quick test generation
# =========================
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=300, temperature=0.9)[0].tolist()

print("\n=== SAMPLE OUTPUT ===\n")
print(decode(generated))