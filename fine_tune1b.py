# Copyright (c) 2025 CMS Manhattan
# All rights reserved.
# Author: Konstantin Vladimirovich Grabko
# Email: grabko@cmsmanhattan.com
# Phone: +1(516)777-0945
#
# MIT License
#
# Copyright (c) 2025 Konstantin Grabko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle  # <-- ONLY PICKLE, NOT NUMPY!

# ========================================
# YOUR PATHS — DO NOT TOUCH
# ========================================
TOKENIZER_PATH = Path("/home/kgrabko/jirackkit/src/main/python/tokenizer/tokenizer.json")
DATASET_PATH = Path("/home/kgrabko/jirackkit/dataset/dialogues_text_clean.txt")
TOKENS_CACHE = Path("/mnt/data/build/tokens_cache.bin")  # on the flash drive

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LEN = 1024
EPOCHS = 3
LR = 5e-5

# ========================================
# Your tokenizer
# ========================================
from tokenizers import Tokenizer

print("Loading your tokenizer...")
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
print(f"Vocab size: {tokenizer.get_vocab_size()}")

def tokenize_line(line: str):
    encoded = tokenizer.encode(line.strip())
    return encoded.ids

# ========================================
# Model
# ========================================
from JiRackPyTorch_GPT5_class_1b import JiRackPyTorch

# ========================================
# Tokenization with cache
# ========================================
def load_and_tokenize_dataset():
    if TOKENS_CACHE.exists():
        print(f"Loading cache from flash drive: {TOKENS_CACHE} ({TOKENS_CACHE.stat().st_size / 1e9:.2f} GB)")
        with open(TOKENS_CACHE, "rb") as f:
            train_ids, val_ids = pickle.load(f)
        print(f"Cache loaded: {len(train_ids):,} train + {len(val_ids):,} val")
        return train_ids, val_ids

    print("No cache — tokenizing from scratch (ONLY ONCE)...")
    print(f"Reading file: {DATASET_PATH} ({DATASET_PATH.stat().st_size / 1e9:.2f} GB)")

    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines):,}")

    num_workers = cpu_count()
    print(f"Tokenization on {num_workers} cores...")
    with Pool(num_workers) as pool:
        tokenized = list(tqdm(pool.imap(tokenize_line, lines), total=len(lines), desc="Tokenizing lines"))

    tokenized = [seq for seq in tokenized if len(seq) > 1]
    import random
    random.shuffle(tokenized)

    val_size = int(len(tokenized) * 0.01)
    val_ids = tokenized[:val_size]
    train_ids = tokenized[val_size:]

    print(f"Tokenization completed: {len(train_ids):,} train, {len(val_ids):,} val")

    TOKENS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cache to flash drive: {TOKENS_CACHE} (will be 8–15 GB)...")
    with open(TOKENS_CACHE, "wb") as f:
        pickle.dump((train_ids, val_ids), f)
    print(f"Cache saved! Size: {TOKENS_CACHE.stat().st_size / 1e9:.2f} GB")

    return train_ids, val_ids

# ========================================
# Dataset
# ========================================
class SimpleDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        return torch.tensor(seq, dtype=torch.long)

# ========================================
# Main
# ========================================
def main():
    print(f"Using device: {DEVICE}. AMP (FP16) {'enabled' if USE_AMP else 'disabled'}")

    print(f"[DEBUG VRAM before model creation] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    start_time = time.time()
    model = JiRackPyTorch()
    print(f"[DEBUG] Model created on CPU: {time.time() - start_time:.2f}s")
    print(f"[DEBUG] Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B")

    print("Starting from scratch — random weights (Skipping state_dict load for stability test!)")

    model.to(DEVICE)
    print(f"[DEBUG] Model moved to cuda: {time.time() - start_time:.2f}s")
    print(f"[DEBUG VRAM after model.to(device)] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    print("[DEBUG] Tokenizing and caching...")
    train_ids, val_ids = load_and_tokenize_dataset()

    train_dataset = SimpleDataset(train_ids)
    val_dataset = SimpleDataset(val_ids)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler() if USE_AMP else None

    model.train()

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            batch = batch.to(DEVICE)

            with autocast(enabled=USE_AMP):
                logits = model(batch)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss / GRAD_ACCUM_STEPS

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Avg loss: {avg_loss:.4f}")

        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"jirack_1b_epoch{epoch+1}.pt")

    print("Training completed!")

if __name__ == "__main__":
    main()