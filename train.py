import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import math
import os
import multiprocessing

from model import MiniGptConfig, MiniGpt

def main():
    # -----------------------
    # 1. Optimizations & Setup
    # -----------------------
    # Enable TF32 for faster matrix multiplication on Ampere+ GPUs (RTX 30xx, 40xx, A100)
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # 2. Tokenizer
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # -----------------------
    # 3. Load & Process Data    
    # -----------------------
    print("Loading WikiText-103 dataset...")
    # Load WikiText-103 (High-quality Wikipedia articles)
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    # Filter out empty lines/headers and select 130,000 samples
    print("Filtering and selecting 130k samples...")
    dataset = dataset.filter(lambda x: len(x['text']) > 100)
    dataset = dataset.select(range(130000))

    dataset = dataset.shuffle(seed=42)
    print(f"Total dataset size: {len(dataset)}")

    # --- OPTIMIZATION: Pre-tokenize whole dataset ---
    print("Tokenizing dataset (this speeds up training)...")
    
    def tokenize_function(examples):
        # Tokenize a batch of texts
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        # Duplicate input_ids to labels
        result["labels"] = result["input_ids"].copy()
        return result

    # batched=True is crucial for speed here
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"] 
    )
    
    # Set format to pytorch tensors so DataLoader yields tensors directly
    tokenized_dataset.set_format("torch")

    # Split
    train_size = int(0.8 * len(tokenized_dataset))
    val_size = len(tokenized_dataset) - train_size
    train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])

    # --- OPTIMIZATION: DataLoader with workers & pin_memory ---
    # On Windows, num_workers > 0 requires the script to be in 'if __name__ == "__main__":'
    num_workers = min(4, os.cpu_count() or 1)
    
    print(f"Using {num_workers} dataloader workers.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Increased batch size (AMP allows larger batches)
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")

    # -----------------------
    # 4. Model Setup
    # -----------------------
    config = MiniGptConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=8,
        n_heads=8,
        d_model=512,
        d_ff=2048
    )

    model = MiniGpt(config).to(device)
    
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    # -----------------------
    # 5. Training with AMP (Mixed Precision)
    # -----------------------
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler() # Initialize GradScaler for AMP

    epochs = 3
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in loop:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()

            # --- OPTIMIZATION: Mixed Precision ---
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            # Backprop with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    logits = model(input_ids)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Perplexity: {perplexity:.4f}")

        # ---- Save Best ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model_large.pt")
            print("Saved best model.\n")

if __name__ == "__main__":
    main()
