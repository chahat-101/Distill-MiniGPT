import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from transformers import AutoTokenizer
from tqdm import tqdm

from model import MiniGpt,MiniGptConfig
from tokenize_data import InstructionDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

dataset = InstructionDataset(r"data\data.jsonl",tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset,val_dataset = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset,shuffle=True,batch_size=2)
val_loader = DataLoader(val_dataset,shuffle=True,batch_size=2)

config = MiniGptConfig(vocab_size=tokenizer.vocab_size)

model = MiniGpt(config).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(),lr = 3e-4)



epochs = 5
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    train_loop = tqdm(train_loader,leave=True)

    for batch in train_loop:

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)

        shift_logits =  logits[:,:-1,:].contiguous()
        shift_labels =  labels[:,1:].contiguous()

        loss = criterion(
            shift_logits.view(-1,shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_loop.set_description(F"Epoch {epoch}")
        train_loop.set_postfix(train_loss = loss.item())

    avg_train_loss = total_loss/len(train_loader)

    print(f"Epoch {epoch} loss: {avg_train_loss}")

    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch["labels"].to(device)

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

    # ------------------
    # Save Best Model
    # ------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print("Saved best model.\n")