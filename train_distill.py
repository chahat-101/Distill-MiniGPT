import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import math
import os

from model import MiniGptConfig, MiniGpt


# ==============================
# Device
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==============================
# Tokenizer
# ==============================

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# ==============================
# Load & Combine Datasets
# ==============================

print("Loading Alpaca...")
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
alpaca = alpaca.select(range(20000))

print("Loading Dolly...")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
dolly = dolly.select(range(15000))


def format_alpaca(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }


def format_dolly(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }


alpaca = alpaca.map(format_alpaca, remove_columns=alpaca.column_names)
dolly = dolly.map(format_dolly, remove_columns=dolly.column_names)

dataset = concatenate_datasets([alpaca, dolly])
dataset = dataset.shuffle(seed=42)

print("Total dataset size:", len(dataset))


# ==============================
# Torch Dataset
# ==============================


class InstructionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


torch_dataset = InstructionDataset(dataset, tokenizer)

train_size = int(0.8 * len(torch_dataset))
val_size = len(torch_dataset) - train_size

train_dataset, val_dataset = random_split(torch_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))


# ==============================
# Student Model
# ==============================

config = MiniGptConfig(
    vocab_size=tokenizer.vocab_size, n_layers=8, n_heads=8, d_model=512, d_ff=2048
)

student = MiniGpt(config).to(device)

print("Student parameters:", sum(p.numel() for p in student.parameters()))


# ==============================
# Teacher Model
# ==============================

teacher = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
teacher.eval()

for param in teacher.parameters():
    param.requires_grad = False


# ==============================
# Loss & Optimizer
# ==============================

ce_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)

temperature = 2.0
alpha = 0.5
epochs = 3

best_val_loss = float("inf")
os.makedirs("checkpoints", exist_ok=True)


# ==============================
# Training Loop
# ==============================

for epoch in range(epochs):
    # -------- TRAIN --------
    student.train()
    total_train_loss = 0

    loop = tqdm(train_loader)

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward
        student_logits = student(input_ids, attention_mask=attention_mask)

        with torch.no_grad():
            teacher_logits = teacher(input_ids, attention_mask=attention_mask).logits

        # Shift for next-token prediction
        shift_student = student_logits[:, :-1, :]
        shift_teacher = teacher_logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # CE Loss
        ce_loss = ce_loss_fn(
            shift_student.reshape(-1, shift_student.size(-1)), shift_labels.reshape(-1)
        )

        # KL Loss with padding mask
        mask = shift_labels != tokenizer.pad_token_id

        student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)

        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)

        kl = (kl * mask).sum() / mask.sum()
        kl *= temperature**2

        # Combined Loss
        loss = alpha * kl + (1 - alpha) * ce_loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)

    # -------- VALIDATION --------
    student.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            student_logits = student(input_ids, attention_mask=attention_mask)
            teacher_logits = teacher(input_ids, attention_mask=attention_mask).logits

            shift_student = student_logits[:, :-1, :]
            shift_teacher = teacher_logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            ce_loss = ce_loss_fn(
                shift_student.reshape(-1, shift_student.size(-1)),
                shift_labels.reshape(-1),
            )

            mask = shift_labels != tokenizer.pad_token_id

            student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)

            kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)

            kl = (kl * mask).sum() / mask.sum()
            kl *= temperature**2

            loss = alpha * kl + (1 - alpha) * ce_loss

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"\nEpoch {epoch}")
    print(f"Train Distill Loss: {avg_train_loss:.4f}")
    print(f"Val Distill Loss:   {avg_val_loss:.4f}")

    # Save best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(student.state_dict(), "checkpoints/best_model_distilled_large.pt")
        print("Saved best distilled model.\n")
