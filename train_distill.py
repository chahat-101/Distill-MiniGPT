import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from transformers import AutoTokenizer,AutoModelForCausalLM
from tokenize_data import InstructionDataset
from tqdm import tqdm
import math

from model import MiniGpt,MiniGptConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = InstructionDataset(tokenizer=tokenizer,split = "train",limit=10000)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset,val_dataset = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset,batch_size=2,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size = 2)

config = MiniGptConfig(vocab_size=tokenizer.vocab_size)
student = MiniGpt(config).to(device)

teacher = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
teacher.eval()

for param in teacher.parameters():
    param.requires_grad = False

ce_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = torch.optim.AdamW(student.parameters(), lr = 3e-4)

temperature = 2.0
alpha = 0.5
epochs = 3
best_val_loss = float('inf')



for epoch in range(epochs):
    student.train()
    total_loss = 0

    loop = tqdm(train_loader)


    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        student_logits = student(input_ids)

        with torch.no_grad():
            teacher_outputs = teacher(input_ids)
            teacher_logits = teacher_outputs.logits

        shift_student_logits = student_logits[:, :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()


        ce_loss = ce_loss_fn(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1)
        )

        student_soft = F.log_softmax(shift_student_logits.view(-1, shift_student_logits.size(-1)) / temperature, dim=-1)
        teacher_soft = F.softmax(shift_teacher_logits.view(-1, shift_teacher_logits.size(-1)) / temperature, dim=-1)

        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean"
        ) * (temperature ** 2)

        loss = alpha * kl_loss + (1 - alpha) * ce_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())


    student.eval()   
    total_val_loss = 0

    with torch.no_grad():
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            student_logits = student(input_ids)

            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits

            shift_student = student_logits[:, :-1, :].contiguous().view(-1, student_logits.size(-1))
            shift_teacher = teacher_logits[:, :-1, :].contiguous().view(-1, teacher_logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)


            ce_loss = ce_loss_fn(shift_student, shift_labels)

            student_soft = F.log_softmax(shift_student / temperature, dim=-1)
            teacher_soft = F.softmax(shift_teacher / temperature, dim=-1)
            kl_loss = F.kl_div(
                student_soft,
                teacher_soft,
                reduction="batchmean"
            ) * (temperature ** 2)

            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # ------------------
    # Save Best Model
    # ------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(student.state_dict(), "checkpoints/best_model_distilled.pt")
        print("Saved best distilled model.\n")

    print(f"\nEpoch {epoch} Distill Loss: {total_loss / len(train_loader)}")

