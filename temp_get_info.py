
import torch
from transformers import AutoTokenizer

from model import MiniGpt, MiniGptConfig
from tokenize_data import InstructionDataset

# --- 1. Calculate Model Parameters ---

# Initialize tokenizer to get vocab_size, which is needed for the model config
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Initialize the model configuration
config = MiniGptConfig(vocab_size=tokenizer.vocab_size)

# Instantiate the model
model = MiniGpt(config)

# Calculate the total number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total number of model parameters: {total_params}")

# --- 2. Calculate Training Data Size ---

# Load the full dataset as it's defined in train.py
# The limit is hardcoded to 10000 in train.py
dataset = InstructionDataset(tokenizer, split="train", limit=10000)

# The training script uses 80% of the dataset for training
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

print(f"Total dataset size (before split): {len(dataset)}")
print(f"Size of the training data (80%): {train_size}")
print(f"Size of the validation data (20%): {val_size}")
