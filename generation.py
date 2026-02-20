import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import MiniGpt,MiniGptConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = MiniGptConfig(vocab_size=tokenizer.vocab_size)

model = MiniGpt(config).to(device)
model_distill = MiniGpt(config).to(device)

model_distill.load_state_dict(torch.load("checkpoints/best_model_distilled.pt",map_location=device))
model.load_state_dict(torch.load("checkpoints/best_model.pt",map_location=device))
model.eval()
model_distill.eval()


def generate(model, prompt, sampling: int, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)

        if hasattr(logits, 'logits'):
            next_token_logits = logits.logits[:, -1, :]
        else:
            next_token_logits = logits[:, -1, :]

        if sampling == 0:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            k = 50
            topk_logits, topk_indices = torch.topk(next_token_logits, k)
            probs = torch.softmax(topk_logits, dim=-1)
            next_token = topk_indices.gather(-1, torch.multinomial(probs, 1))

        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    prompt = "### Instruction:\nExplain machine learning.\n\n### Response:\n"
    output = generate(model,prompt, max_length=50,sampling=1)

    print("--- Original Model Output ---")
    print(output)

    output = generate(model_distill,prompt, max_length=50,sampling=1)   
    print("\n--- Distilled Model Output ---")

    print(output)   