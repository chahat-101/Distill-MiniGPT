import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import MiniGpt, MiniGptConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = MiniGptConfig(vocab_size=tokenizer.vocab_size)

model = MiniGpt(config).to(device)
model_distill = MiniGpt(config).to(device)

model_distill.load_state_dict(
    torch.load("checkpoints/best_model_distilled.pt", map_location=device)
)
model.load_state_dict(
    torch.load("checkpoints/best_model_large.pt", map_location=device)
)
model.eval()
model_distill.eval()


def generate(model, prompt, sampling: int, max_length=150):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)

        if hasattr(logits, "logits"):
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
    if __name__ == "__main__":
        print("Welcome to MiniGPT Inference!")
        print("Format: Just type your prompt (e.g., 'The Taj Mahal is')")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("Prompt > ").strip()
            if user_input.lower() == 'exit':
                break
            if not user_input:
                continue

            # Using raw prompt because the model was trained on WikiText (Wikipedia)
            # and NOT instruction-tuned.
            formatted_prompt = user_input 

            print("\n--- Original Model ---")
            output_large = generate(model, formatted_prompt, max_length=100, sampling=1)
            print(output_large)

            print("\n--- Distilled Model ---")
            output_distill = generate(model_distill, formatted_prompt, max_length=100, sampling=1)
            print(output_distill)
            print("-" * 40 + "\n")
        print("-" * 40 + "\n")
