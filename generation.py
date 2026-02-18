import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import MiniGpt,MiniGptConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = MiniGptConfig(vocab_size=tokenizer.vocab_size)

model = MiniGpt(config).to(device)

model.load_state_dict(torch.load("checkpoints/best_model.pt",map_location=device))
model.eval()

def generate(prompt,max_length = 50):

    input_ids = tokenizer.encode(prompt,return_tensors = "pt").to(device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[:,-1,:]
        next_token = torch.argmax(next_token_logits,dim = -1)

        next_token = next_token.unsqueeze(-1)

        input_ids = torch.cat([input_ids,next_token],dim =1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])

if __name__ == "__main__":
    prompt = "### Instruction:\nExplain machine learning.\n\n### Response:\n"
    output = generate(prompt, max_length=50)
    print("\nGenerated Output:\n")
    print(output)