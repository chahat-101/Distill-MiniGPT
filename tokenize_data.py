import torch 
from torch.utils.data import Dataset,DataLoader,random_split
from datasets import load_dataset
from transformers import AutoTokenizer
from load_data import load_jsonl

def format_example(example):
    return(
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:\n"
        f"{example['output']}\n"
    )


class InstructionDataset(Dataset):

    def __init__(self,tokenizer,max_length = 512,split = "train",limit = 10000):
        self.tokenizer = tokenizer
        self.max_length = max_length

        dataset = load_dataset("tatsu-lab/alpaca",split=split)

        if limit:
            dataset = dataset.select(range(limit))

        self.texts = [format_example(ex) for ex in dataset]

    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self,idx):
        encoding  = self.tokenizer(
            self.texts[idx],
            max_length = self.max_length,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        labels = input_ids.clone()

        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "labels" : labels
        }



if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = InstructionDataset(r"data/data.jsonl",tokenizer)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset , val_dataset = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(train_dataset,batch_size=2,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=2,shuffle=True)


    batch = next(iter(train_loader))

    print(f"batch input_ids shape:",batch["input_ids"].shape)
    print(f"batch attention mask shape:",batch["attention_mask"].shape)
    print(f"batch labels shape:",batch["labels"].shape)