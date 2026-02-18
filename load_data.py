import json

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_example(example):
    return (
        "<bos>\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:\n"
        f"{example['response']}\n"
        "<eos>"
    )


if __name__ == "__main__":
    dataset = load_jsonl("data/data.jsonl")
    print("Total samples:", len(dataset))
    # print("First sample:", dataset[0])

    for (i,data) in enumerate(dataset):
        print(i)
        print(format_example(data))



