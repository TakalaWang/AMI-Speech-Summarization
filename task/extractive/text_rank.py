import os
from gensim.summarization import summarize
from evaluate import load


def get_split(split_path):
    split = {}
    for split_name in ['train', 'valid', 'test']:
        with open(os.path.join(split_path, f"{split_name}.txt")) as f:
            split[split_name] = f.read().splitlines()
    return split

def text_rank(text):
    return summarize(text)

def evaluate(text):
    rouge = load("rouge")
    return rouge.compute(predictions=[text], references=[text])


def main():
    split_path = "/datas/store162/takala/ami/dataset/split"
    dataset_path = "/datas/store162/takala/ami/dataset/"
    result_path = "/datas/store162/takala/ami/results/extractive/text_rank"
    
    split = get_split(split_path)
    rouge = load("rouge")
    
    for split_name in ['train', 'valid', 'test']:
        predictions = []
        references = []
        os.makedirs(os.path.join(result_path, split_name), exist_ok=True)
        for meeting in split[split_name]:
            with open(os.path.join(dataset_path, "dialog", split_name, f"{meeting}"), "r") as f:
                dialogs = f.read()
            summary = text_rank(dialogs)
            with open(os.path.join(dataset_path, "extractive", split_name, f"{meeting}"), "r") as f:
                extractive = f.read()
            with open(os.path.join(result_path, split_name, f"{meeting}"), "w") as f:
                f.write(summary)
            predictions.append(summary)
            references.append(extractive)
            
        rouge_score = rouge.compute(predictions=predictions, references=references)
        print(split_name, "ROUGE:", rouge_score)
        
if __name__ == "__main__":
    main()
