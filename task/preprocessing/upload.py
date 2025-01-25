import os
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
def get_split(split_path):
    split = {}
    for split_name in ['train', 'valid', 'test']:
        with open(os.path.join(split_path, f"{split_name}.txt")) as f:
            split[split_name] = f.read().splitlines()
    return split

def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    REPO_ID = "TakalaWang/AMI"
    DATA_ROOT = "/datas/store162/takala/ami/dataset"

    split = get_split(os.path.join(DATA_ROOT, "split"))
    dataset = DatasetDict()
    for split_name in ["train", "valid", "test"]:
        meeting_ids = []
        abstractives = []
        extractives = []
        dialogs = []
        for meeting_id in split[split_name]:
            meeting_ids.append(meeting_id)
            with open(os.path.join(DATA_ROOT, "abstractive", split_name, meeting_id), "r") as f:
                abstractive = f.read()
            with open(os.path.join(DATA_ROOT, "extractive", split_name, meeting_id), "r") as f:
                extractive = f.read()
            with open(os.path.join(DATA_ROOT, "dialog", split_name, meeting_id), "r") as f:
                dialog = f.read()
            abstractives.append(abstractive)
            extractives.append(extractive)
            dialogs.append(dialog)
        dataset[split_name] = Dataset.from_dict({
            "meeting_id": meeting_ids,
            "abstractive": abstractives,
            "extractive": extractives,
            "dialog": dialogs
        })
    
    dataset.push_to_hub(REPO_ID, token=HF_TOKEN)

if __name__ == "__main__":
    main()