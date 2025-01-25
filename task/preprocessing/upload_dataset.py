import os
import json
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "TakalaWang/AMI_WHISPER_ASR"
DATA_ROOT = "../transcription_data"


def load_split_files(split_dir="split"):
    """Load train, valid, and test splits from text files."""
    splits = {}
    for split_name in ["train", "valid", "test"]:
        file_path = os.path.join(split_dir, f"{split_name}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Split file {file_path} not found.")
        with open(file_path, "r") as f:
            splits[split_name] = f.read().splitlines()
    return splits


def process_meeting_file(file_path):
    """Process a single meeting JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    transcript = "\n".join(
        [sentence['text'] for sentence in data["transcript"]]
    )
    transcript_with_speaker = "\n".join(
        [f"{sentence['speaker']}: {sentence['text']}" for sentence in data["transcript"]]
    )
    return {
        "meeting_id": os.path.splitext(os.path.basename(file_path))[0],
        "transcript": transcript,
        "transcript_with_speaker": transcript_with_speaker,
        "summary": data["summary"],
    }


def create_dataset(splits, data_root):
    """Create a DatasetDict from split definitions and JSON files."""
    all_data = DatasetDict()
    for split, meetings in splits.items():
        split_data = []
        for meeting_id in meetings:
            json_path = os.path.join(data_root, split, f"{meeting_id}.json")
            if not os.path.exists(json_path):
                print(f"Warning: File {json_path} not found. Skipping.")
                continue
            split_data.append(process_meeting_file(json_path))

        all_data[split] = Dataset.from_dict({
            key: [item[key] for item in split_data] for key in ["meeting_id", "transcript", "transcript_with_speaker", "summary"]
        })
    return all_data


def push_to_hub(dataset, repo_id, token):
    """Push a DatasetDict to the Hugging Face Hub."""
    dataset.push_to_hub(repo_id, token=token)


def main():

    splits = load_split_files()

    dataset = create_dataset(splits, DATA_ROOT)

    push_to_hub(dataset, REPO_ID, HF_TOKEN)


if __name__ == "__main__":
    main()
