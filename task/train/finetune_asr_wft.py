import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from wft import WhisperFineTuner

load_dotenv()
login(token=os.environ["HF_TOKEN"])

print(torch.cuda.is_available())

id = "ami-whisper-large-test-1"
org = "TakalaWang"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ft = (
    WhisperFineTuner(id, org)
    .set_baseline("openai/whisper-large", language="en", task="transcribe")
    .prepare_dataset(
        "TakalaWang/AMI_ASR",
        src_audio_column="audio",
        src_transcription_column="text",
        num_proc=1,
    )
    .train()
    .merge_and_push()
)
