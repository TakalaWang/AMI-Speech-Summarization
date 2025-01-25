from unsloth import FastLanguageModel
from transformers import TextStreamer
from datasets import load_dataset
from evaluate import load

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "TakalaWang/unsloth-base-llama-3.1-8B-finetune", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 6000,
    dtype = "bfloat16",
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


alpaca_prompt = """
Summarize the following transcript in meeting scenario.

### Transcript:
{}

### Summary:
{}"""

def generate_text( transcript):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                transcript, # input
                "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    output = model.generate(**inputs, max_new_tokens = 200)
    summary = tokenizer.batch_decode(output)
    return summary

test_dataset = load_dataset("TakalaWang/AMI_WHISPER_ASR", split="test")

predictions = []
references = []
for data in test_dataset:
    transcript = data["transcript_with_speaker"]
    summary = generate_text(transcript)[0]
    summary = summary.split("### Summary:")[1].strip()
    
    predictions.append(summary)
    references.append(data["summary"])
    print(summary)

# count rouge score
rouge = load("rouge")
print(predictions)
print(references)
rouge_score = rouge.compute(predictions = predictions, references = references)
print(rouge_score)
