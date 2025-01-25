import os, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.environ['HF_TOKEN'])

# ===== 1. Load Pre-trained Model and Tokenizer =====
model_name = "meta-llama/Llama-3.1-8B"  # Replace with the actual model path or name
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

# ===== 2. Configure LoRA =====
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# ===== 3. Load Dataset =====
def format_function(examples):
    return f"### Transcript: {examples['transcript']}\n### Summary:{examples['summary']}"

# load dataset from ../transcription_data/train with multiple files
dataset = load_dataset("TakalaWang/AMI_WHISPER_ASR", streaming=True)

# ===== 4. Define Training Arguments =====
training_args = TrainingArguments(
    output_dir="./llama3_lora_ami_summary",          # Directory for checkpoints
    evaluation_strategy="steps",        # Evaluate periodically
    save_strategy="steps",              # Save checkpoints periodically
    save_steps=10,                      # Save every 10 steps
    eval_steps=10,
    max_steps=100,
    load_best_model_at_end=True,        # Load the best model after training
    metric_for_best_model="eval_loss",  # Metric to determine the best checkpoint
    greater_is_better=False,            # Smaller loss is better
    per_device_train_batch_size=1,      # Adjust based on hardware
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    report_to="tensorboard",
    learning_rate=1e-5,
    fp16=True,
    gradient_checkpointing=True,
    push_to_hub=True,                   # Automatically push to Hugging Face Hub
    hub_model_id="llama3-lora-finetuned-ami-summary",  # Unique repository name
)

# ===== 5. Initialize Trainer =====
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
    formatting_func=format_function,
)

# ===== 6. Train the Model =====
trainer.train()

# ===== 7. Push to Hugging Face Hub =====
trainer.push_to_hub("llama3-lora-finetuned-ami-summary")
tokenizer.push_to_hub("llama3-lora-finetuned-ami-summary")

# ===== 8. Evaluate the Model on the Test Set =====
trainer.evaluate(dataset["test"])