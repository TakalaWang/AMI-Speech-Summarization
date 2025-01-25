import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from trl import SFTTrainer
import os
from huggingface_hub import login
from dotenv import load_dotenv
from evaluate import load

load_dotenv()
login(token=os.environ['HF_TOKEN'])

model_id="meta-llama/Meta-Llama-3.1-8B"
output_model="TakalaWang/unsloth-llama-3.1-8B-finetune"


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
model.config.use_cache=False
model.config.pretraining_tp=1

train_dataset = load_dataset("TakalaWang/AMI_WHISPER_ASR", split="train")
eval_dataset = load_dataset("TakalaWang/AMI_WHISPER_ASR", split="valid")

def format_dataset(dataset):
    def formatted_train(input, response)->str:
        return f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
    dataset = dataset.map(lambda x: {"text": formatted_train(x["transcript_with_speaker"], x["summary"])})
    return dataset

train_dataset = format_dataset(train_dataset)
eval_dataset = format_dataset(eval_dataset)

peft_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

# training arguments and save best model
training_arguments = TrainingArguments(
    output_dir=output_model,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    num_train_epochs=3,
    max_steps=150,
    fp16=True,
    push_to_hub=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_arguments,
    tokenizer=tokenizer
)

trainer.train()

trainer.push_to_hub()
