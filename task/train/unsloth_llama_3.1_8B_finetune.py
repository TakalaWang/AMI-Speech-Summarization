from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

max_seq_length = 6000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """
Summarize the following transcript in meeting scenario.

### Transcript:
{}

### Summary:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    transcripts = examples["transcript"]
    summaries      = examples["summary"]
    texts = []
    for transcript, summary in zip(transcripts, summaries):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(transcript, summary) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


train_dataset = load_dataset("TakalaWang/AMI_WHISPER_ASR", split = "train")
train_dataset = train_dataset.map(formatting_prompts_func, batched = True)

eval_dataset = load_dataset("TakalaWang/AMI_WHISPER_ASR", split = "valid")
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3, # Set this for 1 full training run.
        max_steps = 150,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

model.push_to_hub("TakalaWang/unsloth2-llama-3.1-8B-finetune", token = os.getenv("HF_TOKEN")) # Online saving
tokenizer.push_to_hub("TakalaWang/unsloth2-llama-3.1-8B-finetune", token = os.getenv("HF_TOKEN")) # Online saving

