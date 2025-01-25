import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# 登入 Hugging Face
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# 設定參數
base_model = "facebook/bart-large"
dataset_name = "TakalaWang/AMI_ABSTRACTIVE"
model_id = "TakalaWang/ami-bart-large-finetune"
input_column = "transcript"
target_column = "summary"

# 載入數據集
train_dataset = load_dataset(dataset_name, split="train")
eval_dataset = load_dataset(dataset_name, split="validation")

# 載入 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
normalizer = BasicTextNormalizer()

# 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def prepare_dataset(batch):
    # dataset preprocessing
    batch[input_column] = batch[input_column].lower().strip()
    batch[target_column] = batch[target_column].lower().strip()
    
    batch[input_column] = normalizer(batch[input_column])
    batch[target_column] = normalizer(batch[target_column])

    processor_batch = tokenizer(batch[input_column], max_length=1024, padding="max_length", truncation=True)
    processor_batch["labels"] = tokenizer(batch[target_column], max_length=256, padding="max_length", truncation=True)["input_ids"]
    
    return processor_batch
    
# 處理數據集
train_dataset = train_dataset.map(prepare_dataset, remove_columns=["transcript", "summary"])
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=["transcript", "summary"])

# 設定訓練參數
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    fp16=True,
    evaluation_strategy="epoch",
    save_steps=500,
    push_to_hub=True,
    hub_model_id=model_id,
    logging_dir="./logs",
    report_to=["tensorboard"]
)

# 初始化訓練器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # 使用 DataCollatorForSeq2Seq
)

# 開始訓練
trainer.train()

# 合併 LoRA 權重並儲存完整模型
model = model.merge_and_unload()

# 儲存完整模型到本地
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

# 推送完整模型到 Hugging Face Hub
model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)

