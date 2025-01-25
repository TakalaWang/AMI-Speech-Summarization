import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor, 
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 登入 Hugging Face
login(token=os.getenv("HF_TOKEN"))

# 設定基本參數
base_model = "openai/whisper-large-v3"
dataset_name = f"TakalaWang/AMI_ASR"
model_id = f"TakalaWang/ami-whisper-large-finetune"
audio_column_name = "audio"
text_column_name = "text"

# 添加音頻處理配置
audio_max_length = 30  # 最大音頻長度（秒）
sampling_rate = 16000  # 採樣率
max_input_length = audio_max_length * sampling_rate  # 計算最大輸入長度

# 載入數據集 前100 筆
train_dataset = load_dataset(dataset_name, split="train")
eval_dataset = load_dataset(dataset_name, split="validation")

# 載入 processor 和模型
processor = AutoProcessor.from_pretrained(base_model)
model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model)

# 設定 LoRA 配置
peft_config = LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# 將模型轉換為 PEFT 模型
model = get_peft_model(model, peft_config)
model = model.to("cuda")

# 數據預處理函數
def prepare_dataset(batch):
    audio = batch[audio_column_name]

    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch[text_column_name]).input_ids
    return batch

# 處理數據集
train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    prepare_dataset,
    remove_columns=eval_dataset.column_names,
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int = 50258  # Whisper 的默認值

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 設定訓練參數
training_args = Seq2SeqTrainingArguments(
    output_dir="finetune/ami-whisper-large-finetune",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-3,
    warmup_steps=50,
    max_steps=1000,
    fp16=True,
    save_steps=200,
    logging_steps=25,
    push_to_hub=True,
    hub_model_id=model_id,
    report_to=["tensorboard"],
    logging_dir="./logs/whisper_logs",
    run_name=None,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=3,
    resume_from_checkpoint=True,
)

def compute_metrics(pred):
    wer_metric = load("wer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # 解碼預測結果
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

# 初始化訓練器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 開始訓練
trainer.train()

# 合併 LoRA 權重並儲存完整模型
model = model.merge_and_unload()

# 儲存完整模型、tokenizer 和 processor 到本地
model.save_pretrained("./final_model")
processor.save_pretrained("./final_model")

# 推送完整模型到 Hugging Face Hub
model.push_to_hub(model_id)
processor.push_to_hub(model_id)