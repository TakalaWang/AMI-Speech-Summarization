import os
import json
from datetime import datetime

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from bert_score import score
from datasets import load_dataset
from rouge_score import rouge_scorer
from huggingface_hub import HfApi, login
from transformers import (
    pipeline, 
    AutoProcessor,
    AutoTokenizer
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import torch

def init_models(model_name: str):
    """初始化模型和工具"""
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        summarizer = pipeline(
            "summarization",
            model=model_name,
            processor=processor,
            model_kwargs={"device_map": "auto"}
        )
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=tokenizer,
            model_kwargs={"device_map": "auto"}
        )
    
    # 初始化評分器
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    normalizer = BasicTextNormalizer()
    
    return summarizer, rouge, normalizer

def chunk_text(text, max_chunk_size=800):
    """將長文本分成較小的塊"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += 1
        
        if current_length >= max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

def process_batch(batch, summarizer, rouge, normalizer, transcript_column: str, summary_column: str):
    """處理單個批次的數據"""

    normalized_article = normalizer(batch[transcript_column])
        
    # 將長文本分成較小的塊
    chunks = chunk_text(normalized_article)
    chunk_summaries = []
    
    # 對每個塊進行摘要
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=130 // len(chunks),
            min_length=20,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3
        )[0]['summary_text']
        chunk_summaries.append(summary)
    
    if not chunk_summaries:
        print("警告：沒有生成任何摘要")
        return None
        
    # 合併所有塊的摘要
    final_summary = ' '.join(chunk_summaries)
    
    # 正規化最終摘要和參考摘要
    normalized_summary = normalizer(final_summary)
    normalized_highlights = normalizer(batch[summary_column])
    
    # 計算評估指標
    rouge_score = rouge.score(normalized_summary, normalized_highlights)
    P, R, F1 = score(
        [normalized_summary],
        [normalized_highlights],
        lang='en',
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return {
        'summary': final_summary,
        'rouge_scores': {
            'rouge1': rouge_score['rouge1'].fmeasure,
            'rouge2': rouge_score['rouge2'].fmeasure,
            'rougeL': rouge_score['rougeL'].fmeasure
        },
        'bert_scores': {
            'precision': P.item(),
            'recall': R.item(),
            'f1': F1.item()
        }
    }


def calculate_metrics(results):
    """計算整體評估指標"""
    return {
        'rouge1': np.mean([r['rouge_scores']['rouge1'] for r in results]),
        'rouge2': np.mean([r['rouge_scores']['rouge2'] for r in results]),
        'rougeL': np.mean([r['rouge_scores']['rougeL'] for r in results]),
        'bert_score_f1': np.mean([r['bert_scores']['f1'] for r in results])
    }

def save_results(results, metrics, model_name: str):
    """保存結果到 HuggingFace Hub"""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'metrics': metrics,
        'sample_results': results
    }
    
    json_str = json.dumps(summary, ensure_ascii=False, indent=2)
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj=json_str.encode(),
        path_in_repo=f"results/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        repo_id="TakalaWang/AMI_RESULTS",
        repo_type="dataset"
    )

def main():
    # 載入數據集
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    dataset = load_dataset("TakalaWang/AMI_ABSTRACTIVE", split="test")
    transcript_column = "transcript"
    summary_column = "summary"
    model_name = "TakalaWang/ami-bart-large-cnn-finetune"
    
    # 初始化模型和工具
    summarizer, rouge, normalizer = init_models(model_name)
    
    # 處理數據
    results = []  # 修復：添加 results 列表初始化
    for item in tqdm(dataset):
        result = process_batch(item, summarizer, rouge, normalizer, transcript_column, summary_column)
        if result is not None:
            results.append(result)
  

    # 確保有結果才進行計算
    if not results:
        print("錯誤：沒有成功處理的數據")
        return None, None
        
    # 計算整體指標
    metrics = calculate_metrics(results)
    
    # 輸出結果
    print("\n=== 評估分數 ===")
    print(f"成功處理的樣本數: {len(results)}")
    print(f"平均 ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"平均 ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"平均 ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"平均 BERTScore F1: {metrics['bert_score_f1']:.4f}")
    
    # 保存結果
    save_results(results, metrics, model_name)
    
    return results, metrics

if __name__ == "__main__":
    main()
