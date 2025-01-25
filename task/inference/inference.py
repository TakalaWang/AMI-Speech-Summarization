import os
import json
from datetime import datetime
from typing import List, Tuple

import evaluate
import torch
import numpy as np
from tqdm import tqdm
from bert_score import score
from datasets import load_dataset
from rouge_score import rouge_scorer
from huggingface_hub import HfApi
from transformers import (
    pipeline,
    AutoProcessor,
    AutoTokenizer,
    AutoFeatureExtractor
)
from dotenv import load_dotenv
from huggingface_hub import login
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

def init_models(
    asr_model_name: str = "openai/whisper-large",
    sum_model_name: str = "facebook/bart-large"
):
    """初始化模型和工具"""
    # ASR 模型初始化
    try:
        processor = AutoProcessor.from_pretrained(asr_model_name)
        asr = pipeline(
            "automatic-speech-recognition",
            model=asr_model_name,
            processor=processor,
            chunk_length_s=30,
            generate_kwargs={
                "language": "en",
                "task": "transcribe"
            }
        )
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(asr_model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(asr_model_name)
        asr = pipeline(
            "automatic-speech-recognition",
            model=asr_model_name,
            chunk_length_s=30,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            generate_kwargs={
                "language": "en",
                "task": "transcribe"
            }
        )

    # 入 VAD 模型
    torch.set_grad_enabled(False)  # 禁用梯度計算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 載入 VAD 模型，強制重新下載
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        onnx=False,
        trust_repo=True
    )
    
    # 配置 VAD 模型
    vad_model.eval()
    vad_model.to(device)
    vad_model.float()

    # 摘要模型初始化
    try:
        processor = AutoProcessor.from_pretrained(sum_model_name)
        summarizer = pipeline(
            "summarization",
            model=sum_model_name,
            processor=processor,
            model_kwargs={"device_map": "auto"}
        )
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
        summarizer = pipeline(
            "summarization",
            model=sum_model_name,
            tokenizer=tokenizer,
            model_kwargs={"device_map": "auto"}
        )

    # 評估工具初始化
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    normalizer = BasicTextNormalizer()
    
    return asr, summarizer, vad_model, vad_utils, wer_metric, cer_metric, rouge, normalizer

def process_audio_vad(audio: torch.Tensor, sample_rate: int, vad_model, utils) -> List[Tuple[torch.Tensor, float, float]]:
    """處理音頻 VAD"""
    get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils
    
    # 確保音頻是正確的格式
    if audio.dim() == 0:
        audio = audio.unsqueeze(0)
    elif audio.dim() != 1:
        audio = audio.squeeze()
    
    # 將音頻轉換為 float32 並移到正確的設備上
    device = next(vad_model.parameters()).device
    audio = audio.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():  # 禁用梯度計算            
        # 獲取語音時間戳
        speech_timestamps = get_speech_timestamps(
            audio, 
            vad_model, 
            sampling_rate=sample_rate,
            min_speech_duration_ms=100,
            max_speech_duration_s=30,
            return_seconds=False
        )
    
    # 如果沒有檢測到語音段落，返回整個音頻
    if not speech_timestamps:
        return [(audio.cpu(), 0, audio.shape[-1]/sample_rate)]
    
    # 收集語音片段
    chunks_with_time = []
    for ts in speech_timestamps:
        chunk = audio[ts['start']:ts['end']]
        if isinstance(chunk, torch.Tensor) and chunk.numel() > 0:
            start_time = ts['start'] / sample_rate
            end_time = ts['end'] / sample_rate
            duration = end_time - start_time
            chunks_with_time.append((chunk.cpu(), start_time, duration))
    
    return chunks_with_time if chunks_with_time else [(audio.cpu(), 0, audio.shape[-1]/sample_rate)]

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

def process_audio(audio_data, asr, vad_model, vad_utils):
    waveform = torch.tensor(audio_data['array'])
    sample_rate = audio_data['sampling_rate']
    chunks = process_audio_vad(waveform, sample_rate, vad_model, vad_utils)
    
    transcriptions = []
    for chunk, start, end in chunks:

        input_features = {
            "array": chunk.numpy(), 
            "sampling_rate": sample_rate,
            "attention_mask": torch.ones_like(chunk, dtype=torch.long)
        }
        result = asr(
            input_features,
            return_timestamps=True
        )
        text = result['text']
        
        if text.strip():
            transcriptions.append({
                "start": start,
                "end": end,
                "text": text
            })
            
    return transcriptions
    

def process_batch(
    batch,
    asr,
    summarizer,
    vad_model,
    vad_utils,
    normalizer,
):
    """處理單個批次的數據"""
    # 分別處理每個參與者的音頻
    individual_results = {}
    all_transcriptions = []
    
    for participant in ["A", "B", "C", "D"]:
        # 處理單個參與者的音頻
        print(batch)
        participant_transcriptions = process_audio(
            batch[f"audio_{participant}"], 
            asr, 
            vad_model, 
            vad_utils
        )
        print(participant_transcriptions)
        
        if participant_transcriptions:
            # 保存該參與者的轉錄結果
            participant_transcript = ' '.join([t['text'] for t in participant_transcriptions])
            normalized_participant_transcript = normalizer(participant_transcript)
            normalized_ref_transcript = normalizer(batch[f"transcript_{participant}"])
            
            individual_results[participant] = {
                'reference': normalized_ref_transcript,
                'prediction': normalized_participant_transcript,
                'timestamps': participant_transcriptions
            }
            
            # 添加到整體轉錄列表
            all_transcriptions.extend(participant_transcriptions)
    
    # 按時間戳排序所有轉錄
    all_transcriptions.sort(key=lambda x: x['start'])
    
    # 合併所有轉錄
    combined_transcript = ' '.join([t['text'] for t in all_transcriptions])
    if not combined_transcript.strip():
        print("警告：未能生成任何轉錄文本")
        return None
    
    
    normalized_combined_transcript = normalizer(combined_transcript)
    normalized_combined_reference = normalizer(batch["transcription"])
    
    # 生成摘要
    text_chunks = chunk_text(normalized_combined_transcript)
    chunk_summaries = []
    
    for chunk in text_chunks:
        summary = summarizer(
            chunk,
            max_length=130 // len(text_chunks),
            min_length=20,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3
        )[0]['summary_text']
        chunk_summaries.append(summary)
    
    final_summary = ' '.join(chunk_summaries)
    normalized_ref_summary = normalizer(batch["summary"])
    
    return {
        'individual_transcripts': {
            participant: {
                'reference': result['reference'],
                'prediction': result['prediction']
            }
            for participant, result in individual_results.items()
        },
        'combined_transcript': {
            'reference': normalized_combined_reference,
            'prediction': normalized_combined_transcript
        },
        'summary': {
            'reference': normalized_ref_summary,
            'prediction': final_summary
        }
    }
     

def calculate_metrics(results, wer_metric, cer_metric, rouge):
    """計算整體評估指標"""
    # 收集所有個別轉錄結果
    all_individual_refs = []
    all_individual_preds = []
    
    # 收集合併轉錄和摘要結果
    combined_refs = []
    combined_preds = []
    summary_refs = []
    summary_preds = []
    
    for result in results:
        if result is None:
            continue
            
        # 收集所有個別轉錄
        for transcript in result['individual_transcripts'].values():
            all_individual_refs.append(transcript['reference'])
            all_individual_preds.append(transcript['prediction'])
        
        # 收集合併轉錄
        combined = result['combined_transcript']
        combined_refs.append(combined['reference'])
        combined_preds.append(combined['prediction'])
        
        # 收集摘要
        summary = result['summary']
        summary_refs.append(summary['reference'])
        summary_preds.append(summary['prediction'])
    
    # 計算個別轉錄的整體指標
    individual_metrics = {
        'wer': wer_metric.compute(
            references=all_individual_refs,
            predictions=all_individual_preds
        ),
        'cer': cer_metric.compute(
            references=all_individual_refs,
            predictions=all_individual_preds
        )
    }
    
    # 計算合併轉錄的指標
    combined_metrics = {
        'wer': wer_metric.compute(
            references=combined_refs,
            predictions=combined_preds
        ),
        'cer': cer_metric.compute(
            references=combined_refs,
            predictions=combined_preds
        )
    }
    
    # 計算摘要指標
    rouge_scores = []
    bert_scores = []
    for ref, pred in zip(summary_refs, summary_preds):
        rouge_score = rouge.score(pred, ref)
        P, R, F1 = score([pred], [ref], lang='en', verbose=False)
        
        rouge_scores.append({
            'rouge1': rouge_score['rouge1'].fmeasure,
            'rouge2': rouge_score['rouge2'].fmeasure,
            'rougeL': rouge_score['rougeL'].fmeasure
        })
        bert_scores.append(F1.item())
    
    summary_metrics = {
        'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
        'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
        'rougeL': np.mean([s['rougeL'] for s in rouge_scores]),
        'bert_score_f1': np.mean(bert_scores)
    }
    
    return {
        'individual': individual_metrics,
        'combined': combined_metrics,
        'summary': summary_metrics
    }

def save_results(results, metrics, model_name: str):
    """保存結果到 HuggingFace Hub"""
    # 備詳細的結果
    detailed_results = []
    for result in results:
        if result is None:
            continue
            
        # 收集每個樣本的詳細資訊
        sample_detail = {
            # 個別轉錄結果
            'individual_transcripts': {
                participant: {
                    'reference': transcript['reference'],
                    'prediction': transcript['prediction']
                }
                for participant, transcript in result['individual_transcripts'].items()
            },
            # 合併後的轉錄
            'combined_transcript': {
                'reference': result['combined_transcript']['reference'],
                'prediction': result['combined_transcript']['prediction']
            },
            # 摘要結果
            'summary': {
                'reference': result['summary']['reference'],
                'prediction': result['summary']['prediction']
            }
        }
        detailed_results.append(sample_detail)
    
    # 創建結果摘要
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'metrics': {
            'individual_transcription': metrics['individual'],
            'combined_transcription': metrics['combined'],
            'summary_generation': metrics['summary']
        },
        'predictions': detailed_results  # 添加詳細的預測結果
    }
    
    # 轉換為 JSON 字符串
    json_str = json.dumps(summary, ensure_ascii=False, indent=2)
    
    # 上傳到 HuggingFace Hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj=json_str.encode(),
        path_in_repo=f"results/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        repo_id="TakalaWang/AMI_RESULTS",
        repo_type="dataset"
    )

def main():
    # 載入數據集
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    dataset = load_dataset("TakalaWang/AMI_SPEECH_SUMMARY", split="test")
    
    # 設定參數
    asr_model_name = "openai/whisper-large"
    sum_model_name = "facebook/bart-large-cnn"
    
    # 初始化模型和工具
    asr, summarizer, vad_model, vad_utils, wer_metric, cer_metric, rouge, normalizer = init_models(
        asr_model_name,
        sum_model_name
    )
    
    # 處理數據
    results = []
    for item in tqdm(dataset):
        result = process_batch(
            item, 
            asr, 
            summarizer, 
            vad_model, 
            vad_utils, 
            normalizer
        )
        if result["reference"] and result["prediction"]:
            results.append(result)
        else:
            print("ERROR: ", result)
            
    with open("results/inference_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 計算整體指標
    metrics = calculate_metrics(results, wer_metric, cer_metric, rouge)
    
    # 輸出結果
    print("\n=== ASR 評估分數 ===")
    print(f"WER: {metrics['asr']['wer']:.4f}")
    print(f"CER: {metrics['asr']['cer']:.4f}")
    print("\n=== 摘要評估分數 ===")
    print(f"ROUGE-1: {metrics['summary']['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['summary']['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['summary']['rougeL']:.4f}")
    print(f"BERTScore F1: {metrics['summary']['bert_score_f1']:.4f}")
    
    # 保存結果
    model_name = f"{asr_model_name}+{sum_model_name}"
    save_results(results, metrics, model_name)
    
    return {
        'individual_results': results,
        'metrics': metrics
    }

if __name__ == "__main__":
    main() 