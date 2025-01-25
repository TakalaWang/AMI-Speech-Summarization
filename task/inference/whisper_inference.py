import os
import json
from datetime import datetime
from typing import List, Tuple

import torch
import evaluate
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi, login
from transformers import pipeline, AutoProcessor, AutoTokenizer, AutoFeatureExtractor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

def init_model(model_name: str):
    """初始化模型和工具"""
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            processor=processor,
            chunk_length_s=30,
            generate_kwargs={
                "language": "en",
                "task": "transcribe"
            }
        )
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            generate_kwargs={
                "language": "en",
                "task": "transcribe"
            }
        )
    
    # 載入 VAD 模型
    torch.set_grad_enabled(False)  # 禁用梯度計算
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 載入 VAD 模型，強制重新下載
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        onnx=False,
        trust_repo=True
    )
    
    # 配置 VAD 模型
    vad_model.eval()  # 設置為評估模式
    vad_model.to(device)  # 移到正確的設備
    vad_model.float()  # 確保使用 float32
    
    # 初始化評估工具
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    normalizer = BasicTextNormalizer()
    
    return asr, vad_model, vad_utils, wer_metric, cer_metric, normalizer

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
    print(speech_timestamps)
    
    # 如果沒有檢測到語音段落，返回整個音頻
    if not speech_timestamps:
        return [(audio.cpu(), 0, audio.shape[-1]/sample_rate)]
    
    # 收集語音片段
    chunks_with_time = []
    for ts in speech_timestamps:
        # 直接從原始音訊中切割出語音片段
        chunk = audio[ts['start']:ts['end']]
        if isinstance(chunk, torch.Tensor) and chunk.numel() > 0:
            start_time = ts['start'] / sample_rate
            end_time = ts['end'] / sample_rate
            duration = end_time - start_time
            chunks_with_time.append((chunk.cpu(), start_time, duration))
    
    return chunks_with_time if chunks_with_time else [(audio.cpu(), 0, audio.shape[-1]/sample_rate)]

def process_batch(batch, asr, vad_model, vad_utils, normalizer, audio_column: str, transcript_column: str):
    """處理單個批次的數據"""
    # 處理音頻輸入
    audio_data = batch[audio_column]
    waveform = torch.tensor(audio_data['array'])
    sample_rate = audio_data['sampling_rate']
    
    chunks = process_audio_vad(waveform, sample_rate, vad_model, vad_utils)
    
    # ASR 處理每個語音片段
    transcriptions = []
    for chunk, start, end in chunks:
        try:
            # 創建輸入資料，包含 attention_mask
            input_features = {
                "array": chunk.numpy(), 
                "sampling_rate": sample_rate,
                "attention_mask": torch.ones_like(chunk, dtype=torch.long)  # 添加 attention_mask
            }
            
            result = asr(
                input_features,
                return_timestamps=True
            )
            text = result['text']
            
            if text.strip():
                transcriptions.append(text)
            
        except Exception as e:
            print(f"處理音頻片段時出錯: {str(e)}")
            print(f"音頻片段形狀: {chunk.shape}")
            continue
    
    # 合併轉錄結果
    final_transcription = ' '.join(transcriptions)
    
    # 如果沒有成功的轉錄，返回空結果
    if not final_transcription.strip():
        print("警告：未能生成任何轉錄文本")
        return None
    
    # 正規化文本
    normalized_prediction = normalizer(final_transcription)
    normalized_reference = normalizer(batch[transcript_column])
    
    return {
        'reference': normalized_reference,
        'prediction': normalized_prediction
    }

def calculate_metrics(results, wer_metric, cer_metric):
    """計算整體評估指標"""
    references = []
    predictions = []
    for result in results:
        if result['reference'].strip() and result['prediction'].strip():
            references.append(result['reference'])
            predictions.append(result['prediction'])
        

    return {
        'wer': wer_metric.compute(references=references, predictions=predictions),
        'cer': cer_metric.compute(references=references, predictions=predictions)
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
        path_in_repo=f"results/asr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        repo_id="TakalaWang/AMI_RESULTS",
        repo_type="dataset"
    )

def main():
    # 載入數據集
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    dataset = load_dataset("TakalaWang/AMI_SPEECH_SUMMARY", split="train", num_proc=25)
    audio_column = "audio"
    transcript_column = "text"
    model_name = "openai/whisper-large-v3"
    
    # # 初始化模型和工具
    asr, vad_model, vad_utils, wer_metric, cer_metric, normalizer = init_model(model_name)
    
    # 處理數據
    results = []
    print("dataset size", len(dataset))
    for item in tqdm(dataset):
        result = process_batch(item, asr, vad_model, vad_utils, normalizer, audio_column, transcript_column)
        if result["reference"] and result["prediction"]:
            results.append(result)
        else:
            print("ERROR: ", result)
            
            
    with open("results/whisper_medium_results.json", "r") as f:
        # json.dump(results, f, ensure_ascii=False, indent=2)
        results = json.load(f)
    
    # 計算整體指標
    metrics = calculate_metrics(results, wer_metric, cer_metric)
    
    # 輸出結果
    print("\n=== 評估分數 ===")
    print(f"成功處理的樣本數: {len(results)}")
    print(f"WER: {metrics['wer']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")
    
    # 保存結果
    save_results(results, metrics, model_name)
    
    return results, metrics

if __name__ == "__main__":
    main()
