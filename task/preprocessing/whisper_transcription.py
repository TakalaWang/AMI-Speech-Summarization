import os, json

from tqdm import tqdm
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

def init_model(asr_model_name: str):
    """初始化模型和工具"""
    processor = AutoProcessor.from_pretrained(asr_model_name)
    
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_model_name, low_cpu_mem_usage=True, use_safetensors=True
    )
    
    asr_model = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        generate_kwargs={
            "language": "english",
            "task": "transcribe"
        }
    )
    print("ASR model loaded")
    vad_model = load_silero_vad()
    print("VAD model loaded")
    
    return asr_model, vad_model
    
def get_transcription(meeting_name: str, asr_model, vad_model):
    stop_mark = [".", "?", "!"]
    transcription_text = []
    for speaker in range(4):
        name = chr(ord('A') + speaker)
        audio_path = f"/share/corpus/amicorpus/{meeting_name}/audio/{meeting_name}.Headset-{speaker}.wav"
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} not found")
            continue
        
        print(f"Processing {meeting_name} {name} speaker")
        audio = read_audio(audio_path)
        speech_timestamps = get_speech_timestamps(
            audio,
            vad_model,
            threshold=0.4,
            max_speech_duration_s=30,
            min_silence_duration_ms=200, 
            visualize_probs=True
        )
         
        def continue_text(text: str):
            if text.endswith("..."):
                return True 
            if any(text.endswith(mark) for mark in stop_mark):
                return False
            return True
        
        for timestamp in speech_timestamps:
            start, end = timestamp["start"], timestamp["end"]
            chunk = audio[start:end].numpy()
            transcription = asr_model(chunk)['text'].strip()
            if transcription_text and continue_text(transcription_text[-1]['text']):
                transcription_text[-1]['text'] += f" {transcription}"
                transcription_text[-1]['end'] = end
            else:
                transcription_text.append({
                    'speaker': f"Speaker {name}",
                    'start': start,
                    'end': end,
                    'text': transcription
                })
                
    transcription_text.sort(key=lambda x: x['start'])
    return transcription_text
    

def get_split():
    split = {}
    for split_name in ['train', 'valid', 'test']:
        with open(f'split/{split_name}.txt') as f:
            split[split_name] = f.read().splitlines()
    return split

def main():
    transcript_root_path = "../transcription_data"
    
    # 載入數據集
    splits = get_split()
    
    # 初始化模型和工具
    asr_model, vad_model = init_model("openai/whisper-large-v3")
    
    os.makedirs(transcript_root_path, exist_ok=True)
    for split_name, meetings in splits.items():
        os.makedirs(os.path.join(transcript_root_path, split_name), exist_ok=True)
        print(f"Processing {split_name} split")
        for meeting in tqdm(meetings):
            print(f"Processing {meeting} meeting")
            if os.path.exists(os.path.join(transcript_root_path, split_name, f"{meeting}.json")):
                continue
            transcription_text_json = get_transcription(meeting, asr_model, vad_model)
            with open(os.path.join(transcript_root_path, split_name, f"{meeting}.json"), "w") as f:
                json.dump(transcription_text_json, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
