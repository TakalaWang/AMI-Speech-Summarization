import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import os
from evaluate import load
from transformers import GenerationConfig
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

base_model_id="meta-llama/Meta-Llama-3.1-8B"
model_id="TakalaWang/unsloth-llama-3.1-8B-finetune"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype="float16", 
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)
model.config.use_cache=False
model.config.pretraining_tp=1

model.load_adapter(model_id)


def generate_response(user_input):
    generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
        top_k=5,temperature=0.5,repetition_penalty=1.2,
        max_new_tokens=60,pad_token_id=tokenizer.eos_token_id
    )
    inputs = tokenizer(user_input, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, generation_config=generation_config)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

dataset = load_dataset("TakalaWang/AMI_WHISPER_ASR", split="test")


def formatted_train(question)->str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

dataset = dataset.map(lambda x: {"text": formatted_train(x["transcript_with_speaker"])})

# use rouge to evaluate the response of all test data average score
rouge = load("rouge")

rouge_score = 0

for data in dataset:
    response = generate_response(data["text"])
    reference = data["summary"]
    rouge_score += rouge.compute(predictions=[response], references=[reference])

print(rouge_score / len(dataset))
