from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# ===== 1. Load Fine-tuned Model from Hugging Face Hub =====
model_name = "TakalaWang/llama3-lora-finetuned-ami-summary"  # Replace with your HF Hub repo

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, model_name)

# ===== 2. Generate Text =====
model.eval()
test_data = load_dataset("json", data_files="../transcription_data/test/*.json")

for data in test_data:
    input_text = data["transcript"]
    summary = data["summary"]

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Adjust device as needed
    outputs = model.generate(
        **inputs
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
