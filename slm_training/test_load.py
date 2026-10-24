import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model (this downloads ~1GB the first time)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

print("Model loaded. Running a test generation...")

messages = [
    {"role": "user", "content": "Convert these recognized sign tokens into a natural sentence. Tokens: apple, water"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True
).to("cuda")

output = model.generate(**inputs, max_new_tokens=50)
print("\n--- MODEL OUTPUT ---")
print(tokenizer.decode(output[0], skip_special_tokens=True))