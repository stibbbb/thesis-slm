print("CP1: starting", flush=True)

import torch
print("CP2: torch imported", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
print("CP3: transformers imported", flush=True)

from peft import LoraConfig, get_peft_model
print("CP3b: peft imported", flush=True)

from trl import SFTTrainer, SFTConfig
print("CP3c: trl imported", flush=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("CP4: loading tokenizer", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("CP5: tokenizer loaded", flush=True)

print("CP6: loading model onto GPU now", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda"
)
print("CP7: model loaded successfully", flush=True)