print("CP1: starting script", flush=True)

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

import torch
print("CP2: torch imported", flush=True)

from datasets import load_dataset
print("CP3: datasets imported", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
print("CP4: transformers imported", flush=True)

from peft import LoraConfig, get_peft_model
print("CP5: peft imported", flush=True)

from trl import SFTTrainer, SFTConfig
print("CP6: trl imported", flush=True)

print("ALL IMPORTS DONE - SCRIPT END", flush=True)