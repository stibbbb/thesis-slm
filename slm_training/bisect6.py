print("BEFORE IMPORTS", flush=True)

import torch
print("torch OK", flush=True)

from datasets import load_dataset
print("datasets OK", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
print("transformers OK", flush=True)

from peft import LoraConfig, get_peft_model
print("peft OK", flush=True)

from trl import SFTTrainer, SFTConfig
print("trl OK", flush=True)

print("DONE - no crash so far", flush=True)