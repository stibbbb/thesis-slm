import sys
print("BEFORE IMPORTS", flush=True)
sys.stderr.write("BEFORE IMPORTS (stderr)\n")
sys.stderr.flush()

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("AFTER IMPORTS", flush=True)
sys.stderr.write("AFTER IMPORTS (stderr)\n")
sys.stderr.flush()