import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

print("CP1: starting script", flush=True)

# 1. CONFIG
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./trained_qwen0.5b_dummy"

# 2. LOAD TOKENIZER AND BASE MODEL
print(f"CP2: loading base model: {MODEL_NAME}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda"
)
print("CP3: model loaded onto GPU", flush=True)

# 3. FORMAT AND TOKENIZE THE DATASET
def format_example(example):
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n{example['input']}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

def tokenize_example(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("CP4: loading and formatting dataset", flush=True)
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
dataset = dataset.map(format_example)
dataset = dataset.map(tokenize_example)
print("CP5: dataset ready", flush=True)

# 4. LoRA CONFIG
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("CP6: model wrapped with LoRA", flush=True)

# 5. TRAINING CONFIG
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    optim="adamw_torch",
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
print("CP7: trainer built, starting training", flush=True)

# 6. TRAIN
trainer.train()
print("CP8: training finished", flush=True)

trainer.save_model(OUTPUT_DIR)
print(f"CP9: model saved to {OUTPUT_DIR}, done", flush=True)