import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "google/gemma-2-2b-it"
ADAPTER_PATH = "./trained_gemma2_2b_dummy"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="cuda"
)

print("Loading fine-tuned LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("Ready. Running test prompts...\n", flush=True)


def generate(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Only show the assistant's reply, not the whole chat template
    reply = full_text.split("assistant")[-1].strip()
    return reply


test_prompts = [
    "Convert these recognized sign tokens into a natural sentence. Tokens: apple, water",
    "Convert these recognized sign tokens into a natural sentence. Tokens: run, happy",
    "Convert these recognized sign tokens into a natural sentence. Tokens: blue",
    "Reformat this transcribed speech for clear display.\nim going for a run later",
]

for prompt in test_prompts:
    print(f"PROMPT: {prompt}", flush=True)
    result = generate(prompt)
    print(f"OUTPUT: {result}\n", flush=True)

print("Done.", flush=True)