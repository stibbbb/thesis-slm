import sys
print("BEFORE IMPORTS", flush=True)
sys.stderr.write("BEFORE IMPORTS (stderr)\n")
sys.stderr.flush()

import torch
import datasets
import transformers
import peft
import trl

print("AFTER IMPORTS", flush=True)