from model import GPT
import os
import torch
from dataclasses import dataclass

models_dir = os.path.join(os.path.dirname(__file__), "Checkpoints")

ckpt_path = os.path.join(models_dir, 'checkpoint - second train dataset - finetune.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config
@dataclass
class GPTConfig:
    def __init__(self):
        self.gradient_accumulation_steps = 5 # used to simulate larger batch sizes
        self.batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
        self.block_size = 512
        self.vocab_size = 50304
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
        self.bias = False

        # Adamw optimizer
        self.learning_rate = 6e-4 # max learning rate
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

        # Learning rate decay settings
        self.decay_lr = True # whether to decay the learning rate
        self.warmup_iters = 2000 # how many steps to warm up for
        self.lr_decay_iters = 188800 # should be ~= max_iters per Chinchilla
        self.min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

checkpoint = torch.load(ckpt_path, map_location = device)

# Create the model
model = GPT(checkpoint['config'])
state_dict = checkpoint['model']
checkpoint = None # Free up memory

# Fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'

for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)

model.to(device)

userText = input("User:\n")

print(model.generate_text("<|user|>:\n" + userText + "\n<|assistant|>:\n", 512))