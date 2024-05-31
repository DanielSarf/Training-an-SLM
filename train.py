"""
I adapted this wonderful GPT-2 training script (by Andrej Karpathy) for my own use-case
"""

import os
import time
from datetime import datetime
import math
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
import torch

from model import GPT

# -----------------------------------------------------------------------------

out_dir = os.path.join(os.path.dirname(__file__), "Checkpoints")
max_iters = 136000 # Total number of training iterations (just a number I think will go nicely)
eval_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
device = 'cuda' #'cpu' or 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

# Wandb Logging
wandb_log = True
wandb_project = 'SLMAcademicProject'
wandb_run_name = 'Run ' + datetime.now().strftime('%d/%m/%Y - %H:%M:%S')

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

config = GPTConfig()

# -----------------------------------------------------------------------------

tokens_per_iter = config.gradient_accumulation_steps * config.batch_size * config.block_size

print(f"Tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device_type = 'cuda' if device == 'cuda' else 'cpu' # for later use in torch.autocast

# Note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype = ptdtype)

# -----------------------------------------------------------------------------

# Data loader
data_dir = os.path.join(os.path.dirname(__file__), 'Data', 'Training and Validation Data', "First Training Data")

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'Train.bin'), dtype = np.uint16, mode = 'r')
    else:
        data = np.memmap(os.path.join(data_dir, 'Val.bin'), dtype = np.uint16, mode = 'r')

    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i + config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + config.block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # This allows us to x and y to the GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking = True), y.pin_memory().to(device, non_blocking = True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

# -----------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    # Init a new model from scratch
    print("Initializing a new model from scratch")

    model = GPT(config)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    
    # Resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')

    checkpoint = torch.load(ckpt_path, map_location = device)

    # Create the model
    model = GPT(checkpoint['config'])
    state_dict = checkpoint['model']
    
    # Fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    
    iter_num = checkpoint['iter_num']
    
    best_val_loss = checkpoint['best_val_loss']
    
# crop down the model block size if desired, using model surgery
if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    config.block_size = config.block_size # So that the checkpoint will have the right value

model.to(device)

# Initialize a GradScaler. If enabled = False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled = (dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None # Free up memory

# Compile the model
if compile:
    print("Compiling the model... (takes a ~minute)")

    unoptimized_model = model
    
    model = torch.compile(model) # requires PyTorch 2.0

# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    
    return out

# Learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    
    assert 0 <= decay_ratio <= 1
    
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project = wandb_project, name = wandb_run_name, config = config.__dict__)

# Training loop
X, Y = get_batch('train') # Fetch the very first batch (assumes there is enough tokens to train with)

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0

while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()

        print(f"Step {iter_num}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100, # convert to percentage
                "current_output_looks_like": model.generate_text("\n", config.block_size),
            })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }

                print(f"Saving checkpoint to {out_dir}")
                
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        
        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        
        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    
    # Clip the gradient
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    
    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none = True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
 
    if iter_num % log_interval == 0:
        # Get loss as float. Note: this is a CPU-GPU sync point
        # Scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * config.gradient_accumulation_steps
        
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
 
        print(f"Iter {iter_num}: Loss {lossf:.4f}, Time {dt*1000:.2f}ms, MFU {running_mfu * 100:.2f}%")
 
    iter_num += 1
    local_iter_num += 1

    # Termination condition
    if iter_num > max_iters:
        break