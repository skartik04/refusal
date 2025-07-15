import torch
import einops
from transformer_lens import HookedTransformer
from jaxtyping import Int, Float
from typing import List, Callable
import random
import numpy as np

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for deterministic algorithms
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)

# Set deterministic behavior
set_seed(42)

# Use GPU 2
device = torch.device("cuda:0")

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# Load model and avg_direction to the correct GPU
model = HookedTransformer.from_pretrained("Qwen/Qwen-1_8B-Chat")
model = model.to(device)  # Explicitly move model to device
avg_direction = torch.load("avg_direction.pt", map_location=device)

def get_prompt(instruction: str) -> str:
    return QWEN_CHAT_TEMPLATE.format(instruction=instruction)

def tokenize_chat(prompts: List[str]) -> Int[torch.Tensor, "batch seq_len"]:
    formatted = [get_prompt(p) for p in prompts]
    # Don't specify device here, let model.to_tokens handle it naturally
    tokens = model.to_tokens(formatted, padding_side="left")
    return tokens

def make_actadd_hook(direction: torch.Tensor, scale: float = 1.0):
    # Pre-move direction to device and normalize
    direction = direction.to(device)
    direction = direction / direction.norm()
    
    def hook(resid_pre, hook):
        return resid_pre + scale * direction.view(1, 1, -1)
    return hook

def make_ablation_hook(direction: torch.Tensor):
    # Pre-move direction to device and normalize
    direction = direction.to(device)
    direction = direction / direction.norm()
    
    def hook(resid_pre, hook):
        proj_coeff = einops.einsum(resid_pre, direction, "... d_model, d_model -> ...")
        proj = einops.einsum(proj_coeff, direction, "..., d_model -> ... d_model")
        return resid_pre - proj
    return hook

def _generate_with_hooks(
    toks: Int[torch.Tensor, "batch seq_len"],
    max_tokens_generated: int,
    fwd_hooks = [],
) -> List[str]:
    # Ensure input tokens are on same device as model
    toks = toks.to(device)
    all_toks = toks.clone()
    
    for _ in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks)
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        all_toks = torch.cat([all_toks, next_token], dim=1)
        
    full_texts = model.to_string(all_toks)
    prompt_texts = model.to_string(toks)
    return [full[len(prompt):].strip() for full, prompt in zip(full_texts, prompt_texts)]

def run_with_mode(prompt: str, mode: str) -> tuple[str, str]:
    # Reset seed for reproducible generation
    set_seed(42)
    
    model.reset_hooks()
    toks = tokenize_chat([prompt])
    baseline = _generate_with_hooks(toks, max_tokens_generated=64, fwd_hooks=[])[0]

    # Reset seed again for the modified generation
    set_seed(42)
    
    if mode == "refuse":
        layer, pos = 11, 4
        direction = avg_direction[layer, pos]
        hook_fn = make_actadd_hook(direction, scale=10.0)
        hooks = [(f"blocks.{layer}.hook_resid_pre", hook_fn)]
    elif mode == "bypass":
        layer, pos = 15, 4
        direction = avg_direction[layer, pos]
        hook_fn = make_ablation_hook(direction)
        hooks = [
            (f"blocks.{layer}.hook_resid_pre", hook_fn),
            (f"blocks.{layer}.hook_resid_mid", hook_fn),
            (f"blocks.{layer}.hook_resid_post", hook_fn),
        ]
    else:
        raise ValueError("Invalid mode")

    modded = _generate_with_hooks(toks, max_tokens_generated=64, fwd_hooks=hooks)[0]
    return baseline, modded