"""
Autoresearch pretraining script using x-transformers.
Single-GPU, single-file, time-budgeted training on enwik8.

Hardware: Any NVIDIA GPU with >= 8 GB VRAM (tested on RTX 4090).
          FP8 supported on Ada Lovelace / Hopper / Blackwell (SM89+).
Precision: BF16 (default), FP8 via torchao (optional), FP16 (optional)
Optimizer: MuonAdamAtan2 (Muon for matrix params, AdamAtan2 for rest)

Usage:
    python train.py                      # BF16 (default, always works)
    USE_FP8=1 python train.py            # FP8 via torchao (SM89+)
    USE_FP16=1 python train.py           # FP16 instead of BF16

References:
    - x-transformers: https://github.com/lucidrains/x-transformers
    - See docs/adjustable_params.md for full parameter reference
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
import math
import time
import gzip
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# x-transformers (from local ./x-transformers submodule)
# ---------------------------------------------------------------------------

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "x-transformers")
)

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# ---------------------------------------------------------------------------
# Precision selection: FP8 (torchao), FP16, or BF16 (default)
# ---------------------------------------------------------------------------

USE_FP8 = os.environ.get("USE_FP8", "0") == "1"
USE_FP16 = os.environ.get("USE_FP16", "0") == "1"
fp8_available = False

if USE_FP8:
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        fp8_available = True
        print("FP8: torchao float8 training loaded successfully")
    except ImportError:
        print("FP8: torchao not installed, falling back to BF16")
        print("  Install with: pip install torchao")
        USE_FP8 = False

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamAtan2 or fallback to AdamW)
# ---------------------------------------------------------------------------

try:
    from adam_atan2_pytorch import MuonAdamAtan2

    HAS_MUON = True
except ImportError:
    print("Warning: adam-atan2-pytorch not installed, falling back to AdamW")
    print("  Install with: pip install adam-atan2-pytorch")
    HAS_MUON = False

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 4096  # context length for enwik8 training
TIME_BUDGET = 300  # training time budget in seconds (5 minutes)
EVAL_TOKENS = 5_000_000  # validation set size (5M bytes)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture (x-transformers Decoder)
MODEL_DIM = 448  # hidden dimension
MODEL_DEPTH = 6  # number of transformer layers
MODEL_HEADS = 7  # number of attention heads
VOCAB_SIZE = 256  # byte-level (char-level), one token per byte value

# Optimization
LEARNING_RATE = 1.1e-2  # slightly above 1e-2 with clip=0.8
BATCH_SIZE = 24  # per-device micro-batch size
GRADIENT_ACCUMULATE_EVERY = 1  # gradient accumulation steps
WEIGHT_DECAY = 0.1  # AdamW weight decay
GRAD_CLIP = 0.8  # gradient norm clipping (between 0.5 and 1.0)

# LR Schedule (time-based)
WARMUP_RATIO = 0.05  # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.47  # fine-tuning between 0.45 and 0.5
FINAL_LR_FRAC = 0.02  # final LR as fraction of initial

# Logging
VALIDATE_EVERY = 100  # validation frequency (in steps)
GENERATE_EVERY = 500  # text generation frequency (in steps)
GENERATE_LENGTH = 512  # tokens to generate for qualitative eval

# GPU performance reference for MFU estimation
# RTX 4090 BF16 peak: ~330 TFLOPS (adjust for your GPU)
GPU_BF16_PEAK_FLOPS = 330e12

# ---------------------------------------------------------------------------
# Data: enwik8 (character-level)
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "x-transformers", "data", "enwik8.gz"
)


def load_enwik8(data_path=DATA_PATH):
    """Load enwik8 dataset: 90M train, 5M validation."""
    with gzip.open(data_path) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        return torch.from_numpy(train_x), torch.from_numpy(valid_x)


class TextSamplerDataset(Dataset):
    """Random subsequence sampler from a byte tensor."""

    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len


def cycle(loader):
    """Infinite iterator over a DataLoader."""
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# ---------------------------------------------------------------------------
# Evaluation: Bits Per Character (BPC)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_bpc(ar_model, val_loader, num_eval_batches=50):
    """
    Compute bits per character (BPC) on validation set.
    BPC = cross_entropy_loss / ln(2)
    For char-level models, BPC ≡ BPB (bits per byte).
    """
    ar_model.eval()
    total_loss = 0.0
    total_tokens = 0

    for _ in range(num_eval_batches):
        data = next(val_loader)
        # AutoregressiveWrapper splits into x[:, :-1] and targets[:, 1:]
        loss = ar_model(data)
        batch_tokens = data.size(0) * (data.size(1) - 1)
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)
    return bpc


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model():
    """Build x-transformers decoder model wrapped for autoregressive training."""
    model = TransformerWrapper(
        num_tokens=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        post_emb_norm=True,  # LayerNorm after embeddings (BLOOM/YaLM-style)
        emb_frac_gradient=0.1,  # GLM-130B: reduce embedding gradient flow
        attn_layers=Decoder(
            dim=MODEL_DIM,
            depth=MODEL_DEPTH,
            heads=MODEL_HEADS,
            rotary_pos_emb=True,  # RoPE (standard for modern transformers)
            rotary_xpos=True,  # xPos: position-dependent scaling for long context
            attn_flash=True,  # PyTorch SDP flash attention
            attn_qk_norm=True,  # QK normalization for training stability
            attn_laser=True,  # LASER: gradient enhancement via exponentiated values
            ff_glu=True,  # Gated Linear Unit
            ff_swish=True,  # SwiGLU activation (PaLM/LLaMA-style)
            ff_glu_mult_bias=True,  # learnable bias in GLU gate
            use_rmsnorm=True,  # RMSNorm instead of LayerNorm
            add_value_residual=True,  # ResFormer value residuals
            shift_tokens=1,  # shift features by 1 token (helps char-level)
            softclamp_output=True,  # Gemma 2: soft-clamp final hidden states
            zero_init_branch_output=True,  # GPT-NeoX: zero-init output projections
        ),
    )

    # Wrap for autoregressive language modeling (handles input/target split + loss)
    ar_model = AutoregressiveWrapper(model)
    return ar_model


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------


def build_optimizer(model):
    """Build MuonAdamAtan2 optimizer (or fallback to AdamW)."""
    if HAS_MUON:
        # x-transformers TransformerWrapper exposes .muon_parameters()
        # which returns the linear weight matrices suitable for Muon
        inner_model = model.net  # unwrap AutoregressiveWrapper -> TransformerWrapper
        optimizer = MuonAdamAtan2(
            muon_params=inner_model.muon_parameters(),
            params=inner_model.parameters(),
            remove_muon_params_from_params=True,
            lr=LEARNING_RATE,
            weight_decay=0.001,  # very small decoupled WD
            decoupled_wd=True,
            betas=(0.92, 0.99),  # AdamAtan2 betas matching muon_beta1
            muon_rms_factor=0.1,  # smaller Muon updates (default 0.2)
            muon_beta1=0.92,  # less momentum than default 0.95
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    return optimizer


# ---------------------------------------------------------------------------
# LR Schedule (time-based, matching Karpathy's autoresearch pattern)
# ---------------------------------------------------------------------------


def get_lr_multiplier(progress):
    """Time-based LR schedule: warmup -> constant -> cosine cooldown."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Precision context manager
# ---------------------------------------------------------------------------


def get_precision_context():
    """Return the appropriate AMP context manager.

    FP8 note: torchao FP8 converts nn.Linear layers in-place to use FP8
    compute kernels. The autocast wrapper is still BF16 for non-linear ops.
    """
    if USE_FP16:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        # BF16 for both default and FP8 modes (FP8 is handled by torchao layer conversion)
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Estimate FLOPs
# ---------------------------------------------------------------------------


def estimate_flops_per_token(num_params, seq_len, depth, heads, dim):
    """Rough FLOPs per token estimate (forward + backward ≈ 6N + attention)."""
    head_dim = dim // heads
    attn_flops = 12 * heads * head_dim * seq_len * depth
    return 6 * num_params + attn_flops


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    if USE_FP8 and fp8_available:
        precision_tag = "FP8"
    elif USE_FP16:
        precision_tag = "FP16"
    else:
        precision_tag = "BF16"
    print(f"Precision: {precision_tag}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data
    print("Loading enwik8...")
    data_train, data_val = load_enwik8()
    train_dataset = TextSamplerDataset(data_train, MAX_SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, MAX_SEQ_LEN)
    train_loader = cycle(
        DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
    )
    val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True))

    # Model
    print(f"Building model: dim={MODEL_DIM}, depth={MODEL_DEPTH}, heads={MODEL_HEADS}")
    ar_model = build_model()
    ar_model.cuda()

    num_params = sum(p.numel() for p in ar_model.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # FP8: convert nn.Linear layers to use FP8 compute (before optimizer creation)
    if USE_FP8 and fp8_available:
        fp8_config = Float8LinearConfig(pad_inner_dim=True)
        convert_to_float8_training(ar_model.net.attn_layers, config=fp8_config)
        print("FP8: transformer attention/FFN layers converted to float8 training")

    flops_per_token = estimate_flops_per_token(
        num_params, MAX_SEQ_LEN, MODEL_DEPTH, MODEL_HEADS, MODEL_DIM
    )
    print(f"Estimated FLOPs per token: {flops_per_token:.2e}")

    # Optimizer
    optimizer = build_optimizer(ar_model)
    initial_lr = LEARNING_RATE

    # Compile model for speed (PyTorch 2.0+)
    try:
        ar_model = torch.compile(ar_model, dynamic=False)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: failed ({e}), running eager mode")

    # Effective batch size
    effective_batch_tokens = BATCH_SIZE * MAX_SEQ_LEN * GRADIENT_ACCUMULATE_EVERY
    print(
        f"Effective batch: {BATCH_SIZE} x {MAX_SEQ_LEN} x {GRADIENT_ACCUMULATE_EVERY} = {effective_batch_tokens:,} tokens"
    )
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Optimizer: {'MuonAdamAtan2' if HAS_MUON else 'AdamW'}")

    # Training loop
    t_start_training = time.time()
    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0
    precision_ctx = get_precision_context()

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        ar_model.train()

        # Gradient accumulation
        accumulated_loss = 0.0
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            with precision_ctx:
                loss = ar_model(next(train_loader))
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()
            accumulated_loss += loss.item()

        train_loss = accumulated_loss / GRADIENT_ACCUMULATE_EVERY

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(ar_model.parameters(), GRAD_CLIP)

        # LR schedule
        progress = (
            min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0.0
        )
        lrm = get_lr_multiplier(progress)
        current_lr = initial_lr * lrm
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Fast fail
        if train_loss > 100:
            print("\nFAIL: loss exploded")
            sys.exit(1)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        # Don't count first 5 warmup steps (compilation overhead)
        if step > 5:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(effective_batch_tokens / dt) if dt > 0 else 0
        mfu = (
            100 * flops_per_token * effective_batch_tokens / dt / GPU_BF16_PEAK_FLOPS
            if dt > 0
            else 0
        )
        remaining = max(0, TIME_BUDGET - total_training_time)
        bpc = debiased_loss / math.log(2)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.6f} | bpc: {bpc:.4f} | lr: {current_lr:.2e} | dt: {dt * 1000:.0f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}% | remain: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        # Validation
        if step % VALIDATE_EVERY == 0 and step > 0:
            with precision_ctx:
                val_bpc = evaluate_bpc(ar_model, val_loader)
            print(f"\n  [val] bpc: {val_bpc:.4f}")

        # Text generation
        if step % GENERATE_EVERY == 0 and step > 0:
            ar_model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f"\n  [gen] prompt: {prime[:80]}...")
            with precision_ctx:
                sample = ar_model.generate(
                    prompts=inp.unsqueeze(0),
                    seq_len=GENERATE_LENGTH,
                    cache_kv=True,
                )
            output_str = decode_tokens(sample[0].tolist())
            print(f"  [gen] output: {output_str[:200]}...")

        # GC management
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        # Time's up (only stop after warmup)
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r

    total_tokens = step * effective_batch_tokens

    # Final eval
    ar_model.eval()
    with precision_ctx:
        val_bpc = evaluate_bpc(ar_model, val_loader, num_eval_batches=100)

    # Final summary (matching Karpathy's output format for compatibility)
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    steady_state_mfu = (
        (
            100
            * flops_per_token
            * effective_batch_tokens
            * max(1, step - 5)
            / total_training_time
            / GPU_BF16_PEAK_FLOPS
        )
        if total_training_time > 0
        else 0
    )

    print("---")
    print(f"val_bpc:          {val_bpc:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {MODEL_DEPTH}")
    print(f"precision:        {precision_tag}")


if __name__ == "__main__":
    main()
