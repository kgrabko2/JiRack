# Copyright (c) 2025 CMS Manhattan
# All rights reserved.
# Author: Konstantin Vladimirovich Grabko
# Email: grabko@cmsmanhattan.com
# Phone: +1(516)777-0945
#
# This file is part of a project authored by CMS Manhattan. You may use, distribute, and modify
# this code under the terms of the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
# Please read <http://www.gnu.org/licenses/>.

# JiRackPyTorch 1B — final clean version, December 2025

"""
JiRackPyTorch 1B Model Definition
Complete and final version with SWA, RoPE Scaling, and full generative sampling.
FIXED: Test harness unpacking bug resolved.
"""
 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from pathlib import Path
import math
import torch.utils.checkpoint
 
# ========================================
# Model Configuration (Llama-Style 1B)
# ~0.94 B params
# ========================================
VOCAB_SIZE = 50257
MODEL_DIM = 2048
NUM_HEADS = 32
NUM_LAYERS = 16
MAX_SEQ_LEN = 2048  # Training length
FFN_HIDDEN_DIM = MODEL_DIM * 4
HEAD_DIM = MODEL_DIM // NUM_HEADS
EPSILON = 1e-6
DROPOUT_RATE = 0.1
 
# --- Sliding Window Attention Parameter ---
WINDOW_SIZE = 512  # The size of the local attention window and maximum KV cache size
# ---------------------------------------------
 
# --- 1. RMSNorm ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = EPSILON):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
 
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
 
    def forward(self, x):
        return self._norm(x) * self.weight
 
# --- 2. Rotary Positional Embedding (RoPE) with Context Scaling ---
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, max_seq_len: int = MAX_SEQ_LEN):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
 
    if seq_len > max_seq_len:
        scale_factor = seq_len / max_seq_len
        t = torch.arange(seq_len, dtype=torch.float32) / scale_factor
    else:
        t = torch.arange(seq_len, dtype=torch.float32)
 
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
 
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    return freqs_cis[None, None, :, None, :]
 
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    dtype = xq.dtype
 
    xq_f = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_f = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.view_as_complex(xq_f)
    xk_ = torch.view_as_complex(xk_f)
 
    freqs_cis_broadcast = reshape_for_broadcast(freqs_cis, xq_)
    xq_rot = xq_ * freqs_cis_broadcast.squeeze(3)
    xk_rot = xk_ * freqs_cis_broadcast.squeeze(3)
 
    xq_out = torch.view_as_real(xq_rot).flatten(3)
    xk_out = torch.view_as_real(xk_rot).flatten(3)
 
    return xq_out.type(dtype), xk_out.type(dtype)
 
# --- 3. MultiHeadAttention (SWA/SAPA Enabled, Cache Truncation Fixed) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, window_size: int = WINDOW_SIZE):
        super().__init__()
        self.q_proj = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.k_proj = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.v_proj = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.out_proj = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.scale = HEAD_DIM ** -0.5
        self.window_size = window_size
 
        self._build_rope_buffers(MAX_SEQ_LEN)
 
    def _build_rope_buffers(self, max_context_len: int):
        freqs_cis = precompute_freqs_cis(HEAD_DIM, max_context_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
 
    def forward(self, x: torch.Tensor, pos_offset: int, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        device = x.device
        B, T, D = x.shape
        dtype = x.dtype
 
        # --- Context Scaling (RoPE) Check and Update ---
        total_len = T + pos_offset
        if total_len > self.freqs_cis.size(0):
            new_freqs_cis = precompute_freqs_cis(HEAD_DIM, total_len).to(device)
            self.freqs_cis = new_freqs_cis
 
        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        current_k = self.k_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        current_v = self.v_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
 
        # Apply RoPE
        cur_freqs_cis = self.freqs_cis[pos_offset : pos_offset + T].to(device)
        q, k = apply_rotary_emb(q, current_k, cur_freqs_cis)
        v = current_v
 
        new_kv = None
 
        # --- Handle initialization and enforce WINDOW_SIZE truncation ---
        if past_kv is None or past_kv[0] is None:
            if T > self.window_size:
                new_kv = (k[:, :, -self.window_size:], v[:, :, -self.window_size:])
                k = k[:, :, -self.window_size:]
                v = v[:, :, -self.window_size:]
            else:
                new_kv = (k, v)
 
        elif past_kv[0] is not None:
            past_k, past_v = past_kv
            cache_len = past_k.size(2)
 
            sapa_start_idx = max(0, cache_len - (self.window_size - T))
 
            k_windowed = past_k[:, :, sapa_start_idx:, :]
            v_windowed = past_v[:, :, sapa_start_idx:, :]
 
            k = torch.cat([k_windowed, k], dim=2)
            v = torch.cat([v_windowed, v], dim=2)
 
            full_new_k = torch.cat([past_k, current_k], dim=2)
            full_new_v = torch.cat([past_v, current_v], dim=2)
 
            new_kv = (full_new_k[:, :, -self.window_size:], full_new_v[:, :, -self.window_size:])
 
        seqlen_k = k.size(2)
 
        # Attention in FP32 for stability
        q_stab = q.float()
        k_stab = k.float()
        v_stab = v.float()
 
        attn_weights = torch.matmul(q_stab, k_stab.transpose(-2, -1)) * self.scale
 
        # Causal Mask
        past_len_visible = seqlen_k - T
        mask = torch.full((T, seqlen_k), float('-inf'), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=past_len_visible + 1).unsqueeze(0).unsqueeze(0)
 
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
 
        out_raw = torch.matmul(attn_weights, v_stab)
        out = out_raw.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
 
        return out.type(dtype), new_kv
 
# --- 4. SwiGLU Feed-Forward ---
class SwiGLUFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(MODEL_DIM, FFN_HIDDEN_DIM, bias=False)
        self.w3 = nn.Linear(MODEL_DIM, FFN_HIDDEN_DIM, bias=False)
        self.w2 = nn.Linear(FFN_HIDDEN_DIM, MODEL_DIM, bias=False)
        self.dropout = nn.Dropout(DROPOUT_RATE)
 
    def forward(self, x):
        up_output = self.w3(x)
        gate_output = self.w1(x)
        swiglu_output = F.silu(gate_output) * up_output
        out = self.w2(swiglu_output)
        out = self.dropout(out)
        return out
 
# --- 5. Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention(window_size=WINDOW_SIZE)
        self.ffn = SwiGLUFeedForward()
        self.norm1 = RMSNorm(MODEL_DIM)
        self.norm2 = RMSNorm(MODEL_DIM)
        self.attn_dropout = nn.Dropout(DROPOUT_RATE)
 
    def forward(self, x, pos_offset: int, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
 
        if self.training and getattr(self, 'model', None) and self.model.gradient_checkpointing:
            if past_kv is None:
                def create_forward_function(attn, ffn, norm1, norm2, attn_dropout, pos_offset):
                    def forward_fn(x):
                        norm_x1 = norm1(x)
                        attn_out, _ = attn(norm_x1, pos_offset, None)
                        x = x + attn_dropout(attn_out)
 
                        norm_x2 = norm2(x)
                        x = x + ffn(norm_x2)
                        return x
                    return forward_fn
 
                x = torch.utils.checkpoint.checkpoint(
                    create_forward_function(self.attn, self.ffn, self.norm1, self.norm2, self.attn_dropout, pos_offset),
                    x, use_reentrant=False, preserve_rng_state=True
                )
                new_kv = None
            else:
                norm_x = self.norm1(x)
                attn_out, new_kv = self.attn(norm_x, pos_offset, past_kv)
                x = x + self.attn_dropout(attn_out)
 
                norm_x = self.norm2(x)
                x = x + self.ffn(norm_x)
 
        else:
            norm_x = self.norm1(x)
            attn_out, new_kv = self.attn(norm_x, pos_offset, past_kv)
            x = x + self.attn_dropout(attn_out)
 
            norm_x = self.norm2(x)
            x = x + self.ffn(norm_x)
 
        return x, new_kv
 
# --- 6. Main Model (JiRackPyTorch) - FINAL VERSION ---
class JiRackPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f = RMSNorm(MODEL_DIM)
        self.lm_head = nn.Linear(MODEL_DIM, VOCAB_SIZE, bias=False)
        self.emb_dropout = nn.Dropout(DROPOUT_RATE)
 
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_emb.weight
 
        self.gradient_checkpointing = False
 
        signature = "Konstantin V Grabko . original author 2025"
        self.register_buffer("proof_of_authorship_cmsmanhattan", torch.tensor([ord(c) for c in signature], dtype=torch.uint8), persistent=False)
        self.register_buffer("birth_date", torch.tensor([20251127], dtype=torch.int64), persistent=False)
 
        for block in self.blocks:
            object.__setattr__(block, 'model', self)
 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * NUM_LAYERS))
            if module.bias is not None:
                 torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
 
        if isinstance(module, nn.Linear) and hasattr(self, 'lm_head') and module is self.lm_head:
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
 
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
 
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
 
    def forward(self, input_ids: torch.Tensor, past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)
 
        pos_offset = 0
        if past_kv is not None and past_kv[0] is not None and past_kv[0][0] is not None:
            pos_offset = past_kv[0][0].size(2)
 
        new_kv_cache = [] if past_kv is not None else None
        current_past = past_kv
 
        for i, block in enumerate(self.blocks):
            layer_past = current_past[i] if current_past and i < len(current_past) else None
 
            x, layer_kv = block(x, pos_offset, layer_past)
 
            if new_kv_cache is not None and layer_kv is not None:
                new_kv_cache.append(layer_kv)
 
        x = self.ln_f(x)
        logits = self.lm_head(x)
 
        return logits if past_kv is None else (logits, new_kv_cache)
 
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.95, repetition_penalty: float = 1.0, do_sample: bool = True, eos_token_id: int = 50256) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
 
        # Prefill Step
        past_kv = [None] * NUM_LAYERS
        forward_output = self(input_ids, past_kv=past_kv)
 
        if isinstance(forward_output, tuple):
            if len(forward_output) != 2:
                raise ValueError(f"CRITICAL ERROR: forward returned {len(forward_output)} outputs in prefill.")
            logits, past_kv = forward_output
        else:
            logits = forward_output
            past_kv = [None] * NUM_LAYERS
 
        last_logits = logits[:, -1, :]
        output_ids = input_ids.clone()
 
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                unique_tokens = output_ids.unique()
                for token_id in unique_tokens:
                    tid = token_id.item()
                    if output_ids.tolist().count(tid) > 0:
                        log_prob = last_logits[:, tid]
                        last_logits[:, tid] = torch.where(log_prob > 0, log_prob / repetition_penalty, log_prob * repetition_penalty)
 
            if temperature == 0.0 or not do_sample:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                logits_temp = last_logits.float() / temperature
                probs = F.softmax(logits_temp, dim=-1)
 
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-9)
                    next_token_index = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, next_token_index)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
 
            if next_token.item() == eos_token_id:
                break
 
            output_ids = torch.cat([output_ids, next_token], dim=-1)
            next_input = next_token
 
            forward_output = self(next_input, past_kv=past_kv)
 
            if isinstance(forward_output, tuple):
                if len(forward_output) != 2:
                    raise ValueError(f"CRITICAL ERROR: forward returned {len(forward_output)} outputs in decode loop.")
                logits_out, past_kv = forward_output
            else:
                logits_out = forward_output
 
            last_logits = logits_out[:, -1, :]
 
        return output_ids.squeeze(0)
 
 
# === EXPORT SCRIPT (Testing SWA and Generation functionality) ===
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Creating 0.94B-parameter Llama-style model with SWA on {device}...")
 
    model = JiRackPyTorch().to(device)
    model.eval()
 
    # Ensure RoPE freqs are on the target device
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            if module.freqs_cis.device != device:
                module._build_rope_buffers(MAX_SEQ_LEN)
                module.freqs_cis = module.freqs_cis.to(device)
 
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model ready. Parameters: {total_params:.2f}B. SWA Window Size: {WINDOW_SIZE}")
 
    # --- SWA TEST ---
    print("\n--- Testing SWA/KV Cache Truncation (Inference) ---")
    large_input = torch.randint(0, VOCAB_SIZE, (1, WINDOW_SIZE * 2), device=device)
 
    with torch.no_grad():
        output = model(large_input, past_kv=[None] * NUM_LAYERS)
        if isinstance(output, tuple):
            logits_out, kv_cache = output
        else:
            logits_out = output
            kv_cache = [None] * NUM_LAYERS
 
    first_layer_cache_size = kv_cache[0][0].size(2) if kv_cache and kv_cache[0] is not None else 0
 
    print(f"Initial Prefill Length: {large_input.size(1)}. Cache Size after Prefill: {first_layer_cache_size}")
    if first_layer_cache_size == WINDOW_SIZE:
        print("✅ Cache Truncation (SWA) successful.")
    else:
        print(f"❌ Cache Truncation (SWA) failed. Expected {WINDOW_SIZE}, got {first_layer_cache_size}")
 
    single_token = torch.randint(0, VOCAB_SIZE, (1, 1), device=device)
    with torch.no_grad():
        output = model(single_token, past_kv=kv_cache)
        if isinstance(output, tuple):
            logits_out, final_kv_cache = output
        else:
            logits_out = output
            final_kv_cache = kv_cache
 
    final_cache_size = final_kv_cache[0][0].size(2) if final_kv_cache and final_kv_cache[0] is not None else 0
    print(f"Cache Size after 1 token generation: {final_cache_size}")
    if final_cache_size == WINDOW_SIZE:
        print("✅ SWA cache size remains fixed during generation.")
    else:
        print(f"❌ SWA cache size changed. Expected {WINDOW_SIZE}, got {final_cache_size}")
 
    # --- GENERATE TEST ---
    print("\n--- Testing Generation Loop ---")
 
    prompt = torch.randint(0, VOCAB_SIZE, (1, 10), device=device)
    max_tokens_to_generate = 20
 
    try:
        generated_ids = model.generate(
            prompt,
            max_new_tokens=max_tokens_to_generate,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=-1
        )
 
        generated_new_tokens = generated_ids.size(0) - prompt.size(1)
        print(f"Prompt length: {prompt.size(1)}")
        print(f"Generated new tokens: {generated_new_tokens}")
 
        if generated_new_tokens == max_tokens_to_generate:
            print("✅ Generation output length is correct.")
        else:
            print(f"⚠️ Generation stopped early (should not happen with eos_token_id=-1), got {generated_new_tokens} new tokens.")
 
        print("✅ Generation Test Succeeded (no errors)!")
 
    except Exception as e:
        print(f"❌ Generation Test Failed: {e}")
 
    state_dict_path = Path("models/jirack_swa_1b_class.state_dict.pt")
    state_dict_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_dict_path)
    print(f"\nFinal state_dict saved to → {state_dict_path}")