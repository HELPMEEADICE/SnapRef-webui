# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ArcAnima/SnapRef comfyui_nodes/snapref/__init__.py (Apache-2.0).
# Modified 2026-11-12: request-scoped state, exact CPU fidelity, OOM propagation.
"""Asymmetric weighted attention, [batch, tokens, heads, head_dim]."""
import torch
import torch.nn.functional as F


def merge_lse(parts):
    """Each part is (log bias, FP32 output, FP32 logsumexp)."""
    logits = [lse.float() + bias for bias, _, lse in parts]
    maximum = torch.stack([x.expand_as(logits[0]) for x in logits]).amax(0)
    weights = [torch.exp(x - maximum) for x in logits]
    return sum(w * part[1].float() for w, part in zip(weights, parts)) / sum(weights)


def fused_lse(q, k, v):
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("fused LSE requires CUDA FP16/BF16")
    try:
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[:2]
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, AttributeError, NotImplementedError):
        out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)[:2]
    return out.float(), lse.float()[..., :q.shape[2], None]


def masked_attention(q, k, v, refs, sizes, biases, row_bias):
    batch, tokens, heads, dim = q.shape
    kr, vr = (t.to(q).expand(batch, -1, -1, -1) for t in refs)
    all_k, all_v = torch.cat((kr, k), 1), torch.cat((vr, v), 1)
    bias = torch.repeat_interleave(torch.as_tensor(biases, device=q.device, dtype=torch.float32),
                                   torch.as_tensor(sizes, device=q.device))
    bias = bias[None, :] + row_bias[:, None]
    bias = F.pad(bias, (0, k.shape[1])).to(q.dtype)[:, None, None, :]
    # Keep mask broadcastable; materialising B*H*Q*K would exhaust VRAM.
    out = F.scaled_dot_product_attention(q.transpose(1, 2), all_k.transpose(1, 2), all_v.transpose(1, 2), attn_mask=bias)
    return out.transpose(1, 2).reshape(batch, tokens, heads * dim)


def weighted_attention(attn, q, k, v, refs, sizes, biases, row_bias, options):
    """row_bias=-inf means no reference, 0 means trained reference weight."""
    batch, tokens, heads, dim = q.shape
    if torch.isneginf(row_bias).all():
        return attn.torch_attention_op(q, k, v, transformer_options=options)
    if all(b == 0 for b in biases) and bool((row_bias == 0).all()):
        kr, vr = (t.to(q).expand(batch, -1, -1, -1) for t in refs)
        return attn.torch_attention_op(q, torch.cat((kr, k), 1), torch.cat((vr, v), 1), transformer_options=options)
    try:
        out_t, lse_t = fused_lse(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        parts = [(0.0, out_t, lse_t)]
        start = 0
        for size, bias in zip(sizes, biases):
            kr, vr = (t[:, start:start + size].to(q).expand(batch, -1, -1, -1).transpose(1, 2) for t in refs)
            out, lse = fused_lse(q.transpose(1, 2), kr, vr)
            parts.append((bias + row_bias[:, None, None, None], out, lse))
            start += size
        out = merge_lse(parts).to(q.dtype)
        return out.transpose(1, 2).reshape(batch, tokens, heads * dim)
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, AttributeError, NotImplementedError):
        return masked_attention(q, k, v, refs, sizes, biases, row_bias)
