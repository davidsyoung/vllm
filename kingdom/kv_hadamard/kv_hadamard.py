# SPDX-License-Identifier: Apache-2.0
"""kv_hadamard — QuaRot-style Hadamard incoherence transform for the
GLM/DeepSeek MLA KV latent, applied as an EXACT load-time weight fold.

Goal: the KV cache stores the ROTATED latent  ĉ = H·c  so per-group KV
quantizers (nvfp4_ds_mla: 32 groups of 16 over the 512-dim latent;
nf3_ds_mla in the parallel build) see outlier-free, near-Gaussian groups.

Math (H = Sylvester Hadamard 512x512 / sqrt(512); orthogonal AND symmetric,
so H^T = H; applied via fp32 FWHT):

    serving path (vllm MultiHeadLatentAttentionWrapper.forward):
        kv_lora = fused_qkv_a_proj(h)[q_lora:]        # rows of W_DKV | W_KPE
        kv_c, k_pe = kv_lora.split([512, 64])
        c = RMSNorm_gamma(kv_c)                        # <- cached latent
        scores/out via W_UK / W_UV  (split from kv_b_proj at
        MLACommonImpl.process_weights_after_loading; decode absorbs
        q~ = W_UK^T q_nope, out = (sum_t p_t c_t) @ W_UV)

    fold (this module, on CHECKPOINT-name tensors in the weight stream):
      (1) gamma-fold:   kv_b_proj.weight <- kv_b_proj.weight @ diag(gamma)
                        kv_a_layernorm.weight <- ones
          (weightless RMSNorm commutes with any orthogonal rotation:
           ||Hx|| = ||x||, so RMSNorm_1(Hx) = H RMSNorm_1(x), eps included)
      (2) W_DKV rotate: kv_a_proj_with_mqa.weight[:512, :] <- H @ (...)
          (ONLY the 512 latent rows; the trailing 64 k_pe/RoPE rows are
           NEVER rotated)
      (3) W_UK|W_UV:    kv_b_proj.weight <- kv_b_proj.weight @ H^T (= @ H)
          (input-side 512 axis; kv_b_proj is per-head-fused
           [W_UK; W_UV] on the OUTPUT axis, so one transform covers both)

    exactness:  cached ĉ = H·(kv_c/rms(kv_c)) and every consumer weight
    carries diag(gamma)·H^T on its input side, hence
        (W·diag(gamma)·H^T)(H·n(x)) = W·(gamma ⊙ n(x))  — bit-for-the-math
    identical scores and outputs.  The DSA indexer takes wq_b(q_c) and
    wk_weights_proj(hidden_states) with its own fp8 K cache and never sees
    kv_c — verified in vllm/model_executor/layers/mla.py (forward) and
    deepseek_v2.py Indexer.forward — so it is unaffected by construction.

MXFP8 requant: non-expert weights are MXFP8 on disk for layers 1..78
(layer 0 attention is bf16).  The rev-3 hybrid loader dequants them to bf16
in-stream; after rotation the values leave the e8m0/32 grid, so matrices in
the mxfp8 tier are re-quantized here with the SAME block structure
(per-32-input-group e8m0 scale = floor(log2(amax)), e4m3 values — mirrors
vllm _mxfp8_e4m3_quantize_torch) and the per-matrix relative RMS error is
logged (expect ~1-3%).  With HYBRID_MXFP8_NATIVE=1 the online
Mxfp8OnlineLinearMethod then requantizes bit-exactly (fixed point), so the
logged error is the TOTAL added weight error.

MTP: the nextn draft layer (model.layers.78) has its own kv_a/kv_b/gamma
under the same names and receives the identical surgery via the same
patterns.  Skipping it would desynchronize draft scores from the rotated
target cache and collapse MTP acceptance.

Env:
  KV_HADAMARD=1                 enable (default 0 = exact passthrough)
  KV_HADAMARD_RANK=512          kv_lora_rank (rotated block length)
  KV_HADAMARD_ROPE=64           trailing k_pe rows of kv_a_proj_with_mqa
  KV_HADAMARD_REQUANT=tier      tier | all | off
  HYBRID_MXFP8_TIER_JSON=...    tier map path (same default as the loader)

Deployed as a site-packages overlay module imported by the rev-3
hybrid_loader get_all_weights hook (see kingdom/kv_hadamard/Dockerfile).
"""

from __future__ import annotations

import json
import math
import os
import re

import torch

MXFP8_BLOCK = 32
_DEF_TIER_JSON = "/opt/venv/lib/python3.12/site-packages/mxfp8_tier.json"

# Checkpoint tensor names (GLM-5.2 / GlmMoeDsa, DeepSeek V2/V3/V3.2 lineage).
# Matches main layers AND the MTP/nextn draft layer (own copies, same names).
_PAT = re.compile(
    r"^(?P<pfx>.*layers\.\d+\.self_attn\.)"
    r"(?P<which>kv_a_proj_with_mqa|kv_a_layernorm|kv_b_proj)\.weight$"
)


# --------------------------------------------------------------------------
# Hadamard
# --------------------------------------------------------------------------
def hadamard_matrix(
    n: int, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    """Sylvester Hadamard matrix H_n / sqrt(n) (orthogonal, symmetric)."""
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard size {n} is not a power of 2")
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat(
            (torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0
        )
    return h / math.sqrt(n)


def fwht(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fast Walsh-Hadamard transform along `dim`, Sylvester ordering,
    computed in fp32, normalized by 1/sqrt(n).

    fwht(x, -1) == x @ H_n  and  fwht(x, 0) == H_n @ x  with
    H_n = hadamard_matrix(n) (H symmetric, so this is also x @ H^T / H^T @ x).
    Returns a NEW fp32 tensor; the input is not modified.
    """
    xm = x.movedim(dim, -1)
    shape = xm.shape
    n = shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FWHT length {n} is not a power of 2")
    y = xm.to(torch.float32).reshape(-1, n).contiguous()
    if y.data_ptr() == x.data_ptr():
        y = y.clone()
    h = 1
    while h < n:
        y = y.view(-1, 2, h)
        even = y[:, 0, :] + y[:, 1, :]
        odd = y[:, 0, :] - y[:, 1, :]
        y = torch.stack((even, odd), 1).view(-1, n)
        h *= 2
    y = y.mul_(1.0 / math.sqrt(n))
    return y.view(shape).movedim(-1, dim)


# --------------------------------------------------------------------------
# MXFP8 requant (mirrors vllm _mxfp8_e4m3_quantize_torch: e8m0 scale =
# clamp(floor(log2(blockwise amax)) + 127, 0, 254), values fp8 e4m3)
# --------------------------------------------------------------------------
def _mxfp8_qd_once(x: torch.Tensor) -> torch.Tensor:
    o, i = x.shape
    xb = x.reshape(o, i // MXFP8_BLOCK, MXFP8_BLOCK)
    amax = xb.abs().amax(-1).clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = (torch.floor(torch.log2(amax)) + 127.0).clamp(0, 254)
    descale = torch.exp2(scale_biased - 127.0).unsqueeze(-1)
    q = (xb / descale).to(torch.float8_e4m3fn).to(torch.float32) * descale
    return q.reshape(o, i)


def mxfp8_requant(w32: torch.Tensor) -> tuple[torch.Tensor, float]:
    """fp32 [O, I] -> quantize to the MXFP8 e8m0/32 grid -> dequant fp32,
    ITERATED to the quantizer's fixed point (<=3 passes: only blocks whose
    max rounds up across a power of 2 rescale once, then stabilize).

    Same scale selection as the HYBRID_MXFP8_NATIVE online path
    (vllm _mxfp8_e4m3_quantize_torch), so its load-time requant of our
    output is bit-exact identity.  Returns (w_requant fp32, relative RMS
    error vs the input).
    """
    if w32.dtype != torch.float32 or w32.ndim != 2:
        raise ValueError(f"mxfp8_requant expects fp32 2D, got {w32.dtype} {w32.shape}")
    if w32.shape[1] % MXFP8_BLOCK:
        raise ValueError(f"input dim {w32.shape[1]} not divisible by {MXFP8_BLOCK}")
    q = w32
    for _ in range(4):
        q_next = _mxfp8_qd_once(q)
        if torch.equal(q_next, q):
            break
        q = q_next
    else:
        raise RuntimeError("[kv_hadamard] mxfp8 requant did not reach a fixed point")
    denom = w32.square().mean().sqrt().clamp(min=1e-30)
    err = float((q - w32).square().mean().sqrt() / denom)
    return q, err


# --------------------------------------------------------------------------
# The three folds (pure fp32 functions; unit-testable without vllm)
# --------------------------------------------------------------------------
def fold_dkv(w: torch.Tensor, rank: int, rope: int) -> torch.Tensor:
    """kv_a_proj_with_mqa.weight [rank+rope, hidden]:
    rotate ONLY the first `rank` output rows (the c_KV block): W <- H @ W.
    The trailing `rope` k_pe rows are returned bit-identical."""
    if w.ndim != 2 or w.shape[0] != rank + rope:
        raise ValueError(
            f"kv_a_proj_with_mqa shape {tuple(w.shape)} != ({rank}+{rope}, hidden)"
        )
    w32 = w.to(torch.float32)
    out = w32.clone()
    out[:rank] = fwht(w32[:rank], dim=0)
    return out


def fold_ukv(w: torch.Tensor, gamma: torch.Tensor, rank: int) -> torch.Tensor:
    """kv_b_proj.weight [n_heads*(qk_nope+v), rank] (fused [W_UK; W_UV] per
    head on the OUTPUT axis): W <- (W @ diag(gamma)) @ H^T on the INPUT axis.
    One transform covers UK and UV regardless of the fusion."""
    if w.ndim != 2 or w.shape[1] != rank:
        raise ValueError(f"kv_b_proj shape {tuple(w.shape)} has input dim != {rank}")
    if gamma.shape != (rank,):
        raise ValueError(f"gamma shape {tuple(gamma.shape)} != ({rank},)")
    w32 = w.to(torch.float32) * gamma.to(torch.float32).to(w.device).unsqueeze(0)
    return fwht(w32, dim=1)  # H symmetric: @H == @H^T


# --------------------------------------------------------------------------
# Weight-stream transform
# --------------------------------------------------------------------------
def _load_tier(tier_path: str | None) -> set[str] | None:
    path = tier_path or os.environ.get("HYBRID_MXFP8_TIER_JSON", _DEF_TIER_JSON)
    try:
        with open(path) as f:
            return set(json.load(f)["module_prefixes"])
    except Exception as e:  # noqa: BLE001 — degrade to no-requant, loudly
        print(f"[kv_hadamard] tier json unavailable ({path}: {e}) -> "
              "requant=tier matches NOTHING (rotated weights stay bf16)",
              flush=True)
        return None


def _norm_prefix(name: str) -> str:
    """model.layers.4.self_attn.kv_b_proj.weight -> layers.4.self_attn.kv_b_proj
    (same normalization as the hybrid loader's mxfp8 overlay)."""
    base = name[: -len(".weight")]
    i = base.find("layers.")
    return base[i:] if i >= 0 else base


def wrap(weights, rank: int = 512, rope: int = 64,
         requant: str = "tier", tier_path: str | None = None):
    """Generator: fold the Hadamard rotation into the (name, tensor) weight
    stream.  gamma/kv_b arrival order is arbitrary across shards; kv_b is
    buffered until its layer's gamma is seen (and vice versa).  Raises at end
    of stream if any pairing never completed (fail loud, never half-fold)."""
    if requant not in ("tier", "all", "off"):
        raise ValueError(f"requant={requant!r} not in tier|all|off")
    tier = _load_tier(tier_path) if requant == "tier" else None
    print(f"[kv_hadamard] ENABLED: H {rank}x{rank} (Sylvester/sqrt({rank})), "
          f"rope rows {rope} untouched, requant={requant}"
          + (f" ({len(tier)} tier prefixes)" if tier is not None else ""),
          flush=True)

    stats = {"n_a": 0, "n_b": 0, "errs": []}

    def _requant_maybe(name: str, w32: torch.Tensor) -> tuple[torch.Tensor, str]:
        do = requant == "all" or (
            requant == "tier" and tier is not None and _norm_prefix(name) in tier
        )
        if not do:
            return w32, "requant=skip(bf16)"
        q, err = mxfp8_requant(w32)
        stats["errs"].append(err)
        return q, f"requant-mxfp8 rel-RMS {err:.3e}"

    def _finish(name: str, t: torch.Tensor, w32: torch.Tensor, what: str):
        if not torch.isfinite(w32).all():
            raise RuntimeError(f"[kv_hadamard] non-finite values after fold: {name}")
        w32, note = _requant_maybe(name, w32)
        print(f"[kv_hadamard] {name}: {what}, {note}", flush=True)
        return name, w32.to(t.dtype)

    gammas: dict[str, torch.Tensor] = {}
    pend_b: dict[str, tuple[str, torch.Tensor]] = {}
    folded_b: set[str] = set()

    for name, t in weights:
        m = _PAT.match(name)
        if m is None:
            yield name, t
            continue
        pfx, which = m.group("pfx"), m.group("which")
        if which == "kv_a_proj_with_mqa":
            stats["n_a"] += 1
            yield _finish(name, t, fold_dkv(t, rank, rope),
                          f"H@W_DKV rows 0:{rank} of {rank + rope}")
        elif which == "kv_a_layernorm":
            g = t.detach().to(torch.float32).cpu()
            if g.shape != (rank,):
                raise RuntimeError(
                    f"[kv_hadamard] {name} shape {tuple(g.shape)} != ({rank},)")
            gammas[pfx] = g
            yield name, torch.ones_like(t)
            if pfx in pend_b:
                bn, bt = pend_b.pop(pfx)
                stats["n_b"] += 1
                folded_b.add(pfx)
                yield _finish(bn, bt, fold_ukv(bt, g, rank),
                              _ukv_what(g))
        else:  # kv_b_proj
            g = gammas.get(pfx)
            if g is None:
                pend_b[pfx] = (name, t)
            else:
                stats["n_b"] += 1
                folded_b.add(pfx)
                yield _finish(name, t, fold_ukv(t, g, rank), _ukv_what(g))

    if pend_b:
        raise RuntimeError(
            "[kv_hadamard] kv_b_proj seen but kv_a_layernorm (gamma) never "
            f"arrived for: {sorted(pend_b)}")
    unused = sorted(p for p in gammas if p not in folded_b)
    if unused:
        raise RuntimeError(
            "[kv_hadamard] gamma folded to ones but kv_b_proj never arrived "
            f"for: {unused}")
    errs = stats["errs"]
    print(f"[kv_hadamard] fold complete: {stats['n_a']} kv_a rotated, "
          f"{stats['n_b']} kv_b gamma-folded+rotated"
          + (f", requant rel-RMS mean {sum(errs) / len(errs):.3e} "
             f"max {max(errs):.3e} over {len(errs)} matrices" if errs else ""),
          flush=True)


def _ukv_what(g: torch.Tensor) -> str:
    return (f"(W@diag gamma)@H^T (gamma [{g.min():.3g},{g.max():.3g}])")


def maybe_wrap(weights):
    """Loader-hook entry point.  KV_HADAMARD unset/0 -> passthrough
    (byte-identical stream, zero cost)."""
    if os.environ.get("KV_HADAMARD", "0") != "1":
        return weights
    return wrap(
        weights,
        rank=int(os.environ.get("KV_HADAMARD_RANK", "512")),
        rope=int(os.environ.get("KV_HADAMARD_ROPE", "64")),
        requant=os.environ.get("KV_HADAMARD_REQUANT", "tier"),
    )


# --------------------------------------------------------------------------
# CPU self-test (used by the image import-smoke; full suite in
# kingdom/kv_hadamard/test_kv_hadamard.py)
# --------------------------------------------------------------------------
def _selftest(n: int = 512) -> str:
    torch.manual_seed(0)
    h = hadamard_matrix(n)
    if not torch.equal(fwht(torch.eye(n)), h):
        raise AssertionError("FWHT(I) != Sylvester H")
    if not torch.allclose(h @ h.T, torch.eye(n), atol=1e-5):
        raise AssertionError("H not orthogonal")
    x = torch.randn(3, n)
    eps = 1e-6
    rms = lambda v: v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)  # noqa: E731
    if not torch.allclose(rms(fwht(x)), fwht(rms(x)), atol=1e-5):
        raise AssertionError("weightless RMSNorm does not commute with H")
    w = torch.randn(64, n) * torch.exp(torch.randn(1, n))
    q, err = mxfp8_requant(w)
    q2, _ = mxfp8_requant(q)
    if not (1e-4 < err < 0.06):
        raise AssertionError(f"requant rel-RMS {err} out of expected band")
    if not torch.equal(q2, q):
        raise AssertionError("mxfp8 requant not idempotent")
    return f"kv_hadamard selftest OK (n={n}, requant rel-RMS {err:.3e})"


if __name__ == "__main__":
    print(_selftest())
