# SPDX-License-Identifier: Apache-2.0
"""Pure-torch unit tests for kv_hadamard (no vllm import, CPU only).

Run:  python kingdom/kv_hadamard/test_kv_hadamard.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_hadamard as kh  # noqa: E402

torch.manual_seed(1234)

# GLM-5.2 / GlmMoeDsa dims (heads reduced 64 -> 4 for speed; all folded axes
# are at REAL size: rank 512, rope 64, qk_nope 192, v 256).
RANK, ROPE, P, V, HEADS, HIDDEN, T = 512, 64, 192, 256, 4, 768, 9
EPS = 1e-6


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """vLLM RMSNorm: x * rsqrt(mean(x^2) + eps) * weight (fp32)."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w


def _mk_weights():
    """Random model with gamma outliers (lognormal) and heavy-tailed W rows —
    the regime Hadamard incoherence is for."""
    w_dkv = torch.randn(RANK + ROPE, HIDDEN) / math.sqrt(HIDDEN)
    w_dkv[:RANK] *= torch.exp(0.9 * torch.randn(RANK, 1))  # channel outliers
    gamma = torch.exp(0.8 * torch.randn(RANK))
    w_kvb = torch.randn(HEADS * (P + V), RANK) / math.sqrt(RANK)
    return w_dkv, gamma, w_kvb


def _mla_reference(h, w_dkv, gamma, w_kvb, q_nope, q_pe):
    """Unfolded baseline: materialized k/v (the vLLM prefill route)."""
    lat = h @ w_dkv.T
    kv_c, k_pe = lat[:, :RANK], lat[:, RANK:]
    c = rmsnorm(kv_c, gamma)
    kv = (c @ w_kvb.T).view(T, HEADS, P + V)
    k_nope, v = kv.split([P, V], dim=-1)
    scores = (
        torch.einsum("qhp,khp->hqk", q_nope, k_nope)
        + torch.einsum("qhr,kr->hqk", q_pe, k_pe)
    ) / math.sqrt(P + ROPE)
    probs = scores.softmax(-1)
    out = torch.einsum("hqk,khv->qhv", probs, v)
    return c, k_pe, scores, out


def _mla_absorbed(h, w_dkv, gamma, w_kvb, q_nope, q_pe):
    """Folded-weights path exactly as vLLM decode consumes them:
    W_UK/W_UV split from (transformed) kv_b_proj, q~ = bmm(q_nope, W_UK_T),
    scores against the CACHED (rotated) latent, out = bmm(latent, W_UV)."""
    lat = h @ w_dkv.T
    kv_c, k_pe = lat[:, :RANK], lat[:, RANK:]
    c = rmsnorm(kv_c, gamma)  # gamma == ones after fold
    wt = w_kvb.T.view(RANK, HEADS, P + V)  # mirrors mla_attention.py:996
    w_uk, w_uv = wt.split([P, V], dim=-1)  # [512,N,P], [512,N,V]
    ql = torch.einsum("qhp,lhp->hql", q_nope, w_uk)  # bmm(q, W_UK_T)
    scores = (
        torch.einsum("hql,kl->hqk", ql, c)
        + torch.einsum("qhr,kr->hqk", q_pe, k_pe)
    ) / math.sqrt(P + ROPE)
    probs = scores.softmax(-1)
    lat_o = torch.einsum("hqk,kl->hql", probs, c)  # attn over cached latent
    out = torch.einsum("hql,lhv->qhv", lat_o, w_uv)  # bmm(latent, W_UV)
    return c, k_pe, scores, out


def _run_stream(tensors, **kw):
    """Push (name, tensor) pairs through kh.wrap and return dict of outputs."""
    return dict(kh.wrap(iter(tensors), rank=RANK, rope=ROPE, **kw))


L4 = "model.layers.4.self_attn."
L78 = "model.layers.78.self_attn."  # MTP/nextn draft layer


def test_fwht_matches_dense_sylvester():
    for n in (2, 8, 64, 512):
        hm = kh.hadamard_matrix(n)
        assert torch.equal(kh.fwht(torch.eye(n)), hm), f"fwht(I)!=H n={n}"
        assert torch.allclose(hm @ hm.T, torch.eye(n), atol=1e-5), "H not orthogonal"
        assert torch.equal(hm, hm.T), "Sylvester H not symmetric"
        x = torch.randn(5, n)
        assert torch.allclose(kh.fwht(x, -1), x @ hm, atol=1e-4)
        assert torch.allclose(kh.fwht(x.T, 0), hm @ x.T, atol=1e-4)
        # involution: H @ H == I  =>  fwht(fwht(x)) == x
        assert torch.allclose(kh.fwht(kh.fwht(x)), x, atol=1e-4)
    print("PASS fwht_matches_dense_sylvester")


def test_weightless_rmsnorm_commutes():
    x = torch.randn(17, RANK) * torch.exp(torch.randn(1, RANK))
    ones = torch.ones(RANK)
    lhs = rmsnorm(kh.fwht(x), ones)
    rhs = kh.fwht(rmsnorm(x, ones))
    assert torch.allclose(lhs, rhs, atol=1e-5), "weightless RMSNorm must commute"
    # and the WEIGHTED norm must NOT commute (why the gamma-fold exists):
    gamma = torch.exp(torch.randn(RANK))
    assert not torch.allclose(
        rmsnorm(kh.fwht(x), gamma), kh.fwht(rmsnorm(x, gamma)), atol=1e-2
    ), "weighted RMSNorm unexpectedly commuted — test is vacuous"
    print("PASS weightless_rmsnorm_commutes")


def test_end_to_end_fold_exact_fused_kvb():
    """Folded weights + rotated cache == unfolded baseline, via BOTH consumer
    routes (materialized prefill AND absorbed decode), fused kv_b_proj."""
    w_dkv, gamma, w_kvb = _mk_weights()
    h = torch.randn(T, HIDDEN)
    q_nope = torch.randn(T, HEADS, P)
    q_pe = torch.randn(T, HEADS, ROPE)  # post-RoPE values; k_pe rows untouched

    c0, kpe0, s0, o0 = _mla_reference(h, w_dkv, gamma, w_kvb, q_nope, q_pe)

    out = _run_stream(
        [
            (L4 + "kv_a_proj_with_mqa.weight", w_dkv.clone()),
            (L4 + "kv_a_layernorm.weight", gamma.clone()),
            (L4 + "kv_b_proj.weight", w_kvb.clone()),
        ],
        requant="off",
    )
    w_dkv_f = out[L4 + "kv_a_proj_with_mqa.weight"]
    gamma_f = out[L4 + "kv_a_layernorm.weight"]
    w_kvb_f = out[L4 + "kv_b_proj.weight"]
    assert torch.equal(gamma_f, torch.ones(RANK)), "gamma must become ones"
    assert torch.equal(w_dkv_f[RANK:], w_dkv[RANK:]), "RoPE rows must be bit-identical"

    for route, fn in (("materialized", _mla_reference), ("absorbed", _mla_absorbed)):
        c1, kpe1, s1, o1 = fn(h, w_dkv_f, gamma_f, w_kvb_f, q_nope, q_pe)
        assert torch.equal(kpe1, kpe0), f"k_pe changed ({route})"
        assert torch.allclose(s1, s0, atol=2e-4), (
            f"scores mismatch ({route}): {(s1 - s0).abs().max()}"
        )
        assert torch.allclose(o1, o0, atol=2e-4), (
            f"outputs mismatch ({route}): {(o1 - o0).abs().max()}"
        )
        # cached latent is the ROTATED weightless-normalized latent:
        c_wl = rmsnorm((h @ w_dkv.T)[:, :RANK], torch.ones(RANK))
        assert torch.allclose(c1, kh.fwht(c_wl), atol=1e-4), "cache != H@c"
    print(f"PASS end_to_end_fold_exact_fused_kvb "
          f"(max score delta {(s1 - s0).abs().max():.2e})")


def test_stream_pairing_both_orders_and_mtp():
    w_dkv, gamma, w_kvb = _mk_weights()
    direct = kh.fold_ukv(w_kvb, gamma, RANK)
    # order A: gamma then kv_b; order B (other shard order): kv_b then gamma;
    # MTP layer 78 gets the identical surgery; non-matching names pass through.
    passthru = torch.randn(8, 8)
    for order in ("gamma_first", "kvb_first"):
        seq = [
            ("model.layers.4.mlp.gate.weight", passthru),
            (L4 + "kv_a_layernorm.weight", gamma.clone()),
            (L4 + "kv_b_proj.weight", w_kvb.clone()),
            (L78 + "kv_b_proj.weight", w_kvb.clone()),
            (L78 + "kv_a_layernorm.weight", gamma.clone()),
            (L78 + "kv_a_proj_with_mqa.weight", w_dkv.clone()),
        ]
        if order == "kvb_first":
            seq[1], seq[2] = seq[2], seq[1]
        out = _run_stream(seq, requant="off")
        assert torch.equal(out["model.layers.4.mlp.gate.weight"], passthru)
        for pfx in (L4, L78):
            assert torch.allclose(out[pfx + "kv_b_proj.weight"], direct, atol=1e-6), (
                f"{order} {pfx} kv_b fold mismatch"
            )
            assert torch.equal(out[pfx + "kv_a_layernorm.weight"], torch.ones(RANK))
        assert torch.allclose(
            out[L78 + "kv_a_proj_with_mqa.weight"], kh.fold_dkv(w_dkv, RANK, ROPE)
        )
    print("PASS stream_pairing_both_orders_and_mtp (incl. MTP layer 78)")


def test_missing_pair_raises():
    _, gamma, w_kvb = _mk_weights()
    for seq, what in (
        ([(L4 + "kv_b_proj.weight", w_kvb)], "kv_b without gamma"),
        ([(L4 + "kv_a_layernorm.weight", gamma)], "gamma without kv_b"),
    ):
        try:
            _run_stream(seq, requant="off")
        except RuntimeError:
            continue
        raise AssertionError(f"{what} did not raise")
    print("PASS missing_pair_raises")


def test_requant_band_and_idempotent():
    w_dkv, gamma, w_kvb = _mk_weights()
    rot = kh.fold_ukv(w_kvb, gamma, RANK)
    q1, err = kh.mxfp8_requant(rot)
    assert 5e-4 < err < 0.06, f"requant rel-RMS {err} outside ~1-3% band"
    q2, err2 = kh.mxfp8_requant(q1)
    assert torch.equal(q2, q1), "requant not idempotent (fixed point broken)"
    assert err2 == 0.0, f"second-pass requant error {err2} != 0"
    # grid values survive the bf16 leg the loader stream uses:
    q_bf = q1.to(torch.bfloat16).to(torch.float32)
    q3, _ = kh.mxfp8_requant(q_bf)
    assert torch.equal(q3, q_bf), "bf16 leg broke mxfp8 grid alignment"
    # tier gating: only matrices in the tier json are requantized
    import json
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"module_prefixes": ["layers.4.self_attn.kv_b_proj"]}, f)
        tier_path = f.name
    out = _run_stream(
        [
            (L4 + "kv_a_layernorm.weight", gamma.clone()),
            (L4 + "kv_b_proj.weight", w_kvb.clone()),
            (L4 + "kv_a_proj_with_mqa.weight", w_dkv.clone()),
        ],
        requant="tier",
        tier_path=tier_path,
    )
    os.unlink(tier_path)
    assert torch.equal(out[L4 + "kv_b_proj.weight"], q1), "tier member not requantized"
    assert torch.equal(
        out[L4 + "kv_a_proj_with_mqa.weight"], kh.fold_dkv(w_dkv, RANK, ROPE)
    ), "non-tier matrix must stay pure rotated fp32/bf16"
    print(f"PASS requant_band_and_idempotent (rel-RMS {err:.3e})")


def test_fold_exact_with_requant_error_bounded():
    """With requant=all the end-to-end error must be small and attributable
    to the logged weight-side requant RMS (not to the rotation)."""
    w_dkv, gamma, w_kvb = _mk_weights()
    h = torch.randn(T, HIDDEN)
    q_nope = torch.randn(T, HEADS, P)
    q_pe = torch.randn(T, HEADS, ROPE)
    _, _, s0, o0 = _mla_reference(h, w_dkv, gamma, w_kvb, q_nope, q_pe)
    out = _run_stream(
        [
            (L4 + "kv_a_proj_with_mqa.weight", w_dkv.clone()),
            (L4 + "kv_a_layernorm.weight", gamma.clone()),
            (L4 + "kv_b_proj.weight", w_kvb.clone()),
        ],
        requant="all",
    )
    _, _, s1, o1 = _mla_absorbed(
        h,
        out[L4 + "kv_a_proj_with_mqa.weight"],
        out[L4 + "kv_a_layernorm.weight"],
        out[L4 + "kv_b_proj.weight"],
        q_nope,
        q_pe,
    )
    rel = (o1 - o0).norm() / o0.norm()
    assert rel < 0.08, f"requant=all end-to-end rel err {rel} too large"
    assert rel > 1e-5, "requant=all had no effect — gating broken?"
    print(f"PASS fold_exact_with_requant_error_bounded (out rel {rel:.3e})")


def test_group_outlier_reduction():
    """The point of the exercise: per-16-group (nvfp4_ds_mla layout) dynamic
    range of the cached latent shrinks after rotation."""
    w_dkv, gamma, w_kvb = _mk_weights()
    h = torch.randn(4096, HIDDEN)
    kv_c = (h @ w_dkv.T)[:, :RANK]
    ones = torch.ones(RANK)
    c_plain = rmsnorm(kv_c, gamma)  # what an UNROTATED cache stores
    c_rot = kh.fwht(rmsnorm(kv_c, ones))  # what the ROTATED cache stores

    def group_spread(c):  # amax/rms per 16-group, worst group (fp4 pain metric)
        g = c.reshape(c.shape[0], RANK // 16, 16)
        return (g.abs().amax(-1) / g.square().mean(-1).sqrt().clamp(min=1e-9)).max()

    def kurt(c):
        z = (c - c.mean()) / c.std()
        return (z**4).mean()

    s_plain, s_rot = group_spread(c_plain), group_spread(c_rot)
    k_plain, k_rot = kurt(c_plain), kurt(c_rot)
    assert s_rot < s_plain, "rotation did not reduce worst-group spread"
    assert k_rot < k_plain, "rotation did not reduce kurtosis"
    print(
        f"PASS group_outlier_reduction: worst 16-group amax/rms "
        f"{s_plain:.2f} -> {s_rot:.2f}, kurtosis {k_plain:.2f} -> {k_rot:.2f}"
    )


if __name__ == "__main__":
    test_fwht_matches_dense_sylvester()
    test_weightless_rmsnorm_commutes()
    test_end_to_end_fold_exact_fused_kvb()
    test_stream_pairing_both_orders_and_mtp()
    test_missing_pair_raises()
    test_requant_band_and_idempotent()
    test_fold_exact_with_requant_error_bounded()
    test_group_outlier_reduction()
    print(kh._selftest())
    print("ALL kv_hadamard TESTS PASSED")
