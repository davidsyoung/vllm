# kingdom-plus5 — commits dropped in the v14 rebase (8722ac7 → 56a5c3e)

Branch story: `kingdom-plus` carried 7 commits on the eldritch base `8722ac7`.
`kingdom-plus5` rebases them onto `dev/eldritch-enlightenment @ 56a5c3e`
(local-inference-lab/vllm). 6 of 7 were re-applied; 1 was dropped:

## e448767 — cherry: vllm#47599 "warm up FlashInfer b12x NVFP4 MoE" — DROPPED

Superseded by the new base's own warmup framework:

- `warmup_b12x_moe_dynamic()` (`fused_moe/b12x_moe.py`, upstream `7745b75`
  "Warm all B12X MoE planner regimes") is wired into
  `warmup/kernel_warmup.py` and warms every `B12xExperts` signature over
  powers-of-two + cudagraph/compile/serving token sizes up to max — the
  `--moe-backend b12x` path we actually serve GLM-5.2 with.
- The new base additionally ships `warmup/b12x_sparse_indexer_warmup.py`,
  `warmup_b12x_mxfp8_linear`, and DCP-A2A collective warmup.
- Our cherry warmed only `FlashInferB12xExperts` — the *separate*
  `--moe-backend flashinfer_b12x` backend, which kingdom serving does not
  use on this base. Its JIT-stall problem class (small-batch kernels
  invalidated by workspace growth) is addressed for the b12x backend by the
  planned-FP4-weights refactor (`3e407fd`) + planner-regime warmup.

Revisit only if we ever serve `--moe-backend flashinfer_b12x` on this base;
the original commit remains on `kingdom-plus` (e448767).
