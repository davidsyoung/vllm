# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone Qwen3 DSpark speculator for THIS fork's V2 GPU model runner.

DSpark = DFlash parallel block drafting + a sequential low-rank Markov logit
bias. This runtime is the piece that was MISSING on this fork: previously the
standalone ``Qwen3DSparkModel`` was routed to the stock ``DFlashSpeculator``,
whose ``sample_draft`` never consults the Markov head, so the draft degraded to
plain DFlash block drafting. This speculator makes the Markov head actually
fire, matching upstream vLLM's dedicated DSpark runtime (PR #47093 @ ``2b753ad``,
``vllm/v1/worker/gpu/spec_decode/dspark/speculator.py``).

It is a thin subclass of the fork's ``DFlashSpeculator``:

  * The parallel backbone forward, context-KV precompute, the ``prepare_dflash_
    inputs`` kernel, the ``DFlashCudaGraphManager`` (FULL / FULL_DECODE_ONLY),
    attention wiring and every buffer (``sample_indices`` / ``sample_pos`` /
    ``sample_idx_mapping``, laid out ``(req, step)``) are inherited UNCHANGED.
    Our speculators-format checkpoint uses the DFlash ``1 + N`` fill-in block
    (``dspark_bonus_anchor=True``: anchor is the bonus token at query offset 0),
    which is exactly the layout the fork's DFlash input prep already produces.

  * The ONLY override is the sampling stage. DFlash samples all ``N`` mask
    hidden states in a single parallel Gumbel pass. DSpark instead samples
    left-to-right, adding a Markov bias derived from the previously sampled
    token at each of the ``N`` steps (see ``_sample_sequential``). There is NO
    extra backbone forward: the bias is a cheap ``V x r`` / ``r x V`` transition
    applied to the already-computed per-step base logits.

The Markov math replicates upstream ``DSparkSpeculator._sample_sequential``
verbatim, with ONE fork adaptation: the Gumbel key uses ``sample_pos + 1`` (the
fork's DFlash convention -- see ``DraftModelSpeculator.sample_draft``, whose
comment is "We must add 1 to the positions to match the Gumbel noise used for
draft and target sampling") rather than upstream's ``sample_pos - 1``. This
matches THIS fork's target verifier; getting it wrong silently tanks the
acceptance rate (output stays correct because the target verifies every token).
See IMPL_NOTES.md.
"""

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)


class Qwen3DSparkSpeculator(DFlashSpeculator):
    """DFlash block backbone + sequential Markov logit-bias sampling."""

    _speculator_name = "Qwen3DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)

        hf_config = self.draft_model_config.hf_config

        # This fork drives DSpark through the DFlash 1+N fill-in block (the
        # anchor is the bonus token at query offset 0). The alternative
        # upstream "sample_from_anchor" N-slot layout would need a different
        # input-prep kernel than the fork's DFlash one, so it is not supported
        # here. Speculators-format checkpoints set dspark_bonus_anchor=True.
        if not getattr(hf_config, "dspark_bonus_anchor", True):
            raise NotImplementedError(
                "Qwen3DSparkSpeculator on this fork only supports the DFlash "
                "1+N fill-in block (dspark_bonus_anchor=True). The upstream "
                "sample-from-anchor N-slot layout is not wired to the fork's "
                "DFlash input-prep kernel."
            )

        # A rank-0 checkpoint has no Markov head -> it is plain DFlash; refuse
        # it here so the head firing is never silently skipped. Use
        # method='dflash' for such a checkpoint instead.
        markov_rank = int(getattr(hf_config, "markov_rank", 0) or 0)
        if markov_rank <= 0:
            raise ValueError(
                "Qwen3DSparkSpeculator requires markov_rank > 0 (the DSpark "
                "Markov head). A rank-0 checkpoint is plain DFlash; use "
                "method='dflash' instead."
            )

        # super().__init__ (DFlash) already set num_query_per_req = 1 + N and
        # allocated the (req, step)-ordered sample buffers; we reuse them.

        # Per-step column index into draft_logits[:, step, :]. 0-dim slices
        # (self._step_cols[i]) keep gumbel_sample in its scalar-column path,
        # which stays CUDA-graph capturable (no per-token gather).
        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        # Persistent index of each request's anchor (bonus) token in the flat
        # query buffer: query offset 0 of request r lives at
        # r * num_query_per_req. Fixed-address buffer so it is safe to read
        # inside a captured graph.
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting support. Unused for full-vocab
        # GLM-5.2 (draft_id_to_target_id is None); populated in load_draft_model
        # when the checkpoint ships a d2t table.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

        # ---- diagnostic overrides (read once at init -> CUDA-graph safe) ----
        # These exist to bisect the acceptance-rate bug empirically on GPU
        # without rebuilding. Each is a constant at capture time, so the branch
        # it controls bakes into the captured graph.
        #
        # VLLM_DSPARK_GUMBEL_POS_OFFSET (default 1): the draft Gumbel key is
        # `sample_pos + offset`. NOTE: the standard/probabilistic rejection
        # sampler keys acceptance on the TARGET position (rejection_sampler_
        # utils.py) and tests p_target/p_draft, so this offset changes only WHICH
        # valid draft token is proposed, not the acceptance rate. Exposed to
        # confirm that empirically (+1 = fork DFlash convention, -1 = upstream).
        self._gumbel_pos_offset = int(
            os.getenv("VLLM_DSPARK_GUMBEL_POS_OFFSET", "1")
        )
        # VLLM_DSPARK_DISABLE_MARKOV (default 0): when 1, skip the Markov bias
        # (logits_i = base_logits[:, i]) so the draft degrades to plain DFlash
        # block drafting. This isolates the head: if acceptance jumps to
        # DFlash-level with it set, the bug is in the Markov head (weights /
        # scale / bias math); if acceptance stays ~0, the bug is upstream of the
        # head (aux capture / backbone / base logits).
        self._disable_markov = (
            os.getenv("VLLM_DSPARK_DISABLE_MARKOV", "0") == "1"
        )
        # VLLM_DSPARK_MARKOV_SCALE (default 1.0, may be fractional or negative):
        # `logits_i = base_logits[:, i] + scale * markov_bias(prev)`. The Markov
        # math here is byte-identical to upstream 2b753ad AND current main (model
        # + runtime), yet on this fork/checkpoint the bias DEGRADES healthy
        # DFlash logits at every position (accept-len 2.10 < 2.61) -> the trained
        # base:bias magnitude balance is wrong. This knob both classifies and
        # fixes it in one GPU sweep:
        #   0.0        -> bias off (== plain DFlash baseline, ~2.61)
        #   0 < s < 1  best AND > 2.61 -> magnitude/scale bug; `s` is the fix
        #   s ~= 1.0   best but < 2.61 -> bias miscomputed (sign/prev/vocab/basis)
        #   s < 0      best            -> sign/direction inverted
        # scale=1.0 reproduces current behavior exactly (`1.0 * bias == bias`).
        self._markov_scale = float(os.getenv("VLLM_DSPARK_MARKOV_SCALE", "1.0"))
        # VLLM_DSPARK_DEBUG_MARKOV=1: log |base| vs |bias| at step 0. Only fires
        # in EAGER executions (guarded by is_current_stream_capturing so the
        # .item() syncs never enter a captured graph). Run the draft with
        # enforce-eager to see real-data magnitudes; bias >> base confirms scale.
        self._debug_markov = (
            os.getenv("VLLM_DSPARK_DEBUG_MARKOV", "0") == "1"
        )
        logger.info(
            "Qwen3DSparkSpeculator diagnostics: gumbel_pos_offset=%d, "
            "disable_markov=%s, markov_scale=%.4f, debug_markov=%s (aux layer "
            "shift is set in update_dspark via VLLM_DSPARK_AUX_LAYER_SHIFT).",
            self._gumbel_pos_offset,
            self._disable_markov,
            self._markov_scale,
            self._debug_markov,
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        # Reuse the fork's DFlash loader unchanged: it resolves the registry
        # arch (Qwen3DSparkModel -> Qwen3DSparkForCausalLM), shares the target's
        # embed_tokens / lm_head, and loads an optional mask_embedding.pt. The
        # Markov head rides inside self.model and is loaded by the model's own
        # load_weights. This is the identical, already-validated DFlash load
        # path -- only the sampling stage differs.
        model = super().load_draft_model(target_model, target_attn_layer_names)

        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter draft logits into target vocab before
        # sampling. Full-vocab drafts (draft_id_to_target_id is None) skip this.
        if (
            self.draft_logits is not None
            and getattr(model, "draft_id_to_target_id", None) is not None
        ):
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _log_markov_magnitudes(
        self, base0: torch.Tensor, bias0: torch.Tensor
    ) -> None:
        """Eager-only: report |base| vs |bias| at step 0 to expose a scale bug.

        Never called inside a captured graph (guarded by the caller on
        ``is_current_stream_capturing``); the ``.item()`` syncs here are safe
        only in eager execution.
        """
        with torch.no_grad():
            base_am = base0.abs().mean().item()
            base_mx = base0.abs().amax().item()
            bias_am = bias0.abs().mean().item()
            bias_mx = bias0.abs().amax().item()
        logger.info(
            "DSpark markov debug (eager): |base0| mean=%.4f max=%.4f ; "
            "|bias0| mean=%.4f max=%.4f ; bias/base mean-ratio=%.3f ; "
            "markov_scale=%.4f (bias >> base => scale bug)",
            base_am,
            base_mx,
            bias_am,
            bias_mx,
            bias_am / max(base_am, 1e-9),
            self._markov_scale,
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Sequential Markov sampling over the backbone's block hidden states.

        Runs inside the captured FULL graph. The loop is over the Python
        constant ``num_speculative_steps`` (no ``.item()`` / no tensor-value
        branching), so it unrolls into a fixed, capturable op sequence.
        """
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec

        # Per-(req, step) mask hidden states, ordered (req, step) exactly as the
        # fork's DFlash prepare kernel wrote sample_indices.
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]

        # Draft-vocab base logits (no d2t scatter): the Markov bias is added in
        # draft space; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Step 0 conditions the Markov bias on the anchor (bonus) token, read
        # from the fixed query buffer at each request's query offset 0.
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            # (Config-static disable_markov branch -> baked into the graph.)
            if self._disable_markov:
                logits_i = base_logits[:, i]
            else:
                markov_embed = self.model.markov_embed(prev)
                bias = self.model.markov_bias(markov_embed)
                if (
                    self._debug_markov
                    and i == 0
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    self._log_markov_magnitudes(base_logits[:, 0], bias)
                # scale (default 1.0) modulates the bias magnitude; see __init__.
                logits_i = base_logits[:, i] + self._markov_scale * bias

            if self.draft_logits is not None:
                # Probabilistic draft: sample in target vocab. A reduced draft
                # vocab is scattered into its target columns; full vocab is
                # already aligned.
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(
                        1, self._d2t_scatter_index, logits_i.to(buf.dtype)
                    )
                    logits_i = buf
                # Draft Gumbel key = sample_pos + offset. Default offset +1 is
                # the fork DFlash convention (DraftModelSpeculator.sample_draft);
                # upstream DSpark uses -1. The rejection sampler keys acceptance
                # on the TARGET position, so this offset does NOT change the
                # acceptance rate -- only which valid token is proposed. Exposed
                # via VLLM_DSPARK_GUMBEL_POS_OFFSET to confirm empirically.
                draft_sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] + self._gumbel_pos_offset,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                # Greedy draft: argmax in draft vocab, remap to target vocab
                # (identity for full-vocab drafts).
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )

            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under the DFlash CUDA graph): the parallel
        # backbone forward, then sequential Markov sampling over its per-step
        # hidden state outputs. Signature matches DFlashSpeculator._generate_
        # draft so DFlashCudaGraphManager.capture / propose drive it unchanged.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
