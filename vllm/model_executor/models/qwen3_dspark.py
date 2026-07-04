# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DSpark draft model for semi-autoregressive drafting (#47093).

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head.

Graft note (this fork):
-----------------------
This file is grafted from upstream vLLM's ``qwen3_dspark.py`` (PR #47093 @
2b753ad) but the backbone is inherited from THIS fork's DFlash Qwen3 draft
(``qwen3_dflash.py``), whose class contract differs slightly from upstream's
(``self.model`` is built under the ``dflash_model`` prefix; the runner-facing
aux-hidden-state layer ids are +1 shifted by ``update_dflash``/``update_dspark``).
Everything the fork's ``DFlashQwen3ForCausalLM`` provides
(``forward`` / ``compute_logits`` / ``precompute_and_store_context_kv`` /
``combine_hidden_states`` / ``embed_input_ids`` / ``sliding_attention_layer_names``)
is inherited unchanged so the whole DFlash proposer/speculator path is reused
verbatim.

The Markov head (and the ``compute_draft_logits`` / ``markov_embed`` /
``markov_bias`` / ``map_draft_to_target`` helpers) ARE consulted at sampling
time by the fork's standalone DSpark runtime
(``vllm/v1/worker/gpu/spec_decode/dspark_qwen3/speculator.py`` ::
``Qwen3DSparkSpeculator``), which extends the fork's DFlash speculator to add
the sequential Markov bias at each draft step. This model file provides the
head + helpers; the speculator provides the sequential loop. See IMPL_NOTES.md.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    ``markov_w1[token]`` embeds the previously sampled token (target vocab,
    ``vocab_size``); ``markov_w2`` projects it to a draft-vocab bias
    (``draft_vocab_size``) added to the base draft logits. The two sizes
    coincide for full-vocab drafts (the GLM-5.2 speculator is full-vocab).
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
        )
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w2")
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of ``token_ids`` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        return logits_processor(self.markov_w2, markov_embed)


class Qwen3DSparkModel(DFlashQwen3Model):
    """Fork DFlash Qwen3 backbone + DSpark Markov head."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        markov_rank = int(getattr(config, "markov_rank", 0) or 0)
        # markov_rank == 0 => pure-DFlash DSpark checkpoint (no sequential head).
        self.markov_head: DSparkMarkovHead | None = (
            DSparkMarkovHead(
                config.vocab_size,
                draft_vocab_size,
                markov_rank,
                prefix=maybe_prefix(prefix, "markov_head"),
            )
            if markov_rank > 0
            else None
        )


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    """Standalone speculators-format DSpark draft.

    Mirrors THIS fork's ``DFlashQwen3ForCausalLM.__init__`` exactly (same
    ``dflash_model`` submodule prefix, lm_head, logits_processor and d2t
    remap) so the fork's DFlash proposer/speculator drives it unchanged; the
    only structural addition is the Markov head inside ``self.model``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        # NOTE: "dflash_model" (not upstream's "model") — keeps the draft
        # attention layer names in the fork's DFlash namespace so the fork's
        # DFlash KV/attention wiring matches. Checkpoint parameter names are
        # unchanged (they follow the Python attr hierarchy self.model.*).
        self.model = Qwen3DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "dflash_model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    # ---- DSpark sampling helpers (consumed by Qwen3DSparkSpeculator) ----
    # The standalone DSpark runtime (dspark_qwen3/speculator.py) calls these
    # per draft step: compute_draft_logits (draft-vocab base logits, no d2t
    # scatter), markov_embed + markov_bias (the low-rank transition bias), and
    # map_draft_to_target (draft->target id remap, identity for full vocab).
    # The inherited `compute_logits` (with d2t scatter) is NOT used here.

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Draft-vocab logits without the d2t scatter.
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert self.model.markov_head is not None
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        assert self.model.markov_head is not None
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    # ---------------------------- weight loading ---------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        for name, loaded_weight in weights:
            # t2d is training-only; the draft remaps via d2t at sampling time.
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            model_weights[name] = loaded_weight
            # Sets has_own_embed_tokens / has_own_lm_head so load_dflash_model
            # knows whether to keep these or alias the target's.
            process_eagle_weight(self, name)

        # mask_embedding is an unused placeholder; DSpark masks via the vocab row.
        # confidence_head is not wired into inference; skip its weights.
        # embed_tokens / lm_head are optional; when omitted they are shared from
        # the target by load_dflash_model, so skip the unloaded params here.
        skip_substrs = ["mask_embedding", "confidence_head"]
        if self.model.markov_head is None:
            skip_substrs.append("markov")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
