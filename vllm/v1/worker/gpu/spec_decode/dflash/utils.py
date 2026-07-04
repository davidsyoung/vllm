# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share


def get_dflash_causal(draft_model_config: ModelConfig) -> bool:
    """Whether the DFlash draft uses causal (vs non-causal) attention."""
    dflash_config = getattr(draft_model_config.hf_config, "dflash_config", None) or {}
    return dflash_config.get("causal", False)


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Modify the attention config so that we select an attention backend that matches
    # the causal/non-causal mode of the dflash model.
    causal = get_dflash_causal(draft_model_config)
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=not causal,
            # Honor the speculative-config attention backend for the draft
            # (matches llm_base_proposer): otherwise auto-select picks
            # FlashInfer, which downgrades the spec-decode cudagraph to
            # PIECEWISE and cannot do non-causal prefill under DCP.
            backend=speculative_config.attention_backend,
        ),
    )
    # Honor the speculative-config draft KV cache dtype (mirrors
    # eagle/utils.py::_create_draft_vllm_config, which the V2 DFlash/DSpark load
    # path otherwise skipped). Critical when the TARGET uses an MLA/DSA fp8 KV
    # format (e.g. GlmMoeDsa's fp8_ds_mla): the DENSE DFlash/DSpark draft
    # otherwise inherits the engine-wide MLA cache_dtype via
    # current_vllm_config.cache_config, and a dense backend like TRITON_ATTN
    # rejects it ("kv_cache_dtype not supported"). Setting
    # draft_kv_cache_dtype:"auto" (or "bfloat16") gives the dense draft a
    # TRITON-compatible KV dtype independent of the target's MLA fp8. Guarded on
    # `is not None`, so default behavior (draft inherits target) is unchanged.
    if speculative_config.draft_kv_cache_dtype is not None:
        draft_vllm_config = replace(
            draft_vllm_config,
            cache_config=replace(
                draft_vllm_config.cache_config,
                cache_dtype=speculative_config.draft_kv_cache_dtype,
            ),
        )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = dflash_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    # Share lm_head with the target when the draft has no own copy. DFlash
    # may expose draft_id_to_target_id for sampled-token remapping, but its
    # logits still need to be scored by the target-vocab head for acceptance.
    target_lm_head = getattr(target_language_model, "lm_head", None)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
