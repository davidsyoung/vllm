# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

SUPPORTED_SPECULATORS_TYPES = {}


def register_speculator(name):
    def decorator(fn):
        SUPPORTED_SPECULATORS_TYPES[name] = fn
        return fn

    return decorator


@register_speculator("eagle3")
def update_eagle3(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply Eagle-3 specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    Eagle-3 specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - norm_before_fc: Whether to apply RMSNorm before the fc projection
    - eagle_aux_hidden_state_layer_ids: List of layer indices from the base
        model to use as auxiliary inputs for the Eagle3 drafter. These layers
        provide intermediate hidden states that help the drafter make better
        predictions. This is the standard field used in Eagle3 checkpoints.
    """

    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    pre_trained_config["norm_before_residual"] = config_dict.get(
        "norm_before_residual", True
    )
    pre_trained_config["norm_before_fc"] = config_dict.get("norm_before_fc", False)
    pre_trained_config["fc_norm"] = config_dict.get("fc_norm", False)
    pre_trained_config["norm_output"] = config_dict.get("norm_output", False)
    eagle3_arch_map = {
        "qwen3": "Eagle3Qwen3ForCausalLM",
        "llama": "Eagle3LlamaForCausalLM",
    }
    model_type = pre_trained_config.get("model_type", "llama")
    if model_type not in eagle3_arch_map:
        raise ValueError(f"Unsupported model_type {model_type} for Eagle3 speculator")
    pre_trained_config["architectures"] = [eagle3_arch_map[model_type]]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        pre_trained_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]


@register_speculator("peagle")
def update_peagle(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply PEagle (Parallel Eagle) specific configuration transformations to
    the `dict` used to construct the Transformers PreTrainedConfig.

    PEagle specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - norm_before_fc: Whether to apply RMSNorm before the fc projection
    - mask_token_id (required): Token ID used for parallel drafting mask
        placeholders, mapped to pard_token for the proposer
    - eagle_aux_hidden_state_layer_ids: Layer indices from the target model
        whose intermediate hidden states are used as auxiliary inputs
    """
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    pre_trained_config["norm_before_residual"] = config_dict.get(
        "norm_before_residual", False
    )
    pre_trained_config["norm_before_fc"] = config_dict.get("norm_before_fc", False)
    peagle_arch_map = {
        "qwen3": "PeagleQwen3ForCausalLM",
        "llama": "PeagleLlamaForCausalLM",
    }
    model_type = pre_trained_config.get("model_type", "llama")
    if model_type not in peagle_arch_map:
        raise ValueError(f"Unsupported model_type {model_type} for PEagle speculator")
    pre_trained_config["architectures"] = [peagle_arch_map[model_type]]
    pre_trained_config["pard_token"] = config_dict["mask_token_id"]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        pre_trained_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]


@register_speculator("dflash")
def update_dflash(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply DFlash specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    DFlash specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - mask_token_id (required): Token ID used for parallel drafting mask
        placeholders
    - aux_hidden_state_layer_ids (required): DFlash target layer indices whose
        intermediate hidden states are used as context for the DFlash drafter.
        Mapped to dflash_config.target_layer_ids for the DFlash model. The
        runner-facing eagle_aux_hidden_state_layer_ids are shifted by one to
        match vLLM's hidden-state extraction semantics.
    """
    pre_trained_config["architectures"] = ["DFlashDraftModel"]
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    for key in (
        "layer_types",
        "use_sliding_window",
        "sliding_window",
        "max_window_layers",
    ):
        if key in config_dict:
            pre_trained_config[key] = config_dict[key]

    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = [
        i + 1 for i in aux_layer_ids
    ]

    # DFlash configs use different indexing for the target layers, see #40727
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
    }


@register_speculator("dspark")
def update_dspark(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply DSpark specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    DSpark is DFlash + a low-rank Markov logit-bias head (and an optional,
    inference-unused confidence head). Upstream (#47093) ships a dedicated
    DSpark speculator runtime; this fork now ports it as the standalone
    `Qwen3DSparkSpeculator`
    (`v1/worker/gpu/spec_decode/dspark_qwen3/speculator.py`), which extends the
    fork's DFlash speculator to consult the Markov head at sampling time. The
    standalone speculators-format checkpoint (e.g.
    RedHatAI/GLM-5.2-speculator.dspark) is driven by that runtime via the
    `Qwen3DSparkModel` registry entry (see qwen3_dspark.py + IMPL_NOTES.md).

    The DSpark runtime reuses the fork's DFlash block machinery for the parallel
    backbone (context-KV precompute + non-causal query block + the DFlash input
    prep kernel), so this function intentionally MIRRORS `update_dflash`'s
    config layout (a nested `dflash_config` with `mask_token_id` +
    `target_layer_ids`, and the fork's +1 `eagle_aux_hidden_state_layer_ids`
    extraction shift) rather than upstream `update_dspark`'s flat
    `target_layer_ids`/no-shift layout, which targets upstream's different
    runner extraction semantics.

    DSpark specific fields:
    - markov_rank / markov_head_type: low-rank Markov logit-bias head.
    - block_size: semi-autoregressive draft block size (num_spec_tokens + 1).
    - enable_confidence_head / confidence_head_with_markov: confidence head.
    - draft_vocab_size / target_hidden_size / mask_token_id: as DFlash.
    - aux_hidden_state_layer_ids (required): target layer indices feeding the
        drafter (mapped exactly like DFlash for this fork's runner).
    """
    # Registry -> vllm.model_executor.models.qwen3_dspark.Qwen3DSparkForCausalLM
    pre_trained_config["architectures"] = ["Qwen3DSparkModel"]
    # Speculators DSpark uses the 1+N fill-in block (anchor is a bonus token).
    # Kept for parity with upstream; harmless if unread by this fork.
    pre_trained_config["dspark_bonus_anchor"] = True

    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    for key in (
        "layer_types",
        "use_sliding_window",
        "sliding_window",
        "max_window_layers",
    ):
        if key in config_dict:
            pre_trained_config[key] = config_dict[key]

    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    # Fork DFlash runner-facing capture layers are shifted +1 (matches
    # update_dflash). WHY +1: vLLM's aux capture appends the residual stream
    # ENTERING decoder layer `idx` (deepseek_v2.py: `aux_hidden_states.append(
    # hidden_states + residual)` runs BEFORE `layer(idx)`), i.e. capture id `k`
    # yields the OUTPUT of layer `k-1`. So to capture the output of the
    # checkpoint's declared aux layers [8,23,39,55,70], the capture ids must be
    # [9,24,40,56,71] = aux_id + 1. The drafter's own target taps
    # (dflash_config.target_layer_ids, used for the feature COUNT) are aux_id-1.
    #
    # DIAGNOSTIC OVERRIDE: VLLM_DSPARK_AUX_LAYER_SHIFT (default 1) lets us A/B the
    # capture layers on GPU without rebuilding, to empirically confirm which
    # layers the Markov head/backbone was actually trained on:
    #   1  -> [9,24,40,56,71]  (default; captures outputs of [8,23,39,55,70])
    #   0  -> [8,23,39,55,70]  (captures outputs of [7,22,38,54,69])
    #  -1  -> [7,22,38,54,69]
    aux_shift = int(os.getenv("VLLM_DSPARK_AUX_LAYER_SHIFT", "1"))
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = [
        i + aux_shift for i in aux_layer_ids
    ]
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
    }

    # DSpark head + block hyperparameters. `markov_rank` is read directly by
    # Qwen3DSparkModel; the rest are passthrough attributes.
    for key in (
        "markov_rank",
        "markov_head_type",
        "block_size",
        "enable_confidence_head",
        "confidence_head_with_markov",
    ):
        if config_dict.get(key) is not None:
            pre_trained_config[key] = config_dict[key]
