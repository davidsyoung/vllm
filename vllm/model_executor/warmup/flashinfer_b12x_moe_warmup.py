# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the FlashInfer b12x NVFP4 fused-MoE kernel before serving.

The b12x fused MoE (SM120/SM121, RTX Pro 6000 / DGX Spark) is a CuteDSL kernel
that JIT-compiles a specialization per input-shape bucket. Its small-token
("micro") path keys the compiled kernel on both ``num_tokens`` and the internal
workspace's ``max_rows``. A large prefill grows that shared workspace, which
invalidates the previously-compiled small-batch kernels; the next small request
then pays a multi-second, engine-blocking MLIR compile mid-serving -- a
stop-the-world stall that freezes every in-flight request (see issue #47458).

We defuse this at startup: run one dummy forward at the maximum token count so
the shared workspace reaches its serving-time high-water mark, then dummy-run
the small token sizes so the micro-path kernels compile against that final
workspace and stay cache-valid for the whole session. Ordering matters -- the
large shape must run first, otherwise the small-shape kernels compiled here get
invalidated by the first real large prefill.

This is a no-op unless the model actually uses the b12x MoE backend (opt-in via
``--moe-backend=flashinfer_b12x`` on SM120), so it costs nothing elsewhere.
"""

import time
from typing import TYPE_CHECKING

import torch

from vllm.distributed.parallel_state import is_global_first_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.flashinfer import has_flashinfer_b12x_moe

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# Small token counts that route through the b12x micro-path. Each distinct
# ``num_tokens`` in this region compiles its own kernel specialization, so we
# warm a representative set (powers of two up to 256); the cudagraph capture
# sizes are folded in on top of these.
_MICRO_TOKEN_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _select_warmup_token_sizes(
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    """Token sizes to warm, sorted **descending** so the largest (which grows
    the shared workspace to its serving-time maximum) runs first."""
    if max_tokens <= 0:
        return []
    sizes = {s for s in _MICRO_TOKEN_CANDIDATES if 1 <= s <= max_tokens}
    sizes.update(s for s in cudagraph_capture_sizes if 1 <= s <= max_tokens)
    sizes.add(max_tokens)
    return sorted(sizes, reverse=True)


def _model_uses_b12x_moe(model: torch.nn.Module) -> bool:
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        if moe_kernel is None:
            continue
        if isinstance(
            getattr(moe_kernel, "fused_experts", None), FlashInferB12xExperts
        ):
            return True
    return False


@instrument(span_name="FlashInfer b12x MoE warmup")
def flashinfer_b12x_moe_warmup(worker: "Worker") -> None:
    # Cheap gates first: the backend is SM120-only and opt-in, so this returns
    # immediately for the vast majority of deployments.
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and has_flashinfer_b12x_moe()
    ):
        return

    model = worker.get_model()
    if not _model_uses_b12x_moe(model):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    cudagraph_capture_sizes = (
        worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
    )
    token_sizes = _select_warmup_token_sizes(max_tokens, cudagraph_capture_sizes)
    if not token_sizes:
        return

    if is_global_first_rank():
        logger.info(
            "Warming up FlashInfer b12x MoE kernels for token sizes: %s",
            token_sizes,
        )

    started = time.perf_counter()
    # Each rank compiles its b12x kernels independently (separate JIT cache), so
    # every rank must run the warmup -- under pipeline parallelism the stall the
    # issue reports is the sum of per-rank compiles.
    for num_tokens in token_sizes:
        worker.model_runner._dummy_run(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
        )
    torch.accelerator.synchronize()

    if is_global_first_rank():
        logger.info(
            "FlashInfer b12x MoE warmup finished in %.2f seconds.",
            time.perf_counter() - started,
        )
