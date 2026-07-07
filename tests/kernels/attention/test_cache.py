# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import scaled_dequantize
from vllm.platforms import current_platform
from vllm.utils.torch_utils import nvfp4_kv_cache_split_views, set_random_seed

COPYING_DIRECTION = [("cuda", "cpu"), ("cuda", "cuda"), ("cpu", "cuda")]
DTYPES = [torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 256]
BLOCK_SIZES = [8, 16, 32]
CACHE_LAYOUTS = ["NHD", "HND"]
KV_SCALE_TYPES = ["tensor", "attn_head"]

# Parameters for MLA tests.
KV_LORA_RANKS = [256, 512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024, 10000]

NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]

# We assume fp8 is always enabled for testing.
KV_CACHE_DTYPE = ["auto", "fp8"]

RESHAPE_FLASH_IMPLEMENTATIONS = ["cuda", "triton"]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_reshape_and_cache(
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()
    set_random_seed(seed)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    # Clone the KV caches.
    if kv_cache_dtype == "fp8":
        cloned_key_cache = torch.empty_like(key_cache, dtype=torch.float16)
        ops.convert_fp8(cloned_key_cache, key_cache, k_scale.item())
        cloned_value_cache = torch.empty_like(value_cache, dtype=torch.float16)
        ops.convert_fp8(cloned_value_cache, value_cache, v_scale.item())
    else:
        cloned_key_cache = key_cache.clone()
        cloned_value_cache = value_cache.clone()

    # Call the reshape_and_cache kernel.
    opcheck(
        torch.ops._C_cache_ops.reshape_and_cache,
        (
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
        cond=(head_size == HEAD_SIZES[0]),
    )
    ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    if kv_cache_dtype == "fp8":
        result_key_cache = torch.empty_like(key_cache, dtype=torch.float16)
        ops.convert_fp8(result_key_cache, key_cache, k_scale.item())
        result_value_cache = torch.empty_like(value_cache, dtype=torch.float16)
        ops.convert_fp8(result_value_cache, value_cache, v_scale.item())

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indices_lst = block_indices.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indices_lst[i]
        block_offset = block_offsets_lst[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    if kv_cache_dtype == "fp8":
        torch.testing.assert_close(
            result_key_cache, cloned_key_cache, atol=0.001, rtol=0.1
        )
        torch.testing.assert_close(
            result_value_cache, cloned_value_cache, atol=0.001, rtol=0.1
        )
    else:
        torch.testing.assert_close(key_cache, cloned_key_cache)
        torch.testing.assert_close(value_cache, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE + ["nvfp4"])
@pytest.mark.parametrize("kv_cache_layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("kv_scale_type", KV_SCALE_TYPES)
@pytest.mark.parametrize("implementation", RESHAPE_FLASH_IMPLEMENTATIONS)
@torch.inference_mode()
def test_reshape_and_cache_flash(
    kv_cache_factory_flashinfer,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    kv_scale_type: str,
    implementation: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    assert implementation in ["cuda", "triton"]
    if implementation == "triton" and kv_cache_layout == "HND":
        pytest.skip("Triton implementation only supports NHD layout.")

    if kv_scale_type == "attn_head" and implementation != "cuda":
        pytest.skip("Only CUDA implementation supports attn_head scaling.")

    if kv_cache_dtype == "nvfp4":
        if not current_platform.has_device_capability(100):
            pytest.skip("NVFP4 requires compute capability >= 10.0 (Blackwell).")
        if implementation != "cuda":
            pytest.skip("NVFP4 only supports CUDA implementation.")
        if kv_scale_type != "tensor":
            pytest.skip("NVFP4 only supports per-tensor scaling.")
        if head_size % 16 != 0:
            pytest.skip("NVFP4 requires head_size divisible by 16.")
        if (head_size // 16) % 4 != 0:
            pytest.skip(
                "NVFP4 requires (head_size // 16) divisible by 4 "
                "for 4x4 block scale swizzle."
            )
        if block_size % 4 != 0:
            pytest.skip("NVFP4 requires block_size divisible by 4.")
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("NVFP4 quantization only supports fp16/bf16 input.")

    # fp8 conversion requires continugous memory buffer. Reduce the number of
    # blocks and tokens to consume less memory.
    num_tokens = num_tokens // 2
    num_blocks = num_blocks // 2
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)
    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype, device=device)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory_flashinfer(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        device=device,
        cache_layout=kv_cache_layout,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    del key_caches
    del value_caches

    # For nvfp4, the factory returns kv[:, 0] and kv[:, 1] like all dtypes.
    # Split views are still needed for dequant verification.
    key_scale_cache = None
    value_scale_cache = None
    nvfp4_key_data = None
    nvfp4_value_data = None
    if kv_cache_dtype == "nvfp4":
        (nvfp4_key_data,), (key_scale_cache,) = nvfp4_kv_cache_split_views(key_cache)
        (nvfp4_value_data,), (value_scale_cache,) = nvfp4_kv_cache_split_views(
            value_cache
        )

    if kv_cache_dtype == "nvfp4":
        # Global scale = amax / 448 (per-tensor)
        k_scale = (key.abs().amax() / 448.0).to(torch.float32)
        v_scale = (value.abs().amax() / 448.0).to(torch.float32)
    elif kv_scale_type == "tensor":
        k_scale = (key.amax() / 64.0).to(torch.float32)
        v_scale = (value.amax() / 64.0).to(torch.float32)
    else:  # "attn_head"
        k_scale = (key.amax(dim=(0, 2)) / 64.0).to(torch.float32)
        v_scale = (value.amax(dim=(0, 2)) / 64.0).to(torch.float32)

    def permute_and_compact(x):
        y = x if kv_cache_layout == "NHD" else x.permute(0, 2, 1, 3)
        return y.contiguous()

    if kv_cache_dtype != "nvfp4":
        key_cache_compact = permute_and_compact(key_cache)
        value_cache_compact = permute_and_compact(value_cache)

    def convert_fp8_local(output, input, scale, kv_dtype):
        fp8_input = input.view(current_platform.fp8_dtype())
        if scale.numel() == 1:  # per-tensor
            result = scaled_dequantize(
                fp8_input.flatten(0, 2), scale, group_shape=None, out_dtype=output.dtype
            ).reshape(*input.shape)
        else:  # per-head: broadcast scale along the head dimension
            # Original code uses dim 2 for NHD, dim 1 for HND
            if kv_cache_layout == "NHD":
                result = fp8_input.to(output.dtype) * scale.view(1, 1, -1, 1)
            else:
                result = fp8_input.to(output.dtype) * scale.view(1, -1, 1, 1)
        output.copy_(result)

    # Clone the KV caches (for non-nvfp4, used as reference baseline).
    if kv_cache_dtype == "fp8":
        cloned_key_cache = torch.empty_like(key_cache_compact, dtype=torch.float16)
        convert_fp8_local(cloned_key_cache, key_cache_compact, k_scale, kv_cache_dtype)
        cloned_value_cache = torch.empty_like(value_cache_compact, dtype=torch.float16)
        convert_fp8_local(
            cloned_value_cache, value_cache_compact, v_scale, kv_cache_dtype
        )
    elif kv_cache_dtype != "nvfp4":
        cloned_key_cache = key_cache_compact.clone()
        cloned_value_cache = value_cache_compact.clone()

    # Call the reshape_and_cache kernel.
    if implementation == "cuda":
        if kv_cache_dtype != "nvfp4":
            opcheck(
                torch.ops._C_cache_ops.reshape_and_cache_flash,
                (
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                ),
                cond=(head_size == HEAD_SIZES[0]),
            )
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    elif implementation == "triton":
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )

        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    if kv_cache_dtype == "nvfp4":
        # Verify NVFP4 by dequantizing the entire cache and comparing
        # the written positions against original bf16 values.
        # Same pattern as FP8: dequant whole cache, then extract and compare.
        from tests.kernels.quantization.nvfp4_utils import (
            dequant_nvfp4_kv_cache,
        )

        def dequant_nvfp4_cache_nhd(data_cache, scale_cache, global_scale):
            # data_cache:  [N, T, H, data_dim]  NHD (contiguous inner dims)
            # scale_cache: [N, T, H, scale_dim] NHD (contiguous inner dims)
            # Permute to HND layout for the dequant utility.
            data_hnd = data_cache.permute(0, 2, 1, 3)
            scale_hnd = scale_cache.permute(0, 2, 1, 3)
            result_hnd = dequant_nvfp4_kv_cache(
                data_hnd, scale_hnd, global_scale, head_size, block_size
            )
            return result_hnd.permute(0, 2, 1, 3)  # back to [N, T, H, D]

        result_key_cache = dequant_nvfp4_cache_nhd(
            nvfp4_key_data, key_scale_cache, k_scale.item()
        )
        result_value_cache = dequant_nvfp4_cache_nhd(
            nvfp4_value_data, value_scale_cache, v_scale.item()
        )

        # Flatten [num_blocks, block_size] → [num_slots] and index by slot_mapping.
        num_slots = num_blocks * block_size
        result_key_flat = result_key_cache.reshape(num_slots, num_heads, head_size)
        result_value_flat = result_value_cache.reshape(num_slots, num_heads, head_size)

        torch.testing.assert_close(
            result_key_flat[slot_mapping], key.float(), atol=1.5, rtol=0.5
        )
        torch.testing.assert_close(
            result_value_flat[slot_mapping], value.float(), atol=1.5, rtol=0.5
        )
        return

    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    if kv_cache_dtype == "fp8":
        result_key_cache = torch.empty_like(key_cache_compact, dtype=torch.float16)
        convert_fp8_local(result_key_cache, key_cache_compact, k_scale, kv_cache_dtype)
        result_value_cache = torch.empty_like(value_cache_compact, dtype=torch.float16)
        convert_fp8_local(
            result_value_cache,
            value_cache_compact,
            v_scale,
            kv_cache_dtype,
        )

    # Run the reference implementation.
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indices_lst = block_indices.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indices_lst[i]
        block_offset = block_offsets_lst[i]
        if kv_cache_layout == "NHD":
            cloned_key_cache[block_idx, block_offset, :, :] = key[i]
            cloned_value_cache[block_idx, block_offset, :, :] = value[i]
        else:
            cloned_key_cache[block_idx, :, block_offset, :] = key[i]
            cloned_value_cache[block_idx, :, block_offset, :] = value[i]

    if kv_cache_dtype == "fp8":
        torch.testing.assert_close(
            result_key_cache, cloned_key_cache, atol=0.001, rtol=0.1
        )
        torch.testing.assert_close(
            result_value_cache, cloned_value_cache, atol=0.001, rtol=0.1
        )
    else:
        torch.testing.assert_close(key_cache_compact, cloned_key_cache)
        torch.testing.assert_close(value_cache_compact, cloned_value_cache)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("kv_cache_layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("implementation", RESHAPE_FLASH_IMPLEMENTATIONS)
@torch.inference_mode()
def test_reshape_and_cache_flash_unaligned_rows(
    kv_cache_factory_flashinfer,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    implementation: str,
) -> None:
    """Regression test for https://github.com/vllm-project/vllm/issues/41257.

    head_size=46 with num_heads=13 places KV-cache rows at byte offsets
    that are not a multiple of the vector width (NHD row pitch
    13*46*itemsize, HND head pitch 46*itemsize), unlike HEAD_SIZES above
    which are all 16-byte multiples. The CUDA kernel used to issue
    vectorized stores to those rows -> CUDA misaligned address.
    """
    test_reshape_and_cache_flash(
        kv_cache_factory_flashinfer,
        num_tokens=42,
        num_heads=13,
        head_size=46,
        block_size=16,
        num_blocks=128,
        dtype=dtype,
        seed=0,
        device=CUDA_DEVICES[0],
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_layout=kv_cache_layout,
        kv_scale_type="tensor",
        implementation=implementation,
    )


@pytest.mark.parametrize("direction", COPYING_DIRECTION)
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_swap_blocks(
    kv_cache_factory,
    direction: tuple[str, str],
    num_mappings: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    if kv_cache_dtype == "fp8" and "cpu" in direction:
        pytest.skip()
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()

    set_random_seed(seed)

    src_device = device if direction[0] == "cuda" else "cpu"
    dst_device = device if direction[1] == "cuda" else "cpu"

    src_blocks = random.sample(range(num_blocks), num_mappings)
    # For the same device, mapping must not overlap
    if src_device == dst_device:
        remaining_blocks = list(set(range(num_blocks)) - set(src_blocks))
        dst_blocks = random.sample(remaining_blocks, num_mappings)
    else:
        dst_blocks = random.sample(range(num_blocks), num_mappings)

    block_mapping = list(zip(src_blocks, dst_blocks))
    block_mapping_tensor = torch.tensor(
        block_mapping, dtype=torch.int64, device="cpu"
    ).view(-1, 2)

    # Create the KV caches on the first device.
    src_key_caches, src_value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        src_device,
    )

    # Create the KV caches on the second device.
    dist_key_caches, dist_value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        dst_device,
    )

    src_key_caches_clone = src_key_caches[0].clone()
    src_value_caches_clone = src_value_caches[0].clone()

    # Call the swap_blocks kernel.
    do_opcheck = head_size == HEAD_SIZES[0]
    src_cache = src_key_caches[0]
    block_size_in_bytes = src_cache.element_size() * src_cache.stride(0)
    opcheck(
        torch.ops._C_cache_ops.swap_blocks,
        (
            src_key_caches[0],
            dist_key_caches[0],
            block_size_in_bytes,
            block_mapping_tensor,
        ),
        cond=do_opcheck,
    )
    opcheck(
        torch.ops._C_cache_ops.swap_blocks,
        (
            src_value_caches[0],
            dist_value_caches[0],
            block_size_in_bytes,
            block_mapping_tensor,
        ),
        cond=do_opcheck,
    )

    ops.swap_blocks(
        src_key_caches[0],
        dist_key_caches[0],
        block_size_in_bytes,
        block_mapping_tensor,
    )
    ops.swap_blocks(
        src_value_caches[0],
        dist_value_caches[0],
        block_size_in_bytes,
        block_mapping_tensor,
    )

    for src, dst in block_mapping:
        torch.testing.assert_close(
            src_key_caches_clone[src].cpu(), dist_key_caches[0][dst].cpu()
        )
        torch.testing.assert_close(
            src_value_caches_clone[src].cpu(), dist_value_caches[0][dst].cpu()
        )


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fp8_e4m3_conversion(
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)

    low = -224.0
    high = 224.0
    shape = (num_blocks, num_heads, head_size, block_size)
    cache = torch.empty(shape, dtype=dtype, device=device)
    cache.uniform_(low, high)

    cache_fp8 = torch.empty_like(cache, dtype=torch.uint8)
    ops.convert_fp8(cache_fp8, cache)

    converted_cache = torch.empty_like(cache)
    ops.convert_fp8(converted_cache, cache_fp8)

    torch.testing.assert_close(cache, converted_cache, atol=0.001, rtol=0.1)


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(
        num_blocks, block_size, entry_size, dtype=cache_dtype, device=device
    )


def _fill_mla_cache(cache: torch.Tensor, kv_cache_dtype: str):
    rand_dtype = torch.float16 if kv_cache_dtype == "fp8" else cache.dtype

    vals = torch.randn(*cache.shape, device=cache.device, dtype=rand_dtype)
    if kv_cache_dtype == "fp8":
        temp = torch.zeros_like(cache)
        ops.convert_fp8(temp, vals, 1.0, kv_dtype=kv_cache_dtype)
        vals = temp
    cache.copy_(vals)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(
        num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
    )
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = torch.empty_like(ref_temp, dtype=kv_cache.dtype)
        ops.convert_fp8(ref_kv_cache, ref_temp, scale.item(), kv_dtype=kv_cache_dtype)
    else:
        ref_kv_cache = ref_temp

    opcheck(
        torch.ops._C_cache_ops.concat_and_cache_mla,
        (kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)

    if kv_cache_dtype == "fp8":
        result_temp = torch.empty_like(kv_cache, dtype=torch.float16)
        ops.convert_fp8(
            result_temp, kv_cache.contiguous(), scale.item(), kv_dtype=kv_cache_dtype
        )
        expected_temp = torch.empty_like(ref_kv_cache, dtype=torch.float16)
        ops.convert_fp8(
            expected_temp, ref_kv_cache, scale.item(), kv_dtype=kv_cache_dtype
        )
        torch.testing.assert_close(result_temp, expected_temp, atol=0.001, rtol=0.1)
    else:
        torch.testing.assert_close(kv_cache, ref_kv_cache)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_concat_and_cache_ds_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    if current_platform.is_rocm():
        pytest.skip("concat_and_cache_mla doesn't support fp8_ds_mla on ROCm")
    if dtype.itemsize != 2:
        pytest.skip("ds_mla only supports 16-bit input")
    if kv_lora_rank != 512:
        pytest.skip("fp8_ds_mla requires kv_lora_rank == 512")
    kv_cache_dtype = "fp8_ds_mla"
    set_random_seed(seed)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
    entry_size = kv_lora_rank + (4 * 4) + (2 * qk_rope_head_dim)

    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(
        num_blocks,
        block_size,
        entry_size,
        dtype=torch.uint8,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
    )

    ref_cache = torch.zeros_like(kv_cache, dtype=kv_cache.dtype)
    tile_data = torch.zeros(128, dtype=dtype, device=device)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size

        ref_cache_slice = ref_cache[block_idx, block_offset]
        ref_cache_16bit = ref_cache_slice.view(dtype)
        ref_cache_32bit = ref_cache_slice.view(torch.float32)

        kv_c_data = kv_c[i]
        num_tiles = kv_lora_rank // 128
        for tile_idx in range(num_tiles):
            tile_start = tile_idx * 128
            tile_end = (tile_idx + 1) * 128
            tile_data[:] = kv_c_data[tile_start:tile_end]

            # tile_scale = tile_data.amax().to(torch.float32) / 448.
            # NOTE: Using torch's amax() gives different results,
            # so this must be manually computed.
            tile_data_float = tile_data.to(torch.float32)
            manual_max = abs(tile_data_float[0])
            for j in range(1, 128):
                manual_max = max(manual_max, abs(tile_data_float[j]))
            tile_scale = manual_max / 448.0

            ref_cache_32bit[kv_lora_rank // 4 + tile_idx] = tile_scale

            ops.convert_fp8(
                ref_cache_slice[tile_start:tile_end],
                tile_data,
                tile_scale.item(),
                kv_dtype="fp8",
            )

        for j in range(qk_rope_head_dim):
            ref_cache_16bit[kv_lora_rank // 2 + 8 + j] = k_pe[i, j]

    opcheck(
        torch.ops._C_cache_ops.concat_and_cache_mla,
        (kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        kv_cache_slice = kv_cache[block_idx, block_offset]
        ref_cache_slice = ref_cache[block_idx, block_offset]

        kv_nope = kv_cache_slice[:kv_lora_rank]
        ref_nope = ref_cache_slice[:kv_lora_rank]
        kv_scales = kv_cache_slice.view(torch.float32)[
            kv_lora_rank // 4 : kv_lora_rank // 4 + 4
        ]
        ref_scales = ref_cache_slice.view(torch.float32)[
            kv_lora_rank // 4 : kv_lora_rank // 4 + 4
        ]
        kv_rope = kv_cache_slice.view(dtype)[kv_lora_rank // 2 + 8 :]
        ref_rope = ref_cache_slice.view(dtype)[kv_lora_rank // 2 + 8 :]

        torch.testing.assert_close(kv_nope, ref_nope, atol=0.001, rtol=0.1)
        torch.testing.assert_close(kv_scales, ref_scales, atol=0.001, rtol=0.1)
        torch.testing.assert_close(kv_rope, ref_rope, atol=0.001, rtol=0.1)


# ── NF3 (3-bit NormalFloat) MLA KV records ──────────────────────────────────
# Pure-torch reference encoder/decoder pair for the nf3_ds_mla (304 B) and
# nf3bf16_ds_mla (368 B) records. The codebook is madeby561's nf3_kernel.py
# table verbatim; encode = branchless nearest-of-8 by the 7 f32 midpoints
# (>= picks the upper level on ties, matching the kernel's setp.ge chain);
# scale = E4M3(group amax) since max|NF3 level| == 1.0.
NF3_LEVELS = [-1.0, -0.6047, -0.3563, -0.1275, 0.1275, 0.3563, 0.6047, 1.0]
NF3_MIDPOINTS = [
    (NF3_LEVELS[i] + NF3_LEVELS[i + 1]) / 2 for i in range(7)
]
NF3_MAX_HALF_GAP = max(
    NF3_LEVELS[i + 1] - NF3_LEVELS[i] for i in range(7)
) / 2  # 0.19765 (the [-1.0, -0.6047] gap)
NF3_GROUP_SIZE = 16
E4M3_MAX_RCP = 1.0 / 448.0  # exact f32 constant, mirrors the write kernel


def ref_nf3_encode(latent: torch.Tensor):
    """Encode (num_groups, 16) f32 -> (codes u8 in 0..7, e4m3 scale bytes).

    Mirrors the kernel recipe: scale byte = satfinite-E4M3(group amax);
    values normalized by the hardware-exact DECODE of that byte; nearest
    NF3 level via the 7 midpoint thresholds (ties -> upper level)."""
    amax = latent.abs().amax(dim=-1)
    scale_e4m3 = amax.to(torch.float8_e4m3fn)
    decoded_scale = scale_e4m3.float()
    safe = torch.where(decoded_scale > 0, decoded_scale, torch.ones_like(decoded_scale))
    x = latent / safe[:, None]
    thresholds = torch.tensor(
        NF3_MIDPOINTS, dtype=torch.float32, device=latent.device
    )
    codes = (x.unsqueeze(-1) >= thresholds).sum(dim=-1).to(torch.uint8)
    codes = torch.where(
        (decoded_scale > 0)[:, None], codes, torch.zeros_like(codes)
    )
    return codes, scale_e4m3


def ref_nf3_pack(codes: torch.Tensor) -> torch.Tensor:
    """Pack (num_groups, 16) codes -> (num_groups, 6) bytes: one 48-bit
    little-endian word per group, code j at bits [3j, 3j+3)."""
    num_groups = codes.shape[0]
    words = torch.zeros(num_groups, dtype=torch.int64, device=codes.device)
    for j in range(NF3_GROUP_SIZE):
        words |= codes[:, j].to(torch.int64) << (3 * j)
    out = torch.empty(num_groups, 6, dtype=torch.uint8, device=codes.device)
    for b in range(6):
        out[:, b] = ((words >> (8 * b)) & 0xFF).to(torch.uint8)
    return out


def ref_nf3_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack (num_groups, 6) bytes -> (num_groups, 16) codes."""
    num_groups = packed.shape[0]
    words = torch.zeros(num_groups, dtype=torch.int64, device=packed.device)
    for b in range(6):
        words |= packed[:, b].to(torch.int64) << (8 * b)
    codes = torch.empty(
        num_groups, NF3_GROUP_SIZE, dtype=torch.uint8, device=packed.device
    )
    for j in range(NF3_GROUP_SIZE):
        codes[:, j] = ((words >> (3 * j)) & 0x7).to(torch.uint8)
    return codes


def ref_nf3_decode(codes: torch.Tensor, scale_e4m3: torch.Tensor) -> torch.Tensor:
    """Decode codes x stored E4M3 group scales -> (num_groups, 16) f32."""
    lut = torch.tensor(NF3_LEVELS, dtype=torch.float32, device=codes.device)
    return lut[codes.long()] * scale_e4m3.float()[:, None]


@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_nf3_reference_roundtrip(seed: int) -> None:
    """CPU-only self-test of the torch NF3 reference encoder/decoder pair:
    encode random latents, decode via LUT x scales, and bound the error by
    the NF3 grid half-gap; E4M3 rope lane round-trip within its half-step."""
    set_random_seed(seed)
    latent = torch.randn(32, NF3_GROUP_SIZE, dtype=torch.float32) * 3.0
    latent[0] = 0.0  # all-zero group -> zero scale -> exact zeros
    codes, scale = ref_nf3_encode(latent)
    packed = ref_nf3_pack(codes)
    codes2 = ref_nf3_unpack(packed)
    assert torch.equal(codes, codes2), "48-bit LE pack/unpack must round-trip"
    dequant = ref_nf3_decode(codes2, scale)
    decoded_scale = scale.float()
    err = (dequant - latent).abs()
    bound = NF3_MAX_HALF_GAP * decoded_scale[:, None] + 2**-9
    assert (err <= bound).all(), (
        f"NF3 reference round-trip error {err.max().item():.5f} exceeds the "
        f"half-gap bound"
    )
    assert (dequant[0] == 0).all()

    # E4M3 rope lane reference: scale = amax * f32(1/448), val -> e4m3 ->
    # x scale; error bounded by the e4m3 half-step (2^-4 relative) plus the
    # denormal quantum.
    rope = torch.randn(64, dtype=torch.float32)
    rope_scale = rope.abs().max() * torch.tensor(E4M3_MAX_RCP, dtype=torch.float32)
    rope_q = (rope / rope_scale).to(torch.float8_e4m3fn)
    rope_dq = rope_q.float() * rope_scale
    rope_err = (rope_dq - rope).abs()
    rope_bound = rope.abs() * 2**-4 + rope_scale * 2**-9
    assert (rope_err <= rope_bound).all(), (
        f"E4M3 rope reference round-trip error {rope_err.max().item():.6f} "
        f"exceeds the half-step bound"
    )


def _run_concat_and_cache_nf3_family(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"{kv_cache_dtype} requires CUDA")
    if current_platform.is_rocm():
        pytest.skip(f"{kv_cache_dtype} is not supported on ROCm")
    if not current_platform.has_device_capability(100):
        pytest.skip(f"{kv_cache_dtype} requires SM100+ (Blackwell)")
    if dtype.itemsize != 2:
        pytest.skip(f"{kv_cache_dtype} only supports 16-bit input")
    if kv_lora_rank != 512:
        pytest.skip(f"{kv_cache_dtype} requires kv_lora_rank == 512")
    # The write kernels live in the b12x package (CuTe-DSL custom ops).
    pytest.importorskip("b12x.attention.mla.kv_cache")

    rope_e4m3 = kv_cache_dtype == "nf3_ds_mla"
    set_random_seed(seed)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    group_size = NF3_GROUP_SIZE
    num_groups = kv_lora_rank // group_size  # 32
    nope_bytes = num_groups * 6  # 192
    scale_offset = nope_bytes  # 192
    if rope_e4m3:
        # 304 B: NF3 + scales + e4m3 rope @224 + fp32 rope scale @288 + pad.
        rope_offset = scale_offset + num_groups  # 224
        rope_scale_offset = rope_offset + qk_rope_head_dim  # 288
        pad_lo, pad_hi = rope_scale_offset + 4, 304  # [292, 304)
        entry_size = 304
    else:
        # 368 B diagnostic: NF3 + scales + 16B pad + verbatim bf16 rope @240.
        pad_lo, pad_hi = scale_offset + num_groups, scale_offset + num_groups + 16
        rope_offset = pad_hi  # 240
        entry_size = 368
    assert entry_size % 16 == 0

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)

    # Implicit global scale of 1.0 (group scales + the per-token rope scale
    # carry all magnitude); the arg keeps the cache-op family signature.
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=torch.uint8, device=device
    )

    op = getattr(
        torch.ops.b12x,
        "concat_and_cache_nf3_mla" if rope_e4m3 else "concat_and_cache_nf3bf16_mla",
    )
    opcheck(
        op,
        (kv_c, k_pe, kv_cache, slot_mapping),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    # Route through the public entry point: concat_and_cache_mla dispatches
    # to the b12x NF3 ops on the nf3 kv_cache_dtype strings.
    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)

    for i in range(num_tokens):
        slot = slot_mapping_lst[i]
        block_idx = slot // block_size
        block_offset = slot % block_size
        record = kv_cache[block_idx, block_offset]

        # Group scales: E4M3(group_amax) -- max|NF3 level| == 1.0 so the
        # divide degenerates. Round-to-nearest E4M3 stays within half a
        # mantissa step (<= 6.25% relative).
        kv_scales = (
            record[scale_offset : scale_offset + num_groups]
            .view(torch.float8_e4m3fn)
            .float()
        )
        latent = kv_c[i].float().reshape(num_groups, group_size)
        group_amax = latent.abs().amax(dim=-1)
        torch.testing.assert_close(kv_scales, group_amax, atol=2**-9, rtol=0.07)

        # NoPE payload: reference-decode the packed NF3 codes x stored group
        # scales; the element error is bounded by the NF3 grid's largest
        # half-gap (0.19765) x the stored scale (the kernel's rcp.approx
        # inverse can flip exact-midpoint ties, which stays within the same
        # bound by construction).
        codes = ref_nf3_unpack(record[:nope_bytes].reshape(num_groups, 6))
        dequant = ref_nf3_decode(codes, record[scale_offset : scale_offset + num_groups].view(torch.float8_e4m3fn))
        err = (dequant - latent).abs()
        bound = NF3_MAX_HALF_GAP * kv_scales[:, None] + 2**-9
        assert (err <= bound).all(), (
            f"nf3 dequant error {err.max().item():.4f} exceeds the NF3 "
            f"half-gap bound at token {i}"
        )

        # The alignment pad is zero-filled.
        assert (record[pad_lo:pad_hi] == 0).all()

        if rope_e4m3:
            # Stored fp32 rope scale == amax * f32(1/448) (exact constant
            # multiply in the kernel; both sides compute in f32).
            stored_scale = record[
                rope_scale_offset : rope_scale_offset + 4
            ].view(torch.float32)[0]
            rope_f32 = k_pe[i].float()
            ref_scale = rope_f32.abs().max() * torch.tensor(
                E4M3_MAX_RCP, dtype=torch.float32, device=device
            )
            torch.testing.assert_close(
                stored_scale, ref_scale, atol=2**-20, rtol=2**-18
            )
            # E4M3 rope round-trip <= half-step (+ rcp.approx slack).
            rope_dq = (
                record[rope_offset : rope_offset + qk_rope_head_dim]
                .view(torch.float8_e4m3fn)
                .float()
                * stored_scale
            )
            rope_err = (rope_dq - rope_f32).abs()
            rope_bound = rope_f32.abs() * (2**-4 + 2**-10) + stored_scale * 2**-9 + 2**-12
            assert (rope_err <= rope_bound).all(), (
                f"e4m3 rope error {rope_err.max().item():.5f} exceeds the "
                f"half-step bound at token {i}"
            )
        else:
            # RoPE lane is a verbatim 16-bit copy.
            kv_rope = record[rope_offset:].view(dtype)
            torch.testing.assert_close(kv_rope, k_pe[i], atol=0.0, rtol=0.0)

    # Slots outside the mapping stay untouched (indexing/stride isolation).
    written = torch.zeros(total_slots, dtype=torch.bool, device=device)
    written[slot_mapping] = True
    untouched = kv_cache.reshape(total_slots, entry_size)[~written]
    assert (untouched == 0).all()


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_concat_and_cache_nf3_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    _run_concat_and_cache_nf3_family(
        kv_lora_rank, qk_rope_head_dim, num_tokens, block_size, num_blocks,
        dtype, seed, device, kv_cache_dtype="nf3_ds_mla",
    )


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_concat_and_cache_nf3bf16_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    _run_concat_and_cache_nf3_family(
        kv_lora_rank, qk_rope_head_dim, num_tokens, block_size, num_blocks,
        dtype, seed, device, kv_cache_dtype="nf3bf16_ds_mla",
    )


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_swap_blocks_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    entry_size = kv_lora_rank + qk_rope_head_dim

    src_cache = _create_mla_cache(
        num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
    )
    dst_cache = _create_mla_cache(
        num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
    )

    _fill_mla_cache(src_cache, kv_cache_dtype)
    _fill_mla_cache(dst_cache, kv_cache_dtype)

    src_cache_clone = src_cache.clone()

    num_mappings = min(2, num_blocks // 2)
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remaining_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remaining_blocks, num_mappings)
    block_mapping = list(zip(src_blocks, dst_blocks))
    block_mapping_tensor = torch.tensor(
        block_mapping, dtype=torch.int64, device="cpu"
    ).view(-1, 2)

    block_size_in_bytes = src_cache.element_size() * src_cache.stride(0)
    opcheck(
        torch.ops._C_cache_ops.swap_blocks,
        (src_cache, dst_cache, block_size_in_bytes, block_mapping_tensor),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.swap_blocks(src_cache, dst_cache, block_size_in_bytes, block_mapping_tensor)

    for src, dst in block_mapping:
        torch.testing.assert_close(
            src_cache_clone[src].cpu(),
            dst_cache[dst].cpu(),
            msg=f"Block {src} from src should have been swapped to block "
            f"{dst} in dst_cache.",
        )


@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [1024])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_gather_and_maybe_dequant_cache_mla(
    kv_lora_rank,
    qk_rope_head_dim,
    block_size,
    num_blocks,
    max_seq_len,
    batch_size,
    dtype,
    kv_cache_dtype,
    device,
):
    entry_size = kv_lora_rank + qk_rope_head_dim
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    src_cache = _create_mla_cache(
        num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
    )
    _fill_mla_cache(src_cache, kv_cache_dtype=kv_cache_dtype)

    seq_len_tensor = torch.randint(
        max_seq_len, max_seq_len + 1, (batch_size,), device=device
    )

    total_tokens = seq_len_tensor.sum()
    cu_seq_lens = torch.empty((batch_size + 1), dtype=torch.int32, device=device)
    cu_seq_lens[0] = 0
    cu_seq_lens[1:] = seq_len_tensor.cumsum(dim=0).to(dtype=torch.int32)
    token_to_seq = torch.arange(0, batch_size, dtype=torch.int32, device=device)
    token_to_seq = torch.repeat_interleave(token_to_seq, seq_len_tensor)
    print("seq_len_tensor", seq_len_tensor)

    tot_blocks_tensor = (seq_len_tensor + block_size - 1) // block_size
    block_table = torch.empty(
        (batch_size, num_blocks), dtype=torch.int32, device=device
    )

    for b in range(batch_size):
        perm = torch.randperm(num_blocks, device=device)
        block_table[b, :] = perm

    dst = torch.zeros((total_tokens, entry_size), dtype=dtype, device=device)

    expected_batches = []
    for b in range(batch_size):
        s = seq_len_tensor[b]
        if s == 0:
            continue
        tot = tot_blocks_tensor[b]
        blocks = block_table[b, :tot].tolist()

        gathered_rows = []
        for i in range(tot - 1):
            block_data = src_cache[blocks[i]]
            if kv_cache_dtype == "fp8":
                dequantized_block = torch.empty_like(block_data, dtype=dtype)
                ops.convert_fp8(dequantized_block, block_data, scale.item())
                gathered_rows.append(dequantized_block)
            else:
                gathered_rows.append(block_data)
        remaining = s - (tot - 1) * block_size
        last_block_data = src_cache[blocks[-1], :remaining, :]
        if kv_cache_dtype == "fp8":
            dequantized_last_block = torch.empty_like(last_block_data, dtype=dtype)
            ops.convert_fp8(dequantized_last_block, last_block_data, scale.item())
            gathered_rows.append(dequantized_last_block)
        else:
            gathered_rows.append(last_block_data)

        batch_expected = torch.cat(gathered_rows, dim=0)
        expected_batches.append(batch_expected)
    expected = torch.cat(expected_batches, dim=0)

    opcheck(
        torch.ops._C_cache_ops.gather_and_maybe_dequant_cache,
        (
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            total_tokens,
            kv_cache_dtype,
            scale,
            None,
        ),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        total_tokens,
        kv_cache_dtype,
        scale,
        None,
    )
    torch.testing.assert_close(dst, expected)


@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [1024])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "kv_cache_dtype", ["auto"]
)  # You can also test "fp8" if needed.
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_cp_gather_cache_mla(
    kv_lora_rank,
    qk_rope_head_dim,
    block_size,
    num_blocks,
    max_seq_len,
    batch_size,
    dtype,
    kv_cache_dtype,
    device,
):
    entry_size = kv_lora_rank + qk_rope_head_dim
    src_cache = _create_mla_cache(
        num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
    )
    _fill_mla_cache(src_cache, kv_cache_dtype=kv_cache_dtype)

    seq_len_tensor = torch.randint(0, max_seq_len + 1, (batch_size,), device=device)

    total_tokens = seq_len_tensor.sum()
    cu_seq_lens = torch.empty((batch_size + 1), dtype=torch.int32, device=device)
    cu_seq_lens[0] = 0
    cu_seq_lens[1:] = seq_len_tensor.cumsum(dim=0).to(dtype=torch.int32)
    print("seq_len_tensor", seq_len_tensor)

    tot_blocks_tensor = (seq_len_tensor + block_size - 1) // block_size
    block_table = torch.empty(
        (batch_size, num_blocks), dtype=torch.int32, device=device
    )

    for b in range(batch_size):
        perm = torch.randperm(num_blocks, device=device)
        block_table[b, :] = perm

    dst = torch.zeros((total_tokens, entry_size), dtype=src_cache.dtype, device=device)

    expected_batches = []
    for b in range(batch_size):
        s = seq_len_tensor[b]
        if s == 0:
            continue
        tot = tot_blocks_tensor[b]
        blocks = block_table[b, :tot].tolist()

        gathered_rows = []
        for i in range(tot - 1):
            gathered_rows.append(src_cache[blocks[i]])
        remaining = s - (tot - 1) * block_size
        gathered_rows.append(src_cache[blocks[-1], :remaining, :])

        batch_expected = torch.cat(gathered_rows, dim=0)
        expected_batches.append(batch_expected)
    expected = torch.cat(expected_batches, dim=0)

    opcheck(
        torch.ops._C_cache_ops.cp_gather_cache,
        (src_cache, dst, block_table, cu_seq_lens, batch_size, None),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.cp_gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size)
    torch.testing.assert_close(dst, expected)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.cpu_model
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
@torch.inference_mode()
def test_concat_and_cache_mla_cpu(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    device = "cpu"
    kv_cache_dtype = "auto"
    set_random_seed(seed)
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(
        num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
    )
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = torch.empty_like(ref_temp, dtype=kv_cache.dtype)
        ops.convert_fp8(ref_kv_cache, ref_temp, scale.item(), kv_dtype=kv_cache_dtype)
    else:
        ref_kv_cache = ref_temp

    opcheck(
        torch.ops._C_cache_ops.concat_and_cache_mla,
        (kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)
    torch.testing.assert_close(kv_cache, ref_kv_cache)
