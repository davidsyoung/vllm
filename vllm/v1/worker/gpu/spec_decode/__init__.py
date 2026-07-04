# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.method == "dspark":
        # Two distinct DSpark flavors share method=="dspark":
        #   * Standalone speculators-format checkpoint (#47093): a separate
        #     Qwen3-style dense draft repo whose `update_dspark` algo sets
        #     architectures=["Qwen3DSparkModel"]. It is a DFlash subclass
        #     (qwen3_dspark.py) driven by the fork's DFlash block proposer PLUS
        #     the sequential Markov head -> route to the standalone
        #     Qwen3DSparkSpeculator (dspark_qwen3.speculator), which extends
        #     DFlashSpeculator to actually consult the Markov head at sampling
        #     time. (Previously this routed to the stock DFlashSpeculator, which
        #     loaded the head but never consulted it; see IMPL_NOTES.md.)
        #   * Integrated DeepSeek-V4 DSpark: draft ships inside the target
        #     (mtp.* namespace), no Qwen3DSparkModel arch -> keep the fork's
        #     dedicated tilelang DSparkSpeculator (dspark.speculator), untouched.
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        draft_architectures = (
            getattr(draft_model_config, "architectures", None) or []
        )
        if "Qwen3DSparkModel" in draft_architectures:
            from vllm.v1.worker.gpu.spec_decode.dspark_qwen3.speculator import (
                Qwen3DSparkSpeculator,
            )

            return Qwen3DSparkSpeculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)
    elif speculative_config.use_gemma4_mtp():
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(vllm_config, device)
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)
    elif speculative_config.method == "dflash":
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(vllm_config, device)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
