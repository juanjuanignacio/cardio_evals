"""Model loading: TransformerLens HookedTransformer (single GPU) and HF multi-GPU backend."""

from __future__ import annotations

import gc
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hooked_transformer(
    model_name: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    fold_ln: bool = False,
    center_writing_weights: bool = False,
    center_unembed: bool = False,
    hf_token: Optional[str] = None,
):
    """
    Load model via TransformerLens HookedTransformer.

    Key mechanistic interp settings:
    - fold_ln=False: keep LayerNorm separate for correct residual stream patching
    - center_writing_weights=False: don't distort residual stream geometry
    - center_unembed=False: same reason

    Returns (hooked_model, tokenizer).
    """
    import huggingface_hub
    from transformer_lens import HookedTransformer
    from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

    # Authenticate globally so TL can download gated models without extra config
    if hf_token:
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_name in OFFICIAL_MODEL_NAMES:
        # TL has native support — downloads weights directly
        model = HookedTransformer.from_pretrained(
            model_name,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_ln=fold_ln,
            dtype=dtype,
            device=device,
        )
    else:
        # Non-official model: load HF weights first, then convert via closest TL
        # architecture config. The TL arch provides the hook names / config shape;
        # hf_model provides the actual weights.
        tl_arch = _find_tl_architecture(model_name)
        print(f"  Non-official model — loading HF weights from {model_name!r}")
        print(f"  using TL architecture template: {tl_arch!r}")
        # Load to CPU — TL's from_pretrained moves everything to `device` internally.
        # Using device_map here causes mixed-device tensors that crash TL's fold_value_biases.
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            token=hf_token,
        )
        model = HookedTransformer.from_pretrained(
            tl_arch,
            hf_model=hf_model,
            tokenizer=tokenizer,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_ln=fold_ln,
            dtype=dtype,
            device=device,
        )
        del hf_model  # free the raw HF copy; TL has ingested the weights

    model.eval()
    return model, tokenizer


def _find_tl_architecture(model_name: str) -> str:
    """
    Map a HuggingFace model name to the nearest TransformerLens architecture name.
    Used as fallback when the exact model isn't in TL's official list.

    The returned name is used ONLY for TL's architecture config (hook names, layer
    structure). Actual weights are loaded separately from the real HF model and
    passed via hf_model= in load_hooked_transformer.
    """
    name_lower = model_name.lower()

    # DeepSeek distilled models
    if "deepseek" in name_lower:
        if "qwen" in name_lower:
            # DeepSeek-R1-Distill-Qwen-*: Qwen2 architecture (28L/3584d for 7B)
            return "Qwen/Qwen2-7B-Instruct"
        if "llama" in name_lower:
            # DeepSeek-R1-Distill-Llama-*: Llama 3 architecture
            return "meta-llama/Meta-Llama-3-8B-Instruct"

    # Llama family
    if "llama-3.1" in name_lower or "llama-3_1" in name_lower:
        return "meta-llama/Llama-3.1-8B-Instruct"
    if "llama-3.3" in name_lower or "llama-3_3" in name_lower:
        return "meta-llama/Llama-3.1-8B-Instruct"   # same arch, different weights
    if "llama-3" in name_lower:
        if "instruct" in name_lower:
            return "meta-llama/Meta-Llama-3-8B-Instruct"
        return "meta-llama/Meta-Llama-3-8B"

    # Mistral family
    if "mistral" in name_lower:
        return "mistralai/Mistral-7B-Instruct-v0.1"

    # Qwen family — Qwen2 and Qwen2.5 share the same TL architecture config
    if "qwen2" in name_lower or "qwen2.5" in name_lower:
        return "Qwen/Qwen2-7B-Instruct"

    # Phi family
    if "phi-4" in name_lower:
        return "microsoft/Phi-3-medium-4k-instruct"   # 14B Phi; closest TL arch
    if "phi-3" in name_lower:
        return "microsoft/Phi-3-mini-4k-instruct"
    if "phi" in name_lower:
        return "microsoft/phi-2"

    # Gemma family
    if "gemma-2" in name_lower:
        return "google/gemma-7b-it"

    raise ValueError(
        f"No TL architecture mapping for {model_name!r}. "
        "Add it to _find_tl_architecture() in src/model.py."
    )


def load_hf_model_large(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load large model (70B+) in bfloat16 across visible GPUs via device_map="auto".

    IMPORTANT — shared node safety: device_map="auto" uses ALL GPUs visible to the
    process. On a multi-user node, set CUDA_VISIBLE_DEVICES before running:

        CUDA_VISIBLE_DEVICES=0,1 python blocks/...

    This restricts the process to only the GPUs you control, preventing accidental
    colonization of GPUs reserved by other users.

    With 2x H100 80GB (160 GB total): Llama 3.3 70B (~140 GB bfloat16) fits without
    quantization — cleaner results than 4-bit.
    """
    import os
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs visible. Set CUDA_VISIBLE_DEVICES appropriately.")
    if n_gpus > 2:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise RuntimeError(
                f"load_hf_model_large sees {n_gpus} GPUs but CUDA_VISIBLE_DEVICES is not set. "
                "On a shared node this would use GPUs allocated to other users. "
                "Set CUDA_VISIBLE_DEVICES=0,1 (or whichever GPUs are yours) before running."
            )

    # Resolve to local snapshot path to avoid safetensors format issues with
    # device_map='auto' + accelerate when loading via HF model ID string.
    load_path: str = model_name
    try:
        from huggingface_hub import snapshot_download
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hub_dir = os.path.join(hf_home, "hub")
        local_snapshot = snapshot_download(
            model_name,
            cache_dir=hub_dir,
            local_files_only=True,
            token=hf_token,
        )
        load_path = local_snapshot
        print(f"  Resolved local snapshot: {local_snapshot}")
    except Exception as e:
        print(f"  [WARN] Could not resolve local snapshot ({e}), loading from HF Hub directly.")

    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        token=hf_token,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=dtype,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    print(f"  Loaded {model_name} across {n_gpus} GPU(s) in {dtype}")
    return model, tokenizer


# Keep 4-bit loader as fallback for single-GPU setups with insufficient VRAM
def load_hf_model_4bit(
    model_name: str,
    device: str = "cuda:0",
    hf_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model with bitsandbytes 4-bit quantization (NF4).
    Fallback for single-GPU setups where the model doesn't fit at full precision.
    Prefer load_hf_model_large() with multiple GPUs when available.
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError("bitsandbytes not installed. Use load_hf_model_large() instead.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": device},
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


def load_hf_model_full(
    model_name: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with HuggingFace (full precision) for hidden state extraction fallback."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


def free_model_memory(model) -> None:
    """
    Release GPU memory WITHOUT moving model weights to CPU first.

    CRITICAL for shared H100: TransformerLens HookedTransformer.__del__ calls
    .to('cpu') as a finalizer, which causes a 16 GB PCIe transfer spike that
    crashes SSH proxies and stalls other processes on the same GPU.

    Fix: replace every parameter tensor with a 0-element tensor (still on GPU)
    BEFORE deleting. The finalizer then moves ~0 bytes over PCIe instead of 16 GB.
    CUDA frees the original pages when the 0-element tensors go out of scope.

    We do NOT call empty_cache() here — use it only explicitly in scripts that
    need to reclaim VRAM before loading the next model.
    """
    try:
        # Synchronize first: ensure no async CUDA ops are in flight when we start tearing down
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        # Remove all hooks first — they may hold references to full-size activation tensors
        # from run_with_cache / calibration that would otherwise survive the param wipe
        if hasattr(model, "reset_hooks"):
            model.reset_hooks(including_permanent=True)
        for p in model.parameters():
            p.data = p.data.new_empty(0)
        for b in model.buffers():
            b.data = b.data.new_empty(0)
    except Exception:
        pass
    del model
    gc.collect()


def exit_cleanly(code: int = 0) -> None:
    """Exit without Python finalizers to prevent CUDA teardown kernel panics.

    Normal Python exit runs __del__ and module teardown while the CUDA context
    is being destroyed — on shared H100s this causes kernel panics and node reboots.
    os._exit() skips all finalizers; the OS reclaims GPU memory atomically.

    Call this as the LAST statement in any script that loaded a GPU model,
    AFTER all files have been saved and closed.
    """
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]
