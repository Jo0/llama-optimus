# src/llama_optimus/search_space.py
import math
import os
import warnings
from typing import Optional

from .override_patterns import OVERRIDE_PATTERNS  # if needed
from .model_info import ModelMetadata

# count number of available cpu cores
max_threads = os.cpu_count()


# ---------------------------------------------------------------------------
# KV-cache bytes-per-element for each supported cache quantization type
# ---------------------------------------------------------------------------
_CACHE_TYPE_BYTES: dict[str, float] = {
    "f16":    2.0,
    "bf16":   2.0,
    "q8_0":   1.0,
    "q5_0":   0.625,   # 5 bits per element
    "q4_0":   0.5,     # 4 bits per element
    "q4_1":   0.5,
    "iq4_nl": 0.5,
}

# Conservative safety margin: reserve this fraction of VRAM for model weights,
# OS overhead, and activation buffers so the KV-cache budget is realistic.
_VRAM_SAFETY_MARGIN = 0.35


# ---------------------------------------------------------------------------
# Quantization bytes-per-parameter lookup
# ---------------------------------------------------------------------------
_QUANT_BYTES_PER_PARAM: dict[str, float] = {
    # f32 / f16
    "all_f32":  4.0,
    "main_f16": 2.0,
    "q4_0_4_4": 0.5,
    "q4_0_4_8": 0.5,
    "q4_0_8_8": 0.5,
    # q4
    "q4_0": 0.5,
    "q4_1": 0.5625,   # includes 32-entry quantization table
    # q5
    "q5_0": 0.625,
    "q5_1": 0.6875,
    # q8
    "q8_0": 1.0,
    # K-quants
    "q2_K":    0.4375,
    "q3_K_S":  0.6172,
    "q3_K_M":  0.6562,
    "q3_K_L":  0.7031,
    "q4_K_S":  0.5469,
    "q4_K_M":  0.5938,
    "q5_K_S":  0.6797,
    "q5_K_M":  0.7188,
    "q6_K":    0.75,
    "q8_K_L":  1.0,
    # IQ quants
    "iq2_XXS": 0.3125,
    "iq2_XS":  0.375,
    "iq3_XS":  0.4375,
    "iq3_XXS": 0.375,
    "iq1_S":   0.25,
    "iq4_NL":  0.5,
    "iq3_M":   0.5,
    "iq2_M":   0.375,
    "bs8_5":   0.5,
    "iq1_M":   0.25,
    "iq4_XS":  0.5,
    "iq2_LS":  0.375,
    "iq3_XM":  0.5,
    "iq1_XXS": 0.1875,
    "iq2XS":   0.375,
    "iq1XS":   0.25,
    "iq4XS":   0.5,
    "iq2S":    0.375,
    "iq1S":    0.25,
    "iq4M":    0.5,
}


def _estimate_model_weight_bytes(
    model_metadata: ModelMetadata,
    num_layers_override: Optional[int] = None,
) -> float:
    """Estimate the total weight memory (bytes) for a GGUF model.

    Uses a simplified transformer parameter-count formula:

        params ≈ layers × (hidden² × 4  +  hidden × embedding  × 2  +  hidden²)
                + embedding × hidden × 2

    For MoE models the FFN multiplier is higher, but this formula is
    conservative enough for bounding purposes.

    Parameters
        model_metadata: Parsed GGUF metadata.
        num_layers_override: If provided, use this instead of
            ``model_metadata.layer_count`` (useful for testing).

    Returns:
        Estimated weight memory in bytes.
    """
    layers = num_layers_override or model_metadata.layer_count
    hidden = model_metadata.hidden_size
    embd = model_embedding_size = model_metadata.embedding_size

    if not hidden or not embd or layers <= 0:
        return 0.0

    bytes_per_param = _QUANT_BYTES_PER_PARAM.get(model_metadata.quantization, 2.0)

    # Simplified parameter count for a dense transformer:
    #   - Attention: 4 × hidden² per layer (Q, K, V, O projections + output)
    #   - FFN:       ~2 × hidden² per layer (gate + up + down, approximated)
    #   - Norms + biases are negligible
    #   - Token embedding: embedding × hidden
    params_per_layer = hidden * hidden * 6  # attention(4) + ffn(2)
    params_embedding = embd * hidden

    total_params = layers * params_per_layer + params_embedding
    return total_params * bytes_per_param


def _kv_cache_bytes_per_token(
    model_metadata: ModelMetadata,
    cache_type: str = "f16",
) -> float:
    """Estimate KV-cache memory (bytes) required for a **single token**.

    The KV cache stores key and value states for every layer:

        bytes = layers × 2 × hidden × bytes_per_element

    Parameters:
        model_metadata: Parsed GGUF metadata.
        cache_type: KV cache quantization type (default ``f16``).

    Returns:
        Bytes needed per token in the KV cache.
    """
    layers = model_metadata.layer_count
    hidden = model_metadata.hidden_size
    if not hidden or layers <= 0:
        return 0.0

    bytes_per_element = _CACHE_TYPE_BYTES.get(cache_type, 2.0)
    return layers * 2 * hidden * bytes_per_element


def apply_hardware_batch_constraints(
    model_metadata: ModelMetadata,
    vram_total_mb: Optional[float] = None,
    cache_type: str = "f16",
) -> None:
    """Tighten ``batch_size`` / ``ubatch_size`` bounds to fit available VRAM.

    Estimates the per-token KV-cache footprint and the model weight memory,
    then computes the maximum batch size that leaves enough headroom.

    Formula:
        remaining_vram = vram_total × (1 - safety_margin) - weight_bytes
        max_batch = remaining_vram / kv_cache_per_token

    The constraint is applied **in-place** to the global ``SEARCH_SPACE`` dict,
    following the same pattern as :func:`apply_model_constraints`.

    Parameters:
        model_metadata: Parsed GGUF metadata.
        vram_total_mb: Total GPU VRAM in MiB. If ``None``, probes the GPU
            via ``hardware_probe.get_vram_total_mb()``.
        cache_type: KV cache quantization type used for the estimate.

    Side-effects:
        Mutates ``SEARCH_SPACE['batch_size']['high']`` and
        ``SEARCH_SPACE['ubatch_size']['high']``.
    """
    from .hardware_probe import get_vram_total_mb  # noqa: PLC0415

    if vram_total_mb is None:
        vram_total_mb = get_vram_total_mb()

    if vram_total_mb is None or vram_total_mb <= 0:
        warnings.warn(
            "VRAM probe returned None or 0 — batch constraints skipped. "
            "This is normal for CPU-only or non-NVIDIA setups."
        )
        return

    vram_total_bytes = vram_total_mb * 1024.0 * 1024.0

    weight_bytes = _estimate_model_weight_bytes(model_metadata)
    kv_per_token = _kv_cache_bytes_per_token(model_metadata, cache_type)

    if kv_per_token <= 0:
        warnings.warn(
            "KV cache per-token estimate is 0 (missing hidden_size or layers) "
            "— batch constraints skipped."
        )
        return

    # Reserve a fraction for OS, driver, activation buffers
    budget_for_kv = vram_total_bytes * (1.0 - _VRAM_SAFETY_MARGIN) - weight_bytes

    if budget_for_kv <= 0:
        # Model weights alone exceed the budget → clamp to minimum
        max_batch = 8
        reason = "model weights exceed VRAM budget"
    else:
        max_batch = max(8, int(budget_for_kv / kv_per_token))
        reason = f"VRAM budget ({vram_total_mb:.0f} MiB, model ~{weight_bytes / 1e6:.0f} MB)"

    # Clamp batch_size
    old_batch_high = SEARCH_SPACE['batch_size']['high']
    SEARCH_SPACE['batch_size']['high'] = min(old_batch_high, max_batch)

    # Clamp ubatch_size (must be ≤ batch_size)
    old_ubatch_high = SEARCH_SPACE['ubatch_size']['high']
    SEARCH_SPACE['ubatch_size']['high'] = min(
        old_ubatch_high,
        SEARCH_SPACE['batch_size']['high'],
        max_batch,
    )

    if SEARCH_SPACE['batch_size']['high'] < old_batch_high or \
       SEARCH_SPACE['ubatch_size']['high'] < old_ubatch_high:
        warnings.warn(
            f"Batch size constrained by VRAM: "
            f"batch_max={SEARCH_SPACE['batch_size']['high']}, "
            f"ubatch_max={SEARCH_SPACE['ubatch_size']['high']} "
            f"({reason}, kv_per_token={kv_per_token:.0f} B)"
        )

SEARCH_SPACE = {
    'batch_size'     : {'low': 8, 'high': 16384},   #
    'ubatch_size'    : {'low': 4, 'high': 8192},    #
    'threads':    {'low': 1, 'high': max_threads},  # Adjust range to your hardware
    'gpu_layers': {'low': 0, 'high': 149},          # (-ngl) Set max according to model and VRAM; The max value must be determined for each setup
    'flash_attn': [0,1],                            #  --flash-attn <0|1> ; Enables flash attention
    'override_spc'   : list(OVERRIDE_PATTERNS.keys()), # Read list from src/llama_optimus/override_patterns.py
    'cache_type': ['f16', 'bf16', 'q8_0', 'q5_0', 'q4_0', 'q4_1', 'iq4_nl'],  # KV cache quantization type (symmetric K+V); q4_0 saves ~2.6 GB VRAM vs f16
    'mmap': [0, 1],                                  # Memory mapping: 0 = disabled (--mmap 0), 1 = enabled (--mmap 1, default); disabling helps on Windows to avoid pagefile churn
    #'flash_attn_type': [0, 1, 2], # Not yet merged to main llama.cpp
}


def apply_model_constraints(model_metadata: ModelMetadata) -> None:
    """Update SEARCH_SPACE bounds based on model metadata.

    Tightens ``gpu_layers`` upper bound to the actual number of layers
    in the model, so that values like ``-ngl 146`` on a 64-layer model
    are caught at search-space level rather than silently clamped by
    llama-bench.
    """
    if model_metadata.layer_count > 0:
        SEARCH_SPACE['gpu_layers']['high'] = min(
            SEARCH_SPACE['gpu_layers']['high'],
            model_metadata.layer_count,
        )


def get_context_aware_batch_high(context_size: int = None) -> int:
    """Calculate maximum batch size based on context size.

    Heuristic: max_batch = min(SEARCH_SPACE['batch_size']['high'], context_size // 8)

    Emits a one-time warning when the constraint reduces the search space.

    Parameters:
        context_size: The context window size. If None, returns the unconstrained high bound.

    Returns:
        int: The maximum safe batch size for the given context.
    """
    base_high = SEARCH_SPACE['batch_size']['high']

    if context_size is None:
        # No context constraint (Phase 1 default behavior)
        return base_high

    constrained = min(base_high, context_size // 8)

    if constrained < base_high:
        warnings.warn(
            f"Batch size constrained by context: max_batch={constrained} "
            f"(context={context_size}, rule: context_size // 8, unconstrained={base_high})"
        )

    return max(constrained, 8)  # Ensure minimum batch of 8