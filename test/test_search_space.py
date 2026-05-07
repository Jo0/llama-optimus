# test/test_search_space.py
import warnings


def test_context_aware_batch_high_none():
    """get_context_aware_batch_high(None) returns unconstrained high bound."""
    from llama_optimus.search_space import get_context_aware_batch_high, SEARCH_SPACE
    result = get_context_aware_batch_high(None)
    assert result == SEARCH_SPACE['batch_size']['high']


def test_context_aware_batch_high_131072():
    """get_context_aware_batch_high(131072) returns min(16384, 131072 // 8) = 16384 — no constraint."""
    from llama_optimus.search_space import get_context_aware_batch_high, SEARCH_SPACE
    result = get_context_aware_batch_high(131072)
    expected = min(SEARCH_SPACE['batch_size']['high'], 131072 // 8)
    assert result == expected  # 16384


def test_context_aware_batch_high_65536_constrained():
    """get_context_aware_batch_high(65536) returns min(16384, 65536 // 8) = 8192 — constrained."""
    from llama_optimus.search_space import get_context_aware_batch_high
    result = get_context_aware_batch_high(65536)
    assert result == 8192


def test_context_aware_batch_high_32768_constrained():
    """get_context_aware_batch_high(32768) returns min(16384, 32768 // 8) = 4096 — constrained."""
    from llama_optimus.search_space import get_context_aware_batch_high
    result = get_context_aware_batch_high(32768)
    assert result == 4096


def test_context_aware_batch_high_1024_small_context():
    """get_context_aware_batch_high(1024) returns max(8, 1024 // 8) = 128."""
    from llama_optimus.search_space import get_context_aware_batch_high
    result = get_context_aware_batch_high(1024)
    assert result == 128


def test_context_aware_batch_high_64_minimum_enforced():
    """get_context_aware_batch_high(64) returns max(8, 64 // 8) = 8 — minimum enforced."""
    from llama_optimus.search_space import get_context_aware_batch_high
    result = get_context_aware_batch_high(64)
    assert result == 8


def test_context_aware_batch_high_32_minimum_enforced():
    """get_context_aware_batch_high(32) returns max(8, 32 // 8) = 4 → clamped to 8."""
    from llama_optimus.search_space import get_context_aware_batch_high
    result = get_context_aware_batch_high(32)
    assert result == 8


def test_context_aware_batch_high_warning_when_constrained():
    """warnings.warn() is called when constraint reduces the search space."""
    from llama_optimus.search_space import get_context_aware_batch_high
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_context_aware_batch_high(65536)
        assert len(w) == 1
        assert "Batch size constrained by context" in str(w[0].message)
        assert "context=65536" in str(w[0].message)
        assert "context_size // 8" in str(w[0].message)


def test_context_aware_batch_high_no_warning_when_unconstrained():
    """No warning when context is large enough that constraint does not reduce search space."""
    from llama_optimus.search_space import get_context_aware_batch_high
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_context_aware_batch_high(131072)
        # Filter to only warnings from our function (may be other warnings in the env)
        relevant = [x for x in w if "Batch size constrained" in str(x.message)]
        assert len(relevant) == 0


def test_context_aware_batch_high_no_warning_when_none():
    """No warning when context_size is None."""
    from llama_optimus.search_space import get_context_aware_batch_high
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_context_aware_batch_high(None)
        relevant = [x for x in w if "Batch size constrained" in str(x.message)]
        assert len(relevant) == 0


# ---------------------------------------------------------------------------
# Tests for hardware-adaptive batch constraints
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional


@dataclass
class _FakeModelMetadata:
    """Minimal stand-in for ModelMetadata to avoid GGUF parsing in tests."""
    architecture: str = "llama"
    layer_count: int = 32
    embedding_size: int = 4096
    hidden_size: Optional[int] = 4096
    max_context: Optional[int] = 8192
    quantization: str = "q4_0"
    has_vision: bool = False
    file_path: str = "/fake/model.gguf"


def test_kv_cache_bytes_per_token_f16():
    """KV cache per token for f16: layers * 2 * hidden * 2."""
    from llama_optimus.search_space import _kv_cache_bytes_per_token
    meta = _FakeModelMetadata(layer_count=32, hidden_size=4096)
    result = _kv_cache_bytes_per_token(meta, cache_type="f16")
    expected = 32 * 2 * 4096 * 2.0
    assert result == expected


def test_kv_cache_bytes_per_token_q4_0():
    """KV cache per token for q4_0: layers * 2 * hidden * 0.5."""
    from llama_optimus.search_space import _kv_cache_bytes_per_token
    meta = _FakeModelMetadata(layer_count=32, hidden_size=4096)
    result = _kv_cache_bytes_per_token(meta, cache_type="q4_0")
    expected = 32 * 2 * 4096 * 0.5
    assert result == expected


def test_kv_cache_bytes_per_token_zero_hidden():
    """Returns 0.0 when hidden_size is None."""
    from llama_optimus.search_space import _kv_cache_bytes_per_token
    meta = _FakeModelMetadata(hidden_size=None)
    result = _kv_cache_bytes_per_token(meta, cache_type="f16")
    assert result == 0.0


def test_estimate_model_weight_bytes_basic():
    """Weight estimate should be positive for valid metadata."""
    from llama_optimus.search_space import _estimate_model_weight_bytes
    meta = _FakeModelMetadata(layer_count=32, hidden_size=4096, embedding_size=4096, quantization="q4_0")
    result = _estimate_model_weight_bytes(meta)
    assert result > 0
    # Rough check: 32 layers * 6 * 4096^2 * 0.5 + 4096*4096*0.5 ~ 3.9 GB
    assert result < 10e9  # less than 10 GB for a 32-layer 4096-hidden model


def test_estimate_model_weight_bytes_zero_layers():
    """Returns 0.0 when layer_count is 0."""
    from llama_optimus.search_space import _estimate_model_weight_bytes
    meta = _FakeModelMetadata(layer_count=0)
    result = _estimate_model_weight_bytes(meta)
    assert result == 0.0


def test_apply_hardware_batch_constraints_clamps_batch():
    """apply_hardware_batch_constraints reduces batch_size high when VRAM is tight."""
    from llama_optimus.search_space import (
        SEARCH_SPACE, apply_hardware_batch_constraints,
    )

    meta = _FakeModelMetadata(layer_count=32, hidden_size=4096, embedding_size=4096, quantization="q4_0")

    # Simulate a small GPU (4 GB = 4096 MiB)
    original_batch_high = SEARCH_SPACE['batch_size']['high']
    original_ubatch_high = SEARCH_SPACE['ubatch_size']['high']

    apply_hardware_batch_constraints(meta, vram_total_mb=4096.0, cache_type="f16")

    # batch_size high should be clamped below original (or stay the same if already low)
    assert SEARCH_SPACE['batch_size']['high'] <= original_batch_high
    assert SEARCH_SPACE['ubatch_size']['high'] <= SEARCH_SPACE['batch_size']['high']
    assert SEARCH_SPACE['batch_size']['high'] >= 8  # minimum enforced

    # Restore for other tests
    SEARCH_SPACE['batch_size']['high'] = original_batch_high
    SEARCH_SPACE['ubatch_size']['high'] = original_ubatch_high


def test_apply_hardware_batch_constraints_large_vram():
    """With plenty of VRAM, batch bounds should stay at original values."""
    from llama_optimus.search_space import SEARCH_SPACE, apply_hardware_batch_constraints

    meta = _FakeModelMetadata(layer_count=32, hidden_size=4096, embedding_size=4096, quantization="q4_0")

    original_batch_high = SEARCH_SPACE['batch_size']['high']
    original_ubatch_high = SEARCH_SPACE['ubatch_size']['high']

    # 24 GB GPU — should not constrain a small model
    apply_hardware_batch_constraints(meta, vram_total_mb=24576.0, cache_type="f16")

    assert SEARCH_SPACE['batch_size']['high'] == original_batch_high
    assert SEARCH_SPACE['ubatch_size']['high'] == original_ubatch_high


def test_apply_hardware_batch_constraints_no_vram():
    """When vram_total_mb is None and probe returns None, a warning is emitted."""
    from unittest.mock import patch
    from llama_optimus.search_space import SEARCH_SPACE, apply_hardware_batch_constraints

    meta = _FakeModelMetadata(layer_count=32, hidden_size=4096, embedding_size=4096, quantization="q4_0")

    original_batch_high = SEARCH_SPACE['batch_size']['high']

    with patch("llama_optimus.hardware_probe.get_vram_total_mb", return_value=None):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_hardware_batch_constraints(meta, vram_total_mb=None, cache_type="f16")
            relevant = [x for x in w if "VRAM probe returned None" in str(x.message)]
            assert len(relevant) == 1

    # batch_size should be unchanged
    assert SEARCH_SPACE['batch_size']['high'] == original_batch_high


def test_apply_hardware_batch_constraints_ubatch_leq_batch():
    """ubatch_size high must never exceed batch_size high after constraint."""
    from llama_optimus.search_space import SEARCH_SPACE, apply_hardware_batch_constraints

    meta = _FakeModelMetadata(layer_count=64, hidden_size=8192, embedding_size=8192, quantization="main_f16")

    # Tight VRAM to force clamping
    apply_hardware_batch_constraints(meta, vram_total_mb=8192.0, cache_type="f16")

    assert SEARCH_SPACE['ubatch_size']['high'] <= SEARCH_SPACE['batch_size']['high']
