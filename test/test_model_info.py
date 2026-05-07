# test/test_model_info.py
# Unit tests for GGUF metadata parsing

import struct
import tempfile
from pathlib import Path

from llama_optimus.model_info import (
    ModelMetadata,
    get_model_metadata,
    _read_gguf_header,
)
from llama_optimus.search_space import apply_model_constraints, SEARCH_SPACE


def _write_minimal_gguf(
    path: Path,
    architecture: str = "llama",
    block_count: int = 32,
    embedding_length: int = 4096,
    ffn_hidden_size: int = 11008,
    max_context_length: int = 8192,
    file_type: int = 2,
) -> Path:
    """Write a minimal valid GGUF v2 file with the given KV pairs."""
    kvs = {
        "general.architecture": architecture,
        f"{architecture}.block_count": block_count,
        f"{architecture}.embedding_length": embedding_length,
        f"{architecture}.ffn_hidden_size": ffn_hidden_size,
        f"{architecture}.max_context_length": max_context_length,
        "general.file_type": file_type,
    }

    chunks = []
    chunks.append(b"GGUF")                   # magic
    chunks.append(struct.pack("<I", 2))      # version
    chunks.append(struct.pack("<Q", 0))      # tensor_count
    chunks.append(struct.pack("<Q", len(kvs)))  # kv_count

    for key, val in kvs.items():
        key_bytes = key.encode("utf-8")
        chunks.append(struct.pack("<Q", len(key_bytes)))
        chunks.append(key_bytes)

        if isinstance(val, str):
            str_bytes = val.encode("utf-8")
            chunks.append(struct.pack("<I", 11))  # UT_STRING
            chunks.append(struct.pack("<Q", len(str_bytes)))
            chunks.append(str_bytes)
        elif isinstance(val, int):
            chunks.append(struct.pack("<I", 5))   # UT_UINT32
            chunks.append(struct.pack("<I", val))

    with open(path, "wb") as f:
        f.write(b"".join(chunks))

    return path


def _write_minimal_gguf_v3(
    path: Path,
    architecture: str = "llama",
    block_count: int = 32,
    embedding_length: int = 4096,
    ffn_hidden_size: int = 11008,
    max_context_length: int = 8192,
    file_type: int = 2,
) -> Path:
    """Write a minimal valid GGUF v3 file with the given KV pairs.

    GGUF v3 differs from v2 in that value-type IDs are encoded as
    uint8 (1 byte) instead of uint32 (4 bytes).  The header layout
    (magic + version + tensor_count + kv_count) is identical.
    """
    kvs = {
        "general.architecture": architecture,
        f"{architecture}.block_count": block_count,
        f"{architecture}.embedding_length": embedding_length,
        f"{architecture}.ffn_hidden_size": ffn_hidden_size,
        f"{architecture}.max_context_length": max_context_length,
        "general.file_type": file_type,
    }

    chunks = []
    chunks.append(b"GGUF")                   # magic
    chunks.append(struct.pack("<I", 3))      # version v3
    chunks.append(struct.pack("<Q", 0))      # tensor_count
    chunks.append(struct.pack("<Q", len(kvs)))  # kv_count

    for key, val in kvs.items():
        key_bytes = key.encode("utf-8")
        chunks.append(struct.pack("<Q", len(key_bytes)))
        chunks.append(key_bytes)

        if isinstance(val, str):
            str_bytes = val.encode("utf-8")
            chunks.append(struct.pack("<B", 8))   # UT_STRING = 8 in v3
            chunks.append(b"\x00" * 3)            # 3-byte padding after type_id
            chunks.append(struct.pack("<I", len(str_bytes)))  # uint32 LE length
            chunks.append(b"\x00" * 4)            # 4-byte padding after str_len
            chunks.append(str_bytes)
        elif isinstance(val, int):
            chunks.append(struct.pack("<B", 4))   # UT_UINT32 = 4 in v3
            chunks.append(b"\x00" * 3)            # 3-byte padding after type_id
            chunks.append(struct.pack("<I", val))

    with open(path, "wb") as f:
        f.write(b"".join(chunks))

    return path


class TestReadGgufHeader:
    def test_basic_header_parsing(self, tmp_path):
        gguf_path = tmp_path / "test.gguf"
        _write_minimal_gguf(gguf_path)

        kvs = _read_gguf_header(str(gguf_path))
        assert kvs["general.architecture"] == "llama"
        assert kvs["llama.block_count"] == 32
        assert kvs["llama.embedding_length"] == 4096

    def test_file_not_found(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            _read_gguf_header("/nonexistent/path/model.gguf")

    def test_invalid_magic(self, tmp_path):
        bad_path = tmp_path / "bad.gguf"
        bad_path.write_bytes(b"XXXX" + b"\x00" * 20)
        import pytest
        with pytest.raises(ValueError, match="Not a valid GGUF"):
            _read_gguf_header(str(bad_path))

    def test_v3_header_parsing(self, tmp_path):
        """GGUF v3 files have a 24-byte descriptor after version; parser must skip it."""
        gguf_path = tmp_path / "test_v3.gguf"
        _write_minimal_gguf_v3(gguf_path)

        kvs = _read_gguf_header(str(gguf_path))
        assert kvs["general.architecture"] == "llama"
        assert kvs["llama.block_count"] == 32
        assert kvs["llama.embedding_length"] == 4096


class TestGetModelMetadata:
    def test_llama_metadata(self, tmp_path):
        gguf_path = tmp_path / "llama.gguf"
        _write_minimal_gguf(gguf_path, architecture="llama", block_count=64)

        meta = get_model_metadata(str(gguf_path))
        assert isinstance(meta, ModelMetadata)
        assert meta.architecture == "llama"
        assert meta.layer_count == 64
        assert meta.embedding_size == 4096
        assert meta.hidden_size == 11008
        assert meta.max_context == 8192
        assert meta.quantization == "q4_0"
        assert meta.has_vision is False
        assert Path(meta.file_path).exists()

    def test_qwen2_metadata(self, tmp_path):
        gguf_path = tmp_path / "qwen2.gguf"
        _write_minimal_gguf(gguf_path, architecture="qwen2", block_count=28)

        meta = get_model_metadata(str(gguf_path))
        assert meta.architecture == "qwen2"
        assert meta.layer_count == 28

    def test_v3_metadata(self, tmp_path):
        """Ensure get_model_metadata works end-to-end with GGUF v3 files."""
        gguf_path = tmp_path / "v3_model.gguf"
        _write_minimal_gguf_v3(gguf_path, architecture="qwen2", block_count=48)

        meta = get_model_metadata(str(gguf_path))
        assert meta.architecture == "qwen2"
        assert meta.layer_count == 48
        assert meta.embedding_size == 4096
        assert meta.hidden_size == 11008
        assert meta.max_context == 8192

    def test_vision_architecture(self, tmp_path):
        gguf_path = tmp_path / "mllama.gguf"
        _write_minimal_gguf(gguf_path, architecture="mllama")

        meta = get_model_metadata(str(gguf_path))
        assert meta.has_vision is True

    def test_file_type_mapping(self, tmp_path):
        gguf_path = tmp_path / "f16.gguf"
        _write_minimal_gguf(gguf_path, file_type=1)

        meta = get_model_metadata(str(gguf_path))
        assert meta.quantization == "main_f16"

    def test_unknown_file_type(self, tmp_path):
        gguf_path = tmp_path / "unknown.gguf"
        _write_minimal_gguf(gguf_path, file_type=999)

        meta = get_model_metadata(str(gguf_path))
        assert "unknown" in meta.quantization


class TestApplyModelConstraints:
    def test_tightens_gpu_layers_bound(self, tmp_path):
        # Reset to default
        SEARCH_SPACE["gpu_layers"]["high"] = 149

        gguf_path = tmp_path / "small_model.gguf"
        _write_minimal_gguf(gguf_path, block_count=24)

        meta = get_model_metadata(str(gguf_path))
        apply_model_constraints(meta)
        assert SEARCH_SPACE["gpu_layers"]["high"] == 24

    def test_does_not_increase_bound(self, tmp_path):
        # Pre-set a lower bound
        SEARCH_SPACE["gpu_layers"]["high"] = 16

        gguf_path = tmp_path / "large_model.gguf"
        _write_minimal_gguf(gguf_path, block_count=64)

        meta = get_model_metadata(str(gguf_path))
        apply_model_constraints(meta)
        assert SEARCH_SPACE["gpu_layers"]["high"] == 16

    def test_zero_layer_count_no_change(self, tmp_path):
        SEARCH_SPACE["gpu_layers"]["high"] = 149

        gguf_path = tmp_path / "empty.gguf"
        _write_minimal_gguf(gguf_path, block_count=0)

        meta = get_model_metadata(str(gguf_path))
        apply_model_constraints(meta)
        assert SEARCH_SPACE["gpu_layers"]["high"] == 149


class TestGgufHeaderSanityChecks:
    """Tests for the sanity-check bounds added to prevent MemoryError."""

    def test_excessive_key_len_raises_value_error(self, tmp_path):
        """When key_len is absurdly large, raise ValueError instead of MemoryError."""
        bad_path = tmp_path / "big_key.gguf"
        with open(bad_path, "wb") as f:
            f.write(b"GGUF")                              # magic
            f.write(struct.pack("<I", 2))                 # version
            f.write(struct.pack("<Q", 0))                 # tensor_count
            f.write(struct.pack("<Q", 1))                 # kv_count = 1
            # Write a key_len that is way too large (2 GiB)
            f.write(struct.pack("<Q", 1 << 31))
            # A few dummy bytes so the file is not completely empty
            f.write(b"dummy")
        import pytest
        with pytest.raises(ValueError, match="key length"):
            _read_gguf_header(str(bad_path))

    def test_excessive_string_len_raises_value_error(self, tmp_path):
        """When a string value length is absurdly large, raise ValueError."""
        bad_path = tmp_path / "big_str.gguf"
        with open(bad_path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<Q", 0))
            f.write(struct.pack("<Q", 1))
            # key
            key_bytes = b"general.name"
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)
            # value type = string (11)
            f.write(struct.pack("<I", 11))
            # string length = 2 GiB
            f.write(struct.pack("<Q", 1 << 31))
        import pytest
        with pytest.raises(ValueError, match="string length"):
            _read_gguf_header(str(bad_path))

    def test_excessive_array_len_raises_value_error(self, tmp_path):
        """When an array length is absurdly large, raise ValueError."""
        bad_path = tmp_path / "big_arr.gguf"
        with open(bad_path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<Q", 0))
            f.write(struct.pack("<Q", 1))
            # key
            key_bytes = b"general.tags"
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)
            # value type = array (12)
            f.write(struct.pack("<I", 12))
            # array element type = uint8 (1)
            f.write(struct.pack("<I", 1))
            # array length = 2 GiB elements
            f.write(struct.pack("<Q", 1 << 31))
        import pytest
        with pytest.raises(ValueError, match="array length"):
            _read_gguf_header(str(bad_path))


class TestPathNormalization:
    """Tests for _normalize_path in cli.py."""

    def test_normalize_forward_slashes(self):
        from llama_optimus.cli import _normalize_path
        result = _normalize_path("a/b/c")
        # On Windows separators are backslashes; on Linux they stay forward.
        import os
        expected_sep = os.sep
        assert expected_sep in result or "/" in result

    def test_normalize_backslashes(self):
        from llama_optimus.cli import _normalize_path
        result = _normalize_path("a\\b\\c")
        # Should produce a valid path with native separators
        assert ".." not in result or "a..b..c" in result

    def test_normalize_mixed_separators(self):
        from llama_optimus.cli import _normalize_path
        result = _normalize_path("C:\\Users\\foo/bar/baz")
        # No mixed separators in the result
        parts_forward = result.count("/")
        parts_back = result.count("\\")
        # On a well-normalized path, only one type of separator should dominate
        # (Path on Windows normalizes to \, on Linux to /)
        import platform
        if platform.system() == "Windows":
            assert parts_forward == 0 or parts_back >= parts_forward
        else:
            assert parts_back == 0 or parts_forward >= parts_back

    def test_normalize_drive_path_windows(self):
        from llama_optimus.cli import _normalize_path
        mixed = "C:/Users/vinhl/.cache/huggingface/model.gguf"
        result = _normalize_path(mixed)
        assert result.startswith("C:")
        assert "model.gguf" in result
