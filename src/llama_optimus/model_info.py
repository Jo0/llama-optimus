# llama_optimus/model_info.py
# GGUF metadata parser — extract architecture info from model headers

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["ModelMetadata", "get_model_metadata"]

# Architectures that include vision/multimodal components
_VISION_ARCHITECTURES = frozenset({"mllama", "clip", "siglip", "idefics3"})

# GGUF general.file_type integer codes → human-readable strings
_FILE_TYPE_MAP = {
    0: "all_f32",
    1: "main_f16",
    2: "q4_0",
    3: "q4_1",
    4: "q4_1_F16",
    5: "q8_0",
    6: "q5_0",
    7: "q5_1",
    8: "q2_K",
    9: "q3_K_S",
    10: "q3_K_M",
    11: "q3_K_L",
    12: "q4_K_S",
    13: "q4_K_M",
    14: "q5_K_S",
    15: "q5_K_M",
    16: "q6_K",
    17: "q8_K_L",
    18: "iq2_XXS",
    19: "iq2_XS",
    20: "iq3_XS",
    21: "iq3_XXS",
    22: "iq1_S",
    23: "iq4_NL",
    24: "iq3_M",
    25: "iq2_M",
    26: "bs8_5",
    27: "iq1_M",
    28: "iq4_XS",
    29: "iq2_LS",
    30: "iq3_XM",
    31: "iq1_XXS",
    32: "iq2XS",
    33: "iq1XS",
    34: "iq4XS",
    35: "iq2S",
    36: "iq1S",
    37: "iq4M",
    38: "q4_0_4_4",
    39: "q4_0_4_8",
    40: "q4_0_8_8",
}


@dataclass
class ModelMetadata:
    """Metadata extracted from a GGUF model file."""
    architecture: str          # e.g., "qwen2", "llama"
    layer_count: int           # actual number of blocks/layers
    embedding_size: int        # vocab embedding dimension
    hidden_size: Optional[int] # FFN hidden size (for VRAM estimation)
    max_context: Optional[int] # model's trained context length
    quantization: str          # file type descriptor
    has_vision: bool           # True if multimodal (mllama, clip, etc.)
    file_path: str             # path to the GGUF file


def _read_gguf_header(model_path: str) -> dict:
    """Read GGUF header and key-value pairs from a GGUF file.

    Uses the ``gguf`` Python library when available, otherwise falls back
    to a minimal manual parser so the tool remains functional in lightweight
    environments.

    Returns:
        dict with GGUF key-value pairs.

    Raises:
        FileNotFoundError: If *model_path* doesn't exist.
        ValueError: If the file is not a valid GGUF.
    """
    p = Path(model_path)

    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # --- Try the ``gguf`` library first ---
    try:
        from gguf import GGUFReader  # noqa: PLC0415
        reader = GGUFReader(str(p))
        return {field.key: field.parts[0] for field in reader.fields}
    except (ImportError, Exception):
        # Catch broad exceptions: the gguf library may fail on minimally-
        # valid GGUF files (e.g., test fixtures) that lack tensors or have
        # non-standard field layouts. Fall back to the manual parser.
        pass

    # --- Fallback: manual GGUF v2/v3 parser ---
    # GGUF format: magic ("GGUF" / "GGJT") + version (uint32 LE) +
    # tensor-count (uint64 LE) + kv-count (uint64 LE) + KV entries + tensors
    #
    # GGUF v3 differs from v2 in several ways:
    #   1) Type IDs are uint8 (1 byte) instead of uint32 (4 bytes).
    #   2) The type ID values themselves are remapped (see _v3_type_map).
    #   3) String lengths are uint32 (4 bytes) instead of uint64 (8 bytes).
    #   4) Array lengths are uint32 (4 bytes) instead of uint64 (8 bytes).
    with open(p, "rb") as f:
        magic = f.read(4)
        if magic not in (b"GGUF", b"GGJT"):
            raise ValueError(
                f"Not a valid GGUF file (magic={magic!r}) at {model_path}"
            )
        import struct
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unexpected GGUF version {version}")

        is_v3 = version >= 3

        # GGUF header: tensor-count (uint64) then kv-count (uint64)
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        kvs = {}
        # Max sane sizes to avoid MemoryError on corrupted/misaligned files.
        _MAX_KEY_LEN = 1 << 20        # 1 MiB for a key name
        _MAX_STR_LEN = 1 << 30        # 1 GiB for a string value
        _MAX_ARRAY_LEN = 1 << 20      # 1 M elements in an array

        # v2 type IDs (uint32 encoded)
        _v2_type_map = {
            0: "<b",    # int8
            1: "<B",    # uint8
            2: "<h",    # int16
            3: "<H",    # uint16
            4: "<i",    # int32
            5: "<I",    # uint32
            6: "<q",    # int64
            7: "<Q",    # uint64
            8: "<f",    # float32
            9: "<d",    # float64
            10: bool,   # bool
            11: str,    # string (length uint64 + UTF-8 bytes)
            12: list,   # array (type uint32 + len uint64 + elements)
        }

        # v3 type IDs (uint8 encoded) — remapped per GGUF spec
        _v3_type_map = {
            0: "<B",    # uint8
            1: "<b",    # int8
            2: "<H",    # uint16
            3: "<h",    # int16
            4: "<I",    # uint32
            5: "<i",    # int32
            6: "<f",    # float32
            7: bool,    # bool
            8: str,     # string (length uint64 + UTF-8 bytes, same as v2)
            9: list,    # array (type uint8 + len uint64 + elements, same as v2)
            10: "<Q",   # uint64
            11: "<q",   # int64
            12: "<d",   # float64
        }

        _type_map = _v3_type_map if is_v3 else _v2_type_map
        _val_type_fmt = "<B" if is_v3 else "<I"
        # v3: type_id(1B) + pad(3B) + str_len(4B) + pad(4B) + data
        # v2: type_id(4B) + str_len(8B) + data
        _str_len_fmt = "<I" if is_v3 else "<Q"
        _arr_len_fmt = "<I" if is_v3 else "<Q"

        # Pre-compute value-type sizes for numeric scalars so we can
        # skip unknown types while keeping the file pointer aligned.
        _numeric_sizes = {tid: struct.calcsize(sz) for tid, sz in _type_map.items() if isinstance(sz, str)}

        for _ in range(kv_count):
            # key length is always uint64
            key_len = struct.unpack("<Q", f.read(8))[0]
            if key_len > _MAX_KEY_LEN:
                raise ValueError(
                    f"GGUF key length {key_len} exceeds {_MAX_KEY_LEN} "
                    f"(file may be corrupted or stream misaligned) at {model_path}"
                )
            key = f.read(key_len).decode("utf-8", errors="replace")
            # value type
            val_type_id = struct.unpack(_val_type_fmt, f.read(1 if is_v3 else 4))[0]
            if is_v3:
                f.read(3)  # skip 3-byte padding after type ID
            val_type = _type_map.get(val_type_id)
            if val_type is None:
                # Unknown type: skip the value bytes to keep the stream
                # aligned.  Heuristic: peek at the next 8 bytes; if they
                # look like a reasonable length-prefixed blob (< 1 MiB),
                # skip it. Otherwise, seek back and accept misalignment.
                pos_before = f.tell()
                peek = f.read(8)
                if len(peek) == 8:
                    peek_val = struct.unpack("<Q", peek)[0]
                    if peek_val <= _MAX_STR_LEN:
                        f.read(peek_val)
                    else:
                        # Not a plausible length prefix — seek back.
                        f.seek(pos_before)
                continue

            if val_type is str:
                str_len = struct.unpack(_str_len_fmt, f.read(4 if is_v3 else 8))[0]
                if is_v3:
                    f.read(4)  # skip 4-byte padding after str_len
                if str_len > _MAX_STR_LEN:
                    raise ValueError(
                        f"GGUF string length {str_len} exceeds {_MAX_STR_LEN} "
                        f"(file may be corrupted) at {model_path}"
                    )
                val = f.read(str_len).decode("utf-8", errors="replace")
            elif val_type is bool:
                val = bool(struct.unpack("<B", f.read(1))[0])
            elif val_type is list:
                arr_type = struct.unpack(_val_type_fmt, f.read(1 if is_v3 else 4))[0]
                if is_v3:
                    f.read(3)  # skip 3-byte padding after arr_type
                arr_len = struct.unpack(_arr_len_fmt, f.read(4 if is_v3 else 8))[0]
                if is_v3:
                    f.read(4)  # skip 4-byte padding after arr_len
                if arr_len > _MAX_ARRAY_LEN:
                    raise ValueError(
                        f"GGUF array length {arr_len} exceeds {_MAX_ARRAY_LEN} "
                        f"(file may be corrupted) at {model_path}"
                    )
                # Read first element as representative value
                elem_type = _type_map.get(arr_type)
                if elem_type in ("<b", "<B", "<h", "<H", "<i", "<I", "<q", "<Q", "<f", "<d"):
                    elem_size = struct.calcsize(elem_type)
                    elem_data = f.read(elem_size * arr_len)
                    vals = list(struct.iter_unpack(elem_type, elem_data))
                    val = vals
                elif elem_type is str:
                    vals = []
                    for __ in range(arr_len):
                        sl = struct.unpack(_str_len_fmt, f.read(4 if is_v3 else 8))[0]
                        if is_v3:
                            f.read(4)  # skip 4-byte padding after str_len
                        if sl > _MAX_STR_LEN:
                            break
                        vals.append(f.read(sl).decode("utf-8", errors="replace"))
                    val = vals
                elif elem_type is bool:
                    val = list(bool(f.read(1)[0]) for __ in range(arr_len))
                else:
                    # Unknown element type: skip the entire array payload.
                    # For numeric types we know the size; for others, skip
                    # best-effort.
                    elem_size = _numeric_sizes.get(arr_type)
                    if elem_size:
                        f.read(elem_size * arr_len)
                    val = []
            else:
                # numeric scalar
                size = struct.calcsize(val_type)
                val = struct.unpack(val_type, f.read(size))[0]

            kvs[key] = val

        return kvs


def get_model_metadata(model_path: str) -> ModelMetadata:
    """Read GGUF headers and return model metadata.

    Raises:
        FileNotFoundError: If *model_path* doesn't exist.
        ValueError: If the file is not a valid GGUF.
    """
    kvs = _read_gguf_header(model_path)

    # Architecture
    architecture = kvs.get("general.architecture", "unknown")
    if isinstance(architecture, bytes):
        architecture = architecture.decode("utf-8", errors="replace")

    arch_prefix = f"{architecture}."

    # Layer count: {arch}.block_count
    layer_count = kvs.get(f"{arch_prefix}block_count")
    if layer_count is not None:
        layer_count = int(layer_count)
    else:
        layer_count = 0

    # Embedding size
    embedding_size = kvs.get(f"{arch_prefix}embedding_length")
    if embedding_size is not None:
        embedding_size = int(embedding_size)
    else:
        embedding_size = 0

    # Hidden / FFN size — try both key variants
    hidden_size = kvs.get(f"{arch_prefix}ffn_hidden_size")
    if hidden_size is None:
        hidden_size = kvs.get(f"{arch_prefix}hidden_size")
    if hidden_size is not None:
        hidden_size = int(hidden_size)

    # Max context length
    max_context = kvs.get(f"{arch_prefix}max_context_length")
    if max_context is None:
        max_context = kvs.get(f"{arch_prefix}context_length")
    if max_context is not None:
        max_context = int(max_context)

    # Quantization type
    file_type_raw = kvs.get("general.file_type", 1)
    quantization = _FILE_TYPE_MAP.get(int(file_type_raw), f"unknown_{file_type_raw}")

    # Vision detection
    has_vision = architecture.lower() in _VISION_ARCHITECTURES

    return ModelMetadata(
        architecture=architecture,
        layer_count=layer_count,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        max_context=max_context,
        quantization=quantization,
        has_vision=has_vision,
        file_path=str(Path(model_path).resolve()),
    )
