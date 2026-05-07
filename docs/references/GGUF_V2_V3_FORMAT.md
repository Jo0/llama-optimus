# GGUF v2 vs v3 Binary Format Reference

Derived from binary analysis of a real GGUF v3 model file
(`Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf`).

## Shared Header (v2 and v3)

| Offset | Field          | Type     | Endian | Notes                        |
|--------|----------------|----------|--------|------------------------------|
| 0      | magic          | 4 bytes  | —      | `"GGUF"` or `"GGJT"`         |
| 4      | version        | uint32   | LE     | `2` or `3`                   |
| 8      | tensor_count   | uint64   | LE     | Number of tensor entries     |
| 16     | kv_count       | uint64   | LE     | Number of key-value entries  |

## KV Entry Layout

### v2

| Field       | Type   | Endian | Size |
|-------------|--------|--------|------|
| key_len     | uint64 | LE     | 8    |
| key         | bytes  | —      | key_len |
| type_id     | uint32 | LE     | 4    |
| value       | varies | —      | varies |

### v3

| Field       | Type   | Endian | Size | Notes                              |
|-------------|--------|--------|------|------------------------------------|
| key_len     | uint64 | LE     | 8    | Same as v2                         |
| key         | bytes  | —      | key_len | Same as v2                       |
| type_id     | uint8  | —      | 1    | **Changed from uint32**            |
| _pad        | bytes  | —      | 3    | **Zero padding to 8-byte align**   |
| value       | varies | —      | varies | Layout differs (see below)       |

## Value Layout Differences

### Numeric Scalars (int, uint, float, bool)

| Version | Layout                              |
|---------|-------------------------------------|
| v2      | type_id(4B) + value                 |
| v3      | type_id(1B) + pad(3B) + value       |

### Strings

| Version | Layout                                      |
|---------|---------------------------------------------|
| v2      | type_id(4B) + str_len(uint64 LE) + data     |
| v3      | type_id(1B) + pad(3B) + str_len(uint32 LE) + pad(4B) + data |

### Arrays

| Version | Layout                                                        |
|---------|---------------------------------------------------------------|
| v2      | type_id(4B) + arr_type(uint32 LE) + arr_len(uint64 LE) + elements |
| v3      | type_id(1B) + pad(3B) + arr_type(uint8) + pad(3B) + arr_len(uint32 LE) + pad(4B) + elements |

## Type ID Values

### v2 Type IDs (uint32 encoded)

| ID | Type       | Format |
|----|------------|--------|
| 0  | INT8       | `<b`   |
| 1  | UINT8      | `<B`   |
| 2  | INT16      | `<h`   |
| 3  | UINT16     | `<H`   |
| 4  | INT32      | `<i`   |
| 5  | UINT32     | `<I`   |
| 6  | INT64      | `<q`   |
| 7  | UINT64     | `<Q`   |
| 8  | FLOAT32    | `<f`   |
| 9  | FLOAT64    | `<d`   |
| 10 | BOOL       | 1B     |
| 11 | STRING     | len+data |
| 12 | ARRAY      | type+len+elems |

### v3 Type IDs (uint8 encoded, remapped)

| ID | Type       | Format |
|----|------------|--------|
| 0  | UINT8      | `<B`   |
| 1  | INT8       | `<b`   |
| 2  | UINT16     | `<H`   |
| 3  | INT16      | `<h`   |
| 4  | UINT32     | `<I`   |
| 5  | INT32      | `<i`   |
| 6  | FLOAT32    | `<f`   |
| 7  | BOOL       | 1B     |
| 8  | STRING     | len+data |
| 9  | ARRAY      | type+len+elems |
| 10 | UINT64     | `<Q`   |
| 11 | INT64      | `<q`   |
| 12 | FLOAT64    | `<d`   |

## Summary of v3 Changes

1. **Type ID encoding**: `uint32` (4B) → `uint8` (1B)
2. **Type ID values**: Remapped (e.g., STRING: 11→8, ARRAY: 12→9, UINT32: 5→4)
3. **String length**: `uint64` (8B) → `uint32` (4B) + 4B zero padding
4. **Array length**: `uint64` (8B) → `uint32` (4B) + 4B zero padding
5. **Array element type**: `uint32` (4B) → `uint8` (1B) + 3B zero padding
6. **Alignment**: All fields padded to 8-byte boundaries with zero bytes
7. **Endianness**: All fields remain **little-endian** (header, key_len, str_len, arr_len, values)

## Verified Against

- File: `Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf` (GGUF V3, 15.69 GiB)
- All 42 KV pairs parsed correctly
- All 851 tensors accounted for in header
