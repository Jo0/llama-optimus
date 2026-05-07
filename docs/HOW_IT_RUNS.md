# How llama-optimus Runs — Execution Documentation

This document describes the **intended execution flow** of llama-optimus from installation through optimization, including both optimization modes (Phase 1 and Phase 2).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Entry Points](#4-entry-points)
5. [Phase 1: Flag Optimization (Default Mode)](#5-phase-1-flag-optimization-default-mode)
6. [Phase 2: Context Tuning Mode](#6-phase-2-context-tuning-mode)
7. [Module Responsibilities](#7-module-responsibilities)
8. [Search Space Configuration](#8-search-space-configuration)
9. [Hardware Telemetry](#9-hardware-telemetry)
10. [Testing](#10-testing)
11. [Environment Variables](#11-environment-variables)

---

## 1. Overview

llama-optimus is a Python CLI tool that automatically finds the **best llama.cpp inference flags** for a given model and hardware configuration. It uses **Bayesian optimization** (via Optuna) to maximize tokens/second throughput by searching over parameters like batch size, thread count, GPU layer offload, KV cache quantization, and memory mapping.

The tool operates in **three modes**:

| Mode | Flag | Purpose | Binary Used |
|------|------|---------|-------------|
| **Full** (default) | `--mode full` (or no `--mode`) | End-to-end: Phase 1 → save config → Phase 2 | `llama-bench` + `llama-server` |
| **Phase 1 only** | `--mode phase1-baseline-config` | Optimize inference flags via `llama-bench`, save config, exit | `llama-bench` |
| **Phase 2 only** | `--mode phase2-context` | Find largest context size meeting speed target via `llama-server` | `llama-server` |

In **full** mode, the Phase 1 best config is saved to `best_phase1_config.json` on disk *before* Phase 2 starts, enabling recovery if the AFK run is interrupted. Phase 2 receives the config via programmatic handoff (in-memory dict) rather than re-reading from disk.

In **phase2-context** mode, the config is resolved via a fallback chain:
1. Programmatic handoff from Phase 1 (when called internally)
2. Load from `best_phase1_config.json` on disk
3. Fall back to `SEARCH_SPACE` defaults

---

## 2. Prerequisites

### Required Software

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| llama.cpp | ≥ b3667 (b5706 recommended) | Provides `llama-bench` and `llama-server` binaries |
| GGUF Model | Any | Model file to optimize for |
| nvidia-smi | Optional | VRAM telemetry (NVIDIA GPUs only) |

### Required Binaries

The tool requires access to these llama.cpp binaries:

- **`llama-bench`** — Used in Phase 1 for benchmarking configurations
- **`llama-server`** — Used in Phase 2 for context tuning via HTTP API

On Windows, the tool searches these locations (in order):
1. `<LLAMA_BIN>/llama-bench.exe` / `<LLAMA_BIN>/llama-server.exe`
2. `<LLAMA_BIN>/Release/llama-bench.exe` / `<LLAMA_BIN>/Release/llama-server.exe`

---

## 3. Installation

### Option A: PyPI (Recommended for end users)

```bash
pip install llama-optimus
llama-optimus
```

### Option B: Development Install (Python venv)

```bash
# Clone repository
git clone https://github.com/BrunoArsioli/llama-optimus
cd llama-optimus

# Create virtual environment
python -m venv .venv

# Activate (Windows cmd.exe)
.venv\Scripts\activate
# Activate (bash/zsh/WSL)
source .venv/bin/activate

# Install in editable mode (includes dev dependencies)
pip install -e .
# Or install minimal requirements only
pip install -r requirements.txt
```

### Dependencies

From [`pyproject.toml`](pyproject.toml):

| Dependency | Purpose |
|------------|---------|
| `optuna>=3.0` | Bayesian optimization framework |
| `pandas` | CSV parsing for llama-bench output |
| `gguf>=0.10` | GGUF header parsing (model metadata) |
| `requests` | HTTP client for llama-server API (Phase 2) |

---

## 4. Entry Points

The tool has two entry points:

### 4.1 CLI Entry Point (Primary)

**Entry:** `llama_optimus.cli:main` (registered as `llama-optimus` script in pyproject.toml)

```bash
llama-optimus --llama-bin /path/to/llama.cpp/build/bin --model /path/to/model.gguf
```

### 4.2 Legacy Script Entry Point

**File:** [`optimus.py`](optimus.py:1) — Thin wrapper calling `cli.main()` directly.

```bash
python optimus.py
```

---

## 5. Phase 1: Flag Optimization

Phase 1 is the core optimization engine. It uses `llama-bench` to benchmark configurations and Optuna to find the best flags. It can be run standalone via `--mode phase1-baseline-config` or as part of the full end-to-end flow (`--mode full`).

### 5.1 Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. PATH RESOLUTION                                          │
│     --llama-bin / $LLAMA_BIN / interactive prompt            │
│     --model / $MODEL_PATH / interactive prompt               │
│     Resolve llama-bench path (OS-aware)                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. LOG SETUP                                                │
│     Create logs/ directory                                   │
│     Tee stdout → terminal + logs/llama_optimus_<ts>_<model>.log │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. MODEL METADATA (P0)                                      │
│     Parse GGUF header → architecture, layers, embedding,    │
│     hidden_size, max_context, quantization, has_vision       │
│     apply_model_constraints(metadata) → tighten gpu_layers   │
│     upper bound to actual layer count                        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. NGL ESTIMATION                                           │
│     estimate_max_ngl() — binary search for max GPU layers    │
│     that fit in VRAM (unless --ngl-max provided)             │
│     Updates SEARCH_SPACE['gpu_layers']['high']               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  5. WARMUP (unless --no-warmup)                              │
│     warmup_until_stable() — run llama-bench repeatedly       │
│     until hardware reaches thermal steady-state              │
│     Minimum 4 runs, default 35 runs                          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  6. OPTIMIZATION (3 stages)                                  │
│                                                              │
│  Stage 1: Bayesian search over NUMERICAL params              │
│    → batch, u_batch, threads, gpu_layers, cache_type, mmap   │
│    → TPESampler, n_trials iterations                         │
│                                                              │
│  Stage 2: Grid search over CATEGORICAL params                │
│    → flash_attn, override_tensor (fixed numericals from S1)  │
│    → GridSampler                                             │
│                                                              │
│  Stage 3: Bayesian re-search over NUMERICAL params           │
│    → Best categorical params from S2 fixed                   │
│    → TPESampler, n_trials iterations                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  7. OUTPUT                                                   │
│     Print best configuration                                 │
│     Save best config to best_phase1_config.json              │
│     Return best config as dict (for programmatic handoff)    │
│     Print ready-to-use llama-server command                  │
│     Run llama-bench with optimized flags (comparison)        │
│     Run llama-bench with default flags (baseline)            │
└─────────────────────────────────────────────────────────────┐
```

### 5.1a Config Persistence

After Stage 3 completes, `run_optimization()` performs two actions:
1. **Saves** the best configuration to `best_phase1_config.json` on disk (enables recovery if AFK run is interrupted)
2. **Returns** the best configuration as a `dict` (enables programmatic handoff to Phase 2 in full mode)

### 5.2 Stage Details

#### Stage 1 — Numerical Parameter Search

**Function:** [`objective_1()`](src/llama_optimus/core.py:153)

Samples and benchmarks these parameters:
- `batch` — Batch size (constrained by context if provided)
- `u_batch` — Ubatch size
- `threads` — CPU thread count
- `gpu_layers` — GPU layer offload count (`-ngl`)
- `cache_type` — KV cache quantization (`f16`, `q8_0`, `q4_0`)
- `mmap` — Memory mapping toggle (`0` or `1`)

**Sampler:** `TPESampler(multivariate=True)` — Bayesian, learns correlations between parameters.

#### Stage 2 — Categorical Parameter Grid Search

**Function:** [`objective_2()`](src/llama_optimus/core.py:233)

Fixes the best numerical parameters from Stage 1 and grid-searches:
- `flash_attn` — Flash attention (`0` or `1`)
- `override_tensor` — Tensor offload patterns (from [`OVERRIDE_PATTERNS`](src/llama_optimus/override_patterns.py:24))

Only runs override_tensor scan when `--override-mode scan` is set.

**Sampler:** `GridSampler` — Exhaustive over categorical space.

#### Stage 3 — Final Numerical Refinement

**Function:** [`objective_3()`](src/llama_optimus/core.py:313)

Fixes the best categorical parameters from Stage 2 and re-optimizes numerical parameters with the full Bayesian sampler.

### 5.3 Telemetry Wrapper

Every objective function is wrapped by [`_run_with_telemetry()`](src/llama_optimus/core.py:19):

1. Capture pre-benchmark GPU snapshot (`nvidia-smi`)
2. Run the objective (llama-bench)
3. Capture post-benchmark GPU snapshot
4. Apply VRAM penalization if headroom is below threshold
5. Record telemetry as trial user attributes

### 5.4 VRAM Penalization

[`penalize_vram_heavy()`](src/llama_optimus/hardware_probe.py:105) applies multipliers to the throughput score:

| VRAM Headroom | Multiplier | Reason |
|---------------|------------|--------|
| ≥ 13% | 1.0x | Optimal |
| 10–12% | 0.9x | Warning (threshold - 2%) |
| < 10% | 0.7x | Paging likely |

---

## 6. Phase 2: Context Tuning Mode

Phase 2 finds the **largest context window** that meets a minimum speed target, using `llama-server` and its HTTP API.

### 6.1 Invocation

Phase 2 can be invoked in two ways:

**Standalone (after a previous Phase 1 run):**
```bash
llama-optimus --mode phase2-context \
  --contexts 8192,16384,32768,65536,131072 \
  --cache-types f16,q8_0,q4_0 \
  --min-speed 10.0
```

**As part of full end-to-end mode (default):**
```bash
llama-optimus --mode full   # or simply: llama-optimus
```

### 6.2 Config Resolution

Phase 2 resolves its base configuration through a **three-level fallback chain**:

1. **Programmatic handoff** — When called from `--mode full`, the best config dict is passed directly from `run_optimization()` return value
2. **JSON file load** — When called via `--mode phase2-context`, load from `best_phase1_config.json` (or path specified by `--phase1-config`)
3. **SEARCH_SPACE defaults** — If no file exists, fall back to search-space default values

This ensures Phase 2 is always runnable, even without a prior Phase 1 run.

### 6.3 Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Resolve llama-server path (same OS-aware logic)         │
│  2. Parse --contexts and --cache-types into lists           │
│  3. Resolve config via fallback chain:                       │
│     a. Programmatic handoff (from Phase 1, in full mode)    │
│     b. Load from best_phase1_config.json on disk            │
│     c. Fall back to SEARCH_SPACE defaults                   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Grid search: for each (context_size, cache_type):       │
│                                                              │
│     a. Pre-trial GPU telemetry snapshot                      │
│     b. Build llama-server command via ServerFlagBuilder      │
│     c. Start llama-server subprocess                         │
│     d. Poll /health endpoint until ready (60s timeout)       │
│     e. POST /completion → measure tokens/second              │
│     f. Stop llama-server (graceful SIGTERM/SIGBREAK)        │
│     g. Post-trial GPU telemetry snapshot                     │
│     h. Apply VRAM penalization if headroom < threshold       │
│     i. Record ContextTrialResult                             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Output results table with PASS/FAIL indicators          │
│     Highlight best passing configuration                     │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `ServerFlagBuilder` | [`context_tuning.py:33`](src/llama_optimus/context_tuning.py:33) | Translate internal params to llama-server flags |
| `ServerLifecycle` | [`context_tuning.py:90`](src/llama_optimus/context_tuning.py:90) | Start, benchmark, stop llama-server |
| `ContextTrialResult` | [`context_tuning.py:20`](src/llama_optimus/context_tuning.py:20) | Dataclass holding trial results |

### 6.5 Flag Translation

`llama-server` uses different flag syntax than `llama-bench`:

| Parameter | llama-bench flag | llama-server flag |
|-----------|------------------|-------------------|
| mmap enabled | (default) | `--mmap` |
| mmap disabled | `-mmp 0` | `--no-mmap` |
| flash-attn on | `--flash-attn 1` | `--flash-attn on` |
| flash-attn off | `--flash-attn 0` | `--flash-attn off` |

---

## 7. Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| **CLI** | [`cli.py`](src/llama_optimus/cli.py) | Argument parsing, path resolution, log setup, mode dispatch (3 modes) |
| **Core** | [`core.py`](src/llama_optimus/core.py) | Objective functions, ngl estimation, warmup, optimization loop (returns best config dict) |
| **Search Space** | [`search_space.py`](src/llama_optimus/search_space.py) | Parameter bounds, model constraints, context-aware batch limits |
| **Model Info** | [`model_info.py`](src/llama_optimus/model_info.py) | GGUF header parsing, metadata extraction |
| **Hardware Probe** | [`hardware_probe.py`](src/llama_optimus/hardware_probe.py) | nvidia-smi telemetry, VRAM penalization |
| **Override Patterns** | [`override_patterns.py`](src/llama_optimus/override_patterns.py) | Preset `--override-tensor` regex patterns for MoE models |
| **Context Tuning** | [`context_tuning.py`](src/llama_optimus/context_tuning.py) | llama-server lifecycle, API benchmarking, Phase 2 grid search |

---

## 8. Search Space Configuration

Defined in [`search_space.py`](src/llama_optimus/search_space.py:10):

| Parameter | Low | High | Type | Description |
|-----------|-----|------|------|-------------|
| `batch_size` | 8 | 16384 | int | Batch size (constrained by context if provided) |
| `ubatch_size` | 4 | 8192 | int | Ubatch size |
| `threads` | 1 | `os.cpu_count()` | int | CPU thread count |
| `gpu_layers` | 0 | 149 (dynamic) | int | GPU layer offload (`-ngl`) |
| `flash_attn` | — | `[0, 1]` | categorical | Flash attention toggle |
| `cache_type` | — | `['f16', 'q8_0', 'q4_0']` | categorical | KV cache quantization |
| `mmap` | — | `[0, 1]` | categorical | Memory mapping toggle |

### Dynamic Constraints

1. **Layer count constraint** — `gpu_layers.high` is clamped to actual model layer count via `apply_model_constraints()`
2. **Context-aware batch constraint** — `batch.high` is clamped to `context_size // 8` via `get_context_aware_batch_high()`

---

## 9. Hardware Telemetry

### NVIDIA GPU Telemetry

[`get_gpu_telemetry()`](src/llama_optimus/hardware_probe.py:46) runs:
```bash
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu \
  --format=csv,noheader --id=0
```

Returns `GpuTelemetrySnapshot` with:
- `vram_used_mb` — VRAM currently in use
- `vram_total_mb` — Total VRAM
- `temperature_c` — GPU temperature
- `gpu_utilization_pct` — GPU utilization percentage
- `vram_headroom_pct` — Computed: `(1 - used/total) * 100`

### Disabling Telemetry

Use `--no-telemetry` flag when:
- Running on non-NVIDIA GPU (AMD, Intel, Apple Silicon)
- Running in CI/headless environments
- `nvidia-smi` is not installed

---

## 10. Testing

Tests are located in [`test/`](test/) and use `pytest`:

```bash
# Activate venv
.venv\Scripts\activate

# Run all tests
pytest

# Run specific test file
pytest test/test_core.py

# Run with verbose output
pytest -v
```

### Test Coverage

| Test File | Module | Key Areas |
|-----------|--------|-----------|
| [`test_core.py`](test/test_core.py) | `core.py` | CSV parsing, command construction, objective functions, context constraints |
| [`test_context_tuning.py`](test/test_context_tuning.py) | `context_tuning.py` | Server flag builder, lifecycle, benchmark, results table |
| [`test_model_info.py`](test/test_model_info.py) | `model_info.py` | GGUF parsing, metadata extraction, constraint application |
| [`test_hardware_probe.py`](test/test_hardware_probe.py) | `hardware_probe.py` | VRAM headroom, penalization, nvidia-smi parsing |
| [`test_search_space.py`](test/test_search_space.py) | `search_space.py` | Context-aware batch constraints |

Tests use `unittest.mock` to mock subprocess calls, Optuna trials, and HTTP requests — no real llama.cpp binary required.

---

## 11. Environment Variables

| Variable | CLI Flag | Purpose |
|----------|----------|---------|
| `LLAMA_BIN` | `--llama-bin` | Path to llama.cpp `build/bin` directory |
| `MODEL_PATH` | `--model` | Path to `.gguf` model file |

Priority: CLI flag > environment variable > interactive prompt.

---

## Quick Reference: Complete Run Example

```bash
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Set paths (or use CLI flags)
set LLAMA_BIN=C:\path\to\llama.cpp\build\bin
set MODEL_PATH=C:\path\to\model.gguf

# 3. Full end-to-end optimization (default — Phase 1 + Phase 2)
llama-optimus --trials 45 --repeat 3 --metric tg

# 3a. Explicit full mode (same as above)
llama-optimus --mode full --trials 45 --repeat 3 --metric tg

# 4. Phase 1 only (save config and exit)
llama-optimus --mode phase1-baseline-config --trials 45 --repeat 3 --metric tg

# 5. Phase 2 only (uses saved config or SEARCH_SPACE defaults)
llama-optimus --mode phase2-context --contexts 32768,65536,131072 --min-speed 15.0

# 6. Quick test (minimal)
llama-optimus --trials 1 -r 1 --no-warmup --n-tokens 20 --metric tg
```

---

## Architecture Diagram

```
                    ┌──────────────────────┐
                    │   cli.py (main)      │
                    │  - Parse args        │
                    │  - Resolve paths     │
                    │  - Setup logging     │
                    │  - Mode dispatch     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  model_info.py        │
                    │  - Parse GGUF header  │
                    │  - Extract metadata   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  search_space.py      │
                    │  - apply_model_       │
                    │    constraints()      │
                    │  - Tighten bounds     │
                    └──────────┬───────────┘
                               │
               ┌────────────────┼────────────────┐
               │                │                │
      ┌────────▼───────┐ ┌─────▼──────┐ ┌───────▼──────────┐
      │ estimate_max_  │ │ warmup_    │ │ run_optimization │
      │ ngl()          │ │ until_     │ │ (3 stages)       │
      │                │ │ stable()   │ │ → returns dict   │
      └────────────────┘ └────────────┘ └────────┬─────────┘
                                                  │
                               ┌──────────────────┼──────────────────┐
                               │                  │                  │
                      ┌────────▼───────┐ ┌────────▼──────┐ ┌────────▼───────┐
                      │ objective_1()  │ │ objective_2() │ │ objective_3()  │
                      │ (numerical)    │ │ (categorical) │ │ (numerical)    │
                      └───────┬────────┘ └───────┬───────┘ └───────┬────────┘
                              │                  │                  │
                      ┌───────▼──────────────────▼──────────────────▼───────┐
                      │         _run_with_telemetry()                       │
                      │  - Pre nvidia-smi snapshot                          │
                      │  - Run llama-bench via run_llama_bench_with_csv()   │
                      │  - Post nvidia-smi snapshot                         │
                      │  - penalize_vram_heavy()                            │
                      └─────────────────────────────────────────────────────┘

Mode Dispatch (cli.py):

  --mode full (default)
    ┌─────────────────────────────────────────────────────────┐
    │ Phase 1: run_optimization()                             │
    │   → saves best_phase1_config.json to disk               │
    │   → returns best_config dict                            │
    │ Phase 2: _run_phase2_context(best_config=best_config)   │
    │   → uses programmatic handoff                           │
    └─────────────────────────────────────────────────────────┘

  --mode phase1-baseline-config
    ┌─────────────────────────────────────────────────────────┐
    │ Phase 1: run_optimization()                             │
    │   → saves best_phase1_config.json to disk               │
    │   → exits                                               │
    └─────────────────────────────────────────────────────────┘

  --mode phase2-context
    ┌─────────────────────────────────────────────────────────┐
    │ Phase 2: _run_phase2_context()                          │
    │   → config from: handoff → JSON file → SEARCH_SPACE     │
    └─────────────────────────────────────────────────────────┘
```
