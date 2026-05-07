# llama_optimus/cli.py
# handle parsing, validation, and env setup

import argparse, json, os, sys
import platform
import logging
import re
from datetime import datetime
from pathlib import Path
from .core import run_optimization, estimate_max_ngl, warmup_until_stable
from .override_patterns import OVERRIDE_PATTERNS
from .search_space import SEARCH_SPACE, max_threads, apply_model_constraints, apply_hardware_batch_constraints
from .model_info import get_model_metadata
from .context_tuning import run_context_optimization, format_results_table

from llama_optimus import __version__


def _normalize_path(path_str: str) -> str:
    """Normalize a path string to use the OS-native separator.

    Handles mixed separators (e.g. ``C:\\Users\\foo/bar`` on Windows)
    so that downstream ``Path`` operations and subprocess calls are
    consistent.
    """
    # Replace all backslashes with forward slashes, then let Path
    # resolve to the native separator.  Path.on Windows understands
    # both, but normalizing early avoids hybrid paths like
    # ``C:\Users\foo/bar``.
    unified = path_str.replace("\\", "/")
    return str(Path(unified))


class _Tee:
    """Write to multiple streams at once (terminal + log file)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser(
        description="llama-optimus: Benchmark & tune llama.cpp.",
        epilog="""
        Example usage:

            llama-optimus --llama-bin my_path_to/llama.cpp/build/bin --model my_path_to/models/my-model.gguf --trials 35 --metric tg
            
        for a quick test (set a single Optuna trial and a single repetition of llama-bench):
            
            llama-optimus --llama-bin my_path_to/llama.cpp/build/bin --model my_path_to/models/my-model.gguf --trials 1 -r 1 --metric tg
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument("--trials", type=int, default=45, help="Number of Optuna/optimization trials")
    parser.add_argument("--model", type=str, help="Path to model (overrides env var)")
    parser.add_argument("--llama-bin", type=str, help="Path to llama.cpp build/bin folder (overrides env var)")

    parser.add_argument("--metric", type=str, default="tg", choices=["tg", "pp", "mean"], help="Which throughput " \
        "metric to optimize: 'tg' (token generation, default), 'pp' (prompt processing), or 'mean' (average of both)")

    parser.add_argument("--ngl-max",type=int, help="Maximum number of model layers for -ngl "
        "(skip estimation if provided; estimation runs by default).")

    parser.add_argument("--repeat", "-r", type=int, default=3, help="Number of llama-bench runs per configuration "
        "(higher = more robust, lower = faster; for quick assessment: 1)")

    parser.add_argument("--n-tokens", type=int, default=192, help="Number of tokens used in llama-bench to test " \
        "velocity of prompt processing and text generation. Keep in mind there is large variability in tok/s outputs. " \
        "If n_tokens is too low, uncertainty takes over, optimization may suffer. Still, if you need to lower it, " \
        "try to operate with n_tokens > 70 and --repeat 3. " \
        "For fast exploration/testing/debug: --n-tokens 10 --repeat 2 is fine")
    
    parser.add_argument("--n-warmup-tokens", "-nwt", type=int, default=128, help="Number of tokens passed to " \
        "llama-bench during each warmup loop. In case of large models (and you getting small tg tokens/s), "
        "if n_warmup_tokens is too large, it can happen that you warmup in the first warmup cycle, and you end " \
        "up not detecting the warmup. ")
    
    parser.add_argument("--n-warmup-runs", type=int, default=35, help="Maximum warm-up iterations before trials " \
    "begin. To skip warm-up completely, use the --no-warmup flag; Otherwise, there will be a minimum " \
    "number of warmup runs, which is set with `min_runs=3` in core function definition")

    parser.add_argument("--no-warmup", action="store_true", help="Skip the initial system warmup phase before " \
    "optimization (for debugging/testing).")

    #parser.add_argument('--version', "-v", action='version', version='llama-optimus v0.1.0')
    parser.add_argument("--version", "-v", action='version', version=f'llama-optimus v{__version__}')

    parser.add_argument("--override-mode", type=str, default="scan", choices=["none", "scan", "custom"],
    help=f"'none': do not scan this parameter; scan: 'scan' over preset override-tensor patterns; " \
    f"'custom': (future) user provides their own pattern(s). Available override patterns: {OVERRIDE_PATTERNS.keys()}" )

    parser.add_argument("--vram-headroom", type=float, default=0.12,
        help="Minimum VRAM headroom as fraction (default 0.12 = 12%%). "
             "Configs below this threshold are penalized.")

    parser.add_argument("--no-telemetry", action="store_true",
        help="Disable nvidia-smi telemetry (useful for non-NVIDIA GPUs or CI).")
    
    # --- Mode selection ---
    mode_group = parser.add_argument_group("optimization mode")
    mode_group.add_argument("--mode", type=str, default="full", choices=["full", "phase1-baseline-config", "phase2-context"],
        help="Optimization mode: 'full' (default, end-to-end: Phase 1 + Phase 2), "
             "'phase1-baseline-config' (run Phase 1 only and save config), "
             "'phase2-context' (run Phase 2 only, uses saved config or search-space defaults)")

    # --- Phase 2: Context tuning arguments ---
    ctx_group = parser.add_argument_group("phase 2 — context tuning")
    ctx_group.add_argument("--contexts", type=str, default="8192,16384,32768,65536,131072",
        help="Comma-separated list of context sizes to test (default: 8192,16384,32768,65536,131072)")
    ctx_group.add_argument("--cache-types", type=str, default="f16,q8_0,q4_0",
        help="Comma-separated list of KV cache types to test (default: f16,q8_0,q4_0)")
    ctx_group.add_argument("--min-speed", type=float, default=10.0,
        help="Minimum tokens/second threshold for context trials (default: 10.0)")
    ctx_group.add_argument("--phase1-config", type=str, default="best_phase1_config.json",
        help="Path to saved Phase 1 config JSON file (default: best_phase1_config.json)")

    args = parser.parse_args()

    # Set paths based on CLI flags, env vars, or prompt user to provide it
    # Resolve llama_bin_path — normalize separators to avoid hybrid paths
    llama_bin_path = _normalize_path(
        args.llama_bin or os.environ.get("LLAMA_BIN", "")
        or input("Please, provide the path to your 'llama.cpp/build/bin' ").strip()
    )

    # Check the operating system and build llama_bench_path
    bin_dir = Path(llama_bin_path)
    if platform.system() == "Windows":
        # Check CMake build Release/ directory first, then flat directory for prebuilt binaries
        llama_bench_path = str(bin_dir / "llama-bench.exe")
        if not Path(llama_bench_path).is_file():
            llama_bench_path = str(bin_dir / "Release" / "llama-bench.exe")
        if not Path(llama_bench_path).is_file():
            sys.exit(
                f"ERROR: llama-bench.exe not found.\n"
                f"  Searched:\n"
                f"    {bin_dir / 'llama-bench.exe'}\n"
                f"    {bin_dir / 'Release' / 'llama-bench.exe'}"
            )
    else:
        llama_bench_path = str(bin_dir / "llama-bench")
        # Sanity-check
        if not Path(llama_bench_path).is_file():
            sys.exit(f"ERROR: llama-bench not found at {llama_bench_path}")


    # Resolve model_path — normalize separators
    model_path = _normalize_path(
        args.model or os.environ.get("MODEL_PATH", "")
        or input("Please, provide the path to your 'ai_model.gguf' ").strip()
    )

    # Quick check if paths are set. ERROR msg if None or empty.
    if not llama_bin_path or not model_path:
        print("ERROR: LLAMA_BIN or MODEL_PATH not set. Set via environment variable, " \
        "pass via CLI flags, or provide paths just after launching llama-optimus. " \
        "Go to your terminal, navigate to your_path_to/llama.cpp/buil/bin and type 'pwd' to resolve the entire path. " \
        "Go to your terminal, navigate to your_path_to_AI_models/ and type 'pwd' to resolve the path. " \
        "Note: you must pass /path_to_model/model_name.gguf; e.g. your_path_model/gemma3_12B.gguf .", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(llama_bench_path):
        print(f"ERROR: llama-bench not found at {llama_bench_path}. ...", file=sys.stderr)
        sys.exit(1)

    # Set up log file: logs/<timestamp>_<model_stem>.log
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    model_stem = Path(model_path).stem[:60]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"llama_optimus_{timestamp}_{model_stem}.log"
    _log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    # Tee stdout so all print() calls go to both terminal and log file
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    # Add a file handler to the root logger so Optuna's trial logs are also captured
    _file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(_file_handler)

    print(f"Logging to: {log_path}")
    print("")
    print("#################")
    print("# LLAMA-OPTIMUS #")
    print("#################")

    # --- Parse GGUF model metadata (P0-1) ---
    metadata = get_model_metadata(model_path)
    print("")
    print("###############")
    print("# Model Info  #")
    print("###############")
    print(f"Architecture : {metadata.architecture}")
    print(f"Layers       : {metadata.layer_count}")
    print(f"Embedding    : {metadata.embedding_size}")
    print(f"Hidden size  : {metadata.hidden_size}")
    print(f"Max context  : {metadata.max_context or 'N/A'}")
    print(f"Quantization : {metadata.quantization}")
    print(f"Vision       : {'Yes' if metadata.has_vision else 'No'}")
    print(f"File         : {metadata.file_path}")
    print("")

    # Apply model constraints to tighten search-space bounds
    apply_model_constraints(metadata)
    if metadata.layer_count > 0:
        print(f"Model has {metadata.layer_count} layers — search-space gpu_layers "
              f"upper bound set to {SEARCH_SPACE['gpu_layers']['high']}.")
        print("")

    # Apply hardware-aware batch constraints (clamps batch/ubatch to VRAM budget)
    apply_hardware_batch_constraints(metadata)
    print(f"Batch search bounds: [{SEARCH_SPACE['batch_size']['low']}, {SEARCH_SPACE['batch_size']['high']}]")
    print(f"Ubatch search bounds: [{SEARCH_SPACE['ubatch_size']['low']}, {SEARCH_SPACE['ubatch_size']['high']}]")
    print("")

    print(f"Number of CPUs: {max_threads}.")
    print(f"Path to 'llama-bench':{llama_bench_path}")  # in llama.cpp/tools/
    print(f"Path to 'model.gguf' file:{model_path}")
    print("")

    # default: estimate maximum number of layers before run_optimization 
    # in case the user knows ngl_max value, skip ngl_max estimate
    if args.ngl_max is not None: 
        SEARCH_SPACE['gpu_layers']['high'] = args.ngl_max
        print("")
        print(f"User-specified maximum -ngl set to {args.ngl_max}")
        print("")
    else:
        print("")
        print("########################################################################")
        print("# Find maximum number of model layers that can be written to your VRAM #")
        print("########################################################################")
        print("")

        SEARCH_SPACE['gpu_layers']['high'] = estimate_max_ngl(
            llama_bench_path=llama_bench_path, model_path=model_path, 
            min_ngl=0, max_ngl=SEARCH_SPACE['gpu_layers']['high'])
        print("")
        print(f"Setting maximum -ngl to {SEARCH_SPACE['gpu_layers']['high']}")
        print("")

    # system warm-up before optimization
    max_ngl_wup=SEARCH_SPACE['gpu_layers']['high']
    
    if args.no_warmup:
        print("")
        print("#####################################################")
        print("# !!!Optimization running without system warmup!!!  #")
        print("#####################################################")
        print("")
    else: 
        print("")
        print("#######################")
        print("# Starting warmup...  #")
        print("#######################")
        print("")

        # in case n_warmup_runs is set to < 4, warn about the minimum number of warmup runs
        if args.n_warmup_runs < 4:
            print("")
            print("#########################################################################")
            print("# Setting a minimum of 4 warmup runs.                                   #")
            print('# For no warmup, pass the --no-warmup flag during llama-optimus launch  #')
            print("#########################################################################")
            print("")

        # launch warmup
        warmup_until_stable(llama_bench_path=llama_bench_path, model_path=model_path, metric=args.metric, 
                            ngl=max_ngl_wup, min_runs=4, n_warmup_runs=args.n_warmup_runs,
                            n_warmup_tokens=args.n_warmup_tokens, max_threads=max_threads)

    # --- Mode dispatch ---
    if args.mode == "phase2-context":
        # Phase 2 only — load config from file OR use search-space defaults
        _run_phase2_context(
            llama_bin_path=llama_bin_path,
            model_path=model_path,
            metadata=metadata,
            contexts_str=args.contexts,
            cache_types_str=args.cache_types,
            min_speed=args.min_speed,
            vram_headroom=args.vram_headroom,
            no_telemetry=args.no_telemetry,
            phase1_config_path=args.phase1_config,
        )
    elif args.mode == "phase1-baseline-config":
        # Phase 1 only — run Optuna, save config, exit
        print("")
        print("##################################")
        print("# Starting Optimization Loop...  #")
        print("##################################")
        print("")

        run_optimization(
            n_trials=args.trials,
            n_tokens=args.n_tokens,
            metric=args.metric,
            repeat=args.repeat,
            llama_bench_path=llama_bench_path,
            model_path=model_path,
            llama_bin_path=llama_bin_path,
            override_mode=args.override_mode,
            vram_headroom_threshold=args.vram_headroom,
            no_telemetry=args.no_telemetry,
        )
    elif args.mode == "full":
        # Full end-to-end AFK run: Phase 1 → Phase 2
        print("")
        print("##################################")
        print("# Starting Optimization Loop...  #")
        print("##################################")
        print("")

        # Phase 1: Optuna-based flag tuning (saves best_phase1_config.json for recovery)
        best_config = run_optimization(
            n_trials=args.trials,
            n_tokens=args.n_tokens,
            metric=args.metric,
            repeat=args.repeat,
            llama_bench_path=llama_bench_path,
            model_path=model_path,
            llama_bin_path=llama_bin_path,
            override_mode=args.override_mode,
            vram_headroom_threshold=args.vram_headroom,
            no_telemetry=args.no_telemetry,
        )

        # Phase 2: Context optimization using best config from Phase 1
        _run_phase2_context(
            llama_bin_path=llama_bin_path,
            model_path=model_path,
            metadata=metadata,
            contexts_str=args.contexts,
            cache_types_str=args.cache_types,
            min_speed=args.min_speed,
            vram_headroom=args.vram_headroom,
            no_telemetry=args.no_telemetry,
            phase1_config_path=args.phase1_config,
            best_config=best_config,
        )


def _run_phase2_context(
    llama_bin_path: str,
    model_path: str,
    metadata,
    contexts_str: str,
    cache_types_str: str,
    min_speed: float,
    vram_headroom: float,
    no_telemetry: bool,
    phase1_config_path: str = "best_phase1_config.json",
    best_config: dict = None,
) -> None:
    """Phase 2 context tuning mode: grid search over context sizes and cache types.

    Parameters:
        best_config: If provided, use this config directly (programmatic handoff from Phase 1).
                     If None, load from phase1_config_path JSON file or fall back to SEARCH_SPACE defaults.
    """
    from pathlib import Path

    # Resolve llama-server path — use Path for consistent separators
    bin_dir2 = Path(llama_bin_path)
    if platform.system() == "Windows":
        llama_server_path = str(bin_dir2 / "llama-server.exe")
        if not Path(llama_server_path).is_file():
            llama_server_path = str(bin_dir2 / "Release" / "llama-server.exe")
        if not Path(llama_server_path).is_file():
            sys.exit(
                f"ERROR: llama-server.exe not found.\n"
                f"  Searched:\n"
                f"    {bin_dir2 / 'llama-server.exe'}\n"
                f"    {bin_dir2 / 'Release' / 'llama-server.exe'}"
            )
    else:
        llama_server_path = str(bin_dir2 / "llama-server")
        if not Path(llama_server_path).is_file():
            sys.exit(f"ERROR: llama-server not found at {llama_server_path}")

    # Parse contexts and cache types
    contexts = [int(c.strip()) for c in contexts_str.split(",")]
    cache_types = [ct.strip() for ct in cache_types_str.split(",")]

    print("")
    print("####################################")
    print("# Phase 2: Context Optimization    #")
    print("####################################")
    print("")
    print(f"llama-server : {llama_server_path}")
    print(f"Contexts     : {contexts}")
    print(f"Cache types  : {cache_types}")
    print(f"Min speed    : {min_speed} tokens/s")
    print("")

    # Determine best_config: passed directly, loaded from file, or SEARCH_SPACE defaults
    if best_config is not None:
        # Programmatic handoff from Phase 1 (full mode)
        override_key = best_config.get("override_tensor", "none")
        if override_key != "none" and override_key in OVERRIDE_PATTERNS:
            override_pattern = OVERRIDE_PATTERNS[override_key]
        else:
            override_pattern = None
        phase1_config = {
            "gpu_layers": best_config.get("gpu_layers", SEARCH_SPACE["gpu_layers"]["high"]),
            "batch": best_config.get("batch", SEARCH_SPACE.get("batch", {}).get("high", 2048)),
            "threads": best_config.get("threads", max_threads),
            "mmap": best_config.get("mmap", 1),
            "flash_attn": best_config.get("flash_attn", 0),
            "override_tensor": override_pattern,
        }
        print(f"Using Phase 1 config passed programmatically")
        print(f"  Best tg tokens/sec: {best_config.get('best_value', 'N/A')}")
        print(f"  gpu_layers: {phase1_config['gpu_layers']}, batch: {phase1_config['batch']}, "
              f"threads: {phase1_config['threads']}")
        if override_key != "none":
            print(f"  override_tensor: {override_key} -> {override_pattern}")
    else:
        # Load from file OR fall back to SEARCH_SPACE defaults
        config_file = Path(phase1_config_path)
        if config_file.is_file():
            saved_config = json.loads(config_file.read_text())
            override_key = saved_config.get("override_tensor", "none")
            # Resolve override_tensor preset name to its actual regex pattern
            if override_key != "none" and override_key in OVERRIDE_PATTERNS:
                override_pattern = OVERRIDE_PATTERNS[override_key]
            else:
                override_pattern = None
            phase1_config = {
                "gpu_layers": saved_config.get("gpu_layers", SEARCH_SPACE["gpu_layers"]["high"]),
                "batch": saved_config.get("batch", SEARCH_SPACE.get("batch", {}).get("high", 2048)),
                "threads": saved_config.get("threads", max_threads),
                "mmap": saved_config.get("mmap", 1),
                "flash_attn": saved_config.get("flash_attn", 0),
                "override_tensor": override_pattern,
            }
            print(f"Loaded Phase 1 config from {config_file}")
            print(f"  Best tg tokens/sec: {saved_config.get('best_value', 'N/A')}")
            print(f"  gpu_layers: {phase1_config['gpu_layers']}, batch: {phase1_config['batch']}, "
                  f"threads: {phase1_config['threads']}")
            if override_key != "none":
                print(f"  override_tensor: {override_key} -> {override_pattern}")
        else:
            phase1_config = {
                "gpu_layers": SEARCH_SPACE["gpu_layers"]["high"],
                "batch": SEARCH_SPACE.get("batch", {}).get("high", 2048),
                "threads": max_threads,
                "mmap": 1,
                "flash_attn": 0,
                "override_tensor": None,
            }
            print(f"No saved Phase 1 config found at {config_file} — using SEARCH_SPACE defaults")

    # Run context optimization
    results = run_context_optimization(
        llama_server_path=llama_server_path,
        model_path=model_path,
        contexts=contexts,
        cache_types=cache_types,
        min_speed=min_speed,
        model_metadata=metadata,
        best_phase1_config=phase1_config,
        vram_headroom=vram_headroom if not no_telemetry else None,
        no_telemetry=no_telemetry,
    )

    # Display results table
    table = format_results_table(results, min_speed)
    print("")
    print(table)
    print("")

    # Highlight best configuration
    passing = [r for r in results if r.tokens_per_sec >= min_speed]
    if passing:
        best = max(passing, key=lambda r: r.context_size)
        print(f"** Best passing configuration: context={best.context_size}, "
              f"cache={best.cache_type}, speed={best.tokens_per_sec:.1f} tok/s")
    else:
        print(f"** No configuration met the minimum speed of {min_speed} tok/s")


if __name__ == "__main__":
    main()
