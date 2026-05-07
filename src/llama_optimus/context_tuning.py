# context_tuning.py
# llama-server lifecycle, API benchmarking, and flag translation for P2 context tuning

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .hardware_probe import get_gpu_telemetry  # optional, from P1-1

import requests


@dataclass
class ContextTrialResult:
    """Result of a single context tuning trial."""
    context_size: int
    cache_type: str
    tokens_per_sec: float
    vram_used_mb: Optional[float] = None
    gpu_temp: Optional[float] = None
    server_exit_code: Optional[int] = None
    error: Optional[str] = None
    command: List[str] = field(default_factory=list)


class ServerFlagBuilder:
    """Translate optimization parameters to llama-server flags (not llama-bench flags).

    llama-server uses different flag syntax than llama-bench for mmap and
    flash-attn.  This class encapsulates that translation so the rest of the
    codebase can reason in the internal 0/1 domain.
    """

    @staticmethod
    def build_mmap_flag(mmap: int) -> List[str]:
        """llama-server uses boolean flags: ``--mmap`` (enabled) / ``--no-mmap`` (disabled)."""
        return ["--no-mmap"] if mmap == 0 else ["--mmap"]

    @staticmethod
    def build_flash_attn_flag(flash_attn: int) -> List[str]:
        """llama-server uses: ``--flash-attn [on|off|auto]``."""
        return ["--flash-attn", "on"] if flash_attn == 1 else ["--flash-attn", "off"]

    @staticmethod
    def build_command(
        llama_server_path: str,
        model_path: str,
        context_size: int,
        cache_type: str,
        ngl: int,
        batch: int,
        threads: int,
        mmap: int = 1,
        flash_attn: int = 0,
        host: str = "127.0.0.1",
        port: int = 8889,
        log_format: str = "json",
        override_tensor: Optional[str] = None,
        no_mmproj: bool = False,
    ) -> List[str]:
        """Build complete llama-server command with correct flag syntax."""
        cmd: List[str] = [
            llama_server_path,
            "--model", model_path,
            "-c", str(context_size),
            "-ctk", cache_type,
            "-ctv", cache_type,
            "-ngl", str(ngl),
            "-b", str(batch),
            "-t", str(threads),
        ]
        cmd += ServerFlagBuilder.build_mmap_flag(mmap)
        cmd += ServerFlagBuilder.build_flash_attn_flag(flash_attn)
        cmd += ["--host", host, "--port", str(port)]
        cmd += ["--log-format", log_format]
        if override_tensor:
            cmd += ["--override-tensor", override_tensor]
        if no_mmproj:
            cmd += ["--no-mmproj"]
        return cmd


class ServerLifecycle:
    """Start, wait for readiness, benchmark, and stop llama-server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8889,
        startup_timeout: int = 60,
        benchmark_timeout: int = 120,
    ):
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.benchmark_timeout = benchmark_timeout
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[tempfile.NamedTemporaryFile] = None

    def start(self, cmd: List[str]) -> tuple[bool, str]:
        """Start llama-server as a subprocess.

        Returns:
            (success, error_message) — error_message is non-empty on failure.
        """
        self._log_file = tempfile.NamedTemporaryFile(
            suffix=".log", delete=False, mode="w", encoding="utf-8"
        )

        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )

        health_url = f"http://{self.host}:{self.port}/health"
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    return (True, "")
            except requests.ConnectionError:
                time.sleep(1)
            except Exception as e:
                return (False, str(e))

        return (False, f"Server did not start within {self.startup_timeout}s")

    def benchmark_throughput(
        self,
        prompt: Optional[str] = None,
        n_predict: int = 256,
    ) -> float:
        """Send a timed completion request and measure tokens/second.

        Protocol:
        1. POST /completion with prompt + n_predict
        2. Measure wall-clock time for response
        3. Count tokens in response (from server timing or token count)
        4. Return tokens_per_sec

        Returns:
            tokens_per_sec (float), or 0.0 on failure.
        """
        if prompt is None:
            prompt = "Write a detailed technical analysis of " * 20

        url = f"http://{self.host}:{self.port}/completion"
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0,
            "repeat_penalty": 1.0,
            "cache_prompt": True,
        }

        start_time = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=self.benchmark_timeout)
            elapsed = time.time() - start_time

            if resp.status_code != 200:
                return 0.0

            data = resp.json()
            generated = data.get("content", "")
            timing_ms = data.get("timing", {}).get("generation_ms", elapsed * 1000)

            tokens_generated = data.get("tokens_predicted", n_predict)
            if tokens_generated == 0:
                tokens_generated = max(1, len(generated) // 4)

            tps = tokens_generated / (timing_ms / 1000.0)
            return tps
        except Exception as e:
            print(f"  Benchmark error: {e}")
            return 0.0

    def stop(self, graceful: bool = True) -> int:
        """Stop the server subprocess.

        Returns:
            Exit code (None if process already exited).
        """
        if self._process is None:
            return 0

        if graceful:
            try:
                if os.name == "nt":
                    os.kill(self._process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    self._process.terminate()
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        else:
            self._process.kill()

        exit_code = self._process.returncode
        self._process = None

        if self._log_file:
            self._log_file.close()
            try:
                os.unlink(self._log_file.name)
            except OSError:
                pass

        return exit_code

    def run_trial(
        self,
        cmd: List[str],
        prompt: Optional[str] = None,
        n_predict: int = 256,
    ) -> ContextTrialResult:
        """Run a complete trial: start server, benchmark, stop.

        Main entry point for P2-2 grid search to call.
        """
        # Extract context_size and cache_type from the command for the result
        context_size = 0
        cache_type = "f16"
        for i, arg in enumerate(cmd):
            if arg == "-c" and i + 1 < len(cmd):
                try:
                    context_size = int(cmd[i + 1])
                except ValueError:
                    pass
            if arg == "-ctk" and i + 1 < len(cmd):
                cache_type = cmd[i + 1]

        success, error_msg = self.start(cmd)
        if not success:
            return ContextTrialResult(
                context_size=context_size,
                cache_type=cache_type,
                tokens_per_sec=0.0,
                vram_used_mb=None,
                gpu_temp=None,
                server_exit_code=None,
                error=error_msg,
                command=cmd,
            )

        tps = self.benchmark_throughput(prompt, n_predict)
        exit_code = self.stop()

        return ContextTrialResult(
            context_size=context_size,
            cache_type=cache_type,
            tokens_per_sec=tps,
            vram_used_mb=None,
            gpu_temp=None,
            server_exit_code=exit_code,
            error=None,
            command=cmd,
        )


def run_context_optimization(
    llama_server_path: str,
    model_path: str,
    contexts: List[int],
    cache_types: List[str],
    min_speed: float,
    model_metadata,
    best_phase1_config: dict,
    vram_headroom: Optional[float] = None,
    no_telemetry: bool = False,
) -> List[ContextTrialResult]:
    """Run context × cache_type grid search using llama-server.

    Parameters:
        llama_server_path: Path to llama-server binary
        model_path: Path to GGUF model
        contexts: List of context sizes to test (e.g., [32768, 65536, 131072])
        cache_types: List of cache types (e.g., ['f16', 'q8_0', 'q4_0'])
        min_speed: Minimum tokens/second threshold
        model_metadata: Model info from P0-1 (max_context, has_vision, etc.)
        best_phase1_config: Best parameters from Phase 1 (batch, threads, ngl, etc.)
        vram_headroom: Optional VRAM headroom threshold for penalization
        no_telemetry: Skip nvidia-smi queries if True

    Returns:
        List of ContextTrialResult, sorted by context_size descending.
    """
    results: List[ContextTrialResult] = []

    # Validate context sizes against model metadata
    max_ctx = model_metadata.max_context or float("inf")
    valid_contexts = [c for c in contexts if c <= max_ctx]
    pruned_contexts = [c for c in contexts if c > max_ctx]

    if pruned_contexts:
        print(f"  Context sizes pruned (exceed model max_context={max_ctx}): {pruned_contexts}")

    lifecycle = ServerLifecycle()

    for ctx_size in valid_contexts:
        for cache_type in cache_types:
            print(f"\n  Testing: context={ctx_size}, cache_type={cache_type}")

            # Pre-trial telemetry (optional)
            pre_snap = None
            if not no_telemetry:
                pre_snap = get_gpu_telemetry()

            # Build command using best Phase 1 parameters + context + cache type
            cmd = ServerFlagBuilder.build_command(
                llama_server_path=llama_server_path,
                model_path=model_path,
                context_size=ctx_size,
                cache_type=cache_type,
                ngl=best_phase1_config.get("gpu_layers", 0),
                batch=best_phase1_config.get("batch", 2048),
                threads=best_phase1_config.get("threads", 8),
                mmap=best_phase1_config.get("mmap", 1),
                flash_attn=best_phase1_config.get("flash_attn", 0),
                override_tensor=best_phase1_config.get("override_tensor"),
                no_mmproj=model_metadata.has_vision,  # Auto-enable for vision models
            )

            trial = lifecycle.run_trial(cmd)

            # Post-trial telemetry (optional)
            if not no_telemetry:
                post_snap = get_gpu_telemetry()
                trial.vram_used_mb = post_snap.vram_used_mb if post_snap else None
                trial.gpu_temp = post_snap.temperature_c if post_snap else None

                # Apply VRAM penalization if headroom is tight
                if vram_headroom and post_snap and post_snap.vram_headroom_pct < (vram_headroom * 100):
                    original_tps = trial.tokens_per_sec
                    trial.tokens_per_sec *= 0.7  # penalize tight VRAM
                    print(f"  VRAM penalization applied: {original_tps:.1f} → {trial.tokens_per_sec:.1f} tk/s")

            results.append(trial)

    return sorted(results, key=lambda x: -x.context_size)


def format_results_table(results: List[ContextTrialResult], min_speed: float) -> str:
    """Format results as a table with pass/fail indicators.

    Columns: Context | Cache Type | Speed (tk/s) | VRAM (MB) | Temp (°C) | Status
    """
    header = f"{'Context':>8} | {'Cache':>6} | {'Speed':>7} | {'VRAM (MB)':>9} | {'Temp (°C)':>8} | {'Status':>6}"
    separator = "-" * len(header)

    rows = [header, separator]

    for r in sorted(results, key=lambda x: (-x.context_size, x.cache_type)):
        status = "PASS" if r.tokens_per_sec >= min_speed else "FAIL"
        vram_str = f"{r.vram_used_mb:.0f}" if r.vram_used_mb else "—"
        temp_str = f"{r.gpu_temp:.0f}" if r.gpu_temp else "—"
        rows.append(
            f"{r.context_size:>8} | {r.cache_type:>6} | "
            f"{r.tokens_per_sec:>6.1f} | {vram_str:>9} | "
            f"{temp_str:>8} | {status:>6}"
        )

    return "\n".join(rows)
