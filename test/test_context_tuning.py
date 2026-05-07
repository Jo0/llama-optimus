# test/test_context_tuning.py
# Unit tests for context_tuning.py — server lifecycle, API benchmarking,
# flag translation, grid search, and results formatting.

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest import mock

import requests

from llama_optimus.context_tuning import (
    ContextTrialResult,
    ServerFlagBuilder,
    ServerLifecycle,
    run_context_optimization,
    format_results_table,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeModelMetadata:
    """Minimal stand-in for the real ModelMetadata."""
    max_context: Optional[int] = 131072
    has_vision: bool = False


@dataclass
class FakeGpuSnapshot:
    vram_used_mb: float = 16000.0
    vram_total_mb: float = 24000.0
    vram_headroom_pct: float = 33.3
    temperature_c: float = 60.0
    gpu_index: int = 0


# ---------------------------------------------------------------------------
# ContextTrialResult
# ---------------------------------------------------------------------------

class TestContextTrialResult:
    def test_defaults(self):
        r = ContextTrialResult(context_size=65536, cache_type="q4_0", tokens_per_sec=20.0)
        assert r.vram_used_mb is None
        assert r.gpu_temp is None
        assert r.server_exit_code is None
        assert r.error is None
        assert r.command == []

    def test_full(self):
        r = ContextTrialResult(
            context_size=32768,
            cache_type="f16",
            tokens_per_sec=10.5,
            vram_used_mb=20000.0,
            gpu_temp=72.0,
            server_exit_code=0,
            error=None,
            command=["llama-server", "-c", "32768"],
        )
        assert r.context_size == 32768
        assert len(r.command) == 3


# ---------------------------------------------------------------------------
# ServerFlagBuilder — mmap
# ---------------------------------------------------------------------------

class TestServerFlagBuilderMmap:
    def test_mmap_enabled(self):
        flags = ServerFlagBuilder.build_mmap_flag(1)
        assert flags == ["--mmap"]

    def test_mmap_disabled(self):
        flags = ServerFlagBuilder.build_mmap_flag(0)
        assert flags == ["--no-mmap"]

    def test_mmap_truthy(self):
        # Any non-zero value should enable mmap
        flags = ServerFlagBuilder.build_mmap_flag(2)
        assert flags == ["--mmap"]


# ---------------------------------------------------------------------------
# ServerFlagBuilder — flash attention
# ---------------------------------------------------------------------------

class TestServerFlagBuilderFlashAttn:
    def test_flash_attn_enabled(self):
        flags = ServerFlagBuilder.build_flash_attn_flag(1)
        assert flags == ["--flash-attn", "on"]

    def test_flash_attn_disabled(self):
        flags = ServerFlagBuilder.build_flash_attn_flag(0)
        assert flags == ["--flash-attn", "off"]

    def test_flash_attn_truthy(self):
        # The code checks `flash_attn == 1`, so value 2 → off
        flags = ServerFlagBuilder.build_flash_attn_flag(2)
        assert flags == ["--flash-attn", "off"]


# ---------------------------------------------------------------------------
# ServerFlagBuilder — full command
# ---------------------------------------------------------------------------

class TestServerFlagBuilderCommand:
    def test_minimal_command(self):
        cmd = ServerFlagBuilder.build_command(
            llama_server_path="./llama-server",
            model_path="/models/test.gguf",
            context_size=65536,
            cache_type="q4_0",
            ngl=42,
            batch=2048,
            threads=8,
        )
        assert cmd[0] == "./llama-server"
        assert "--model" in cmd
        assert "/models/test.gguf" in cmd
        assert "-c" in cmd
        assert "65536" in cmd
        assert "-ctk" in cmd
        assert "-ctv" in cmd
        assert "q4_0" in cmd
        assert "-ngl" in cmd
        assert "42" in cmd
        assert "-b" in cmd
        assert "2048" in cmd
        assert "-t" in cmd
        assert "8" in cmd

    def test_mmap_and_flash_attn_defaults(self):
        cmd = ServerFlagBuilder.build_command(
            llama_server_path="./llama-server",
            model_path="/models/test.gguf",
            context_size=32768,
            cache_type="f16",
            ngl=0,
            batch=1024,
            threads=4,
        )
        # Default mmap=1 → --mmap
        assert "--mmap" in cmd
        # Default flash_attn=0 → --flash-attn off
        assert "--flash-attn" in cmd
        idx = cmd.index("--flash-attn")
        assert cmd[idx + 1] == "off"

    def test_override_tensor_included(self):
        cmd = ServerFlagBuilder.build_command(
            llama_server_path="./llama-server",
            model_path="/models/test.gguf",
            context_size=32768,
            cache_type="q8_0",
            ngl=0,
            batch=1024,
            threads=4,
            override_tensor="token_embd.weight,0,0",
        )
        assert "--override-tensor" in cmd
        assert "token_embd.weight,0,0" in cmd

    def test_override_tensor_none_excluded(self):
        cmd = ServerFlagBuilder.build_command(
            llama_server_path="./llama-server",
            model_path="/models/test.gguf",
            context_size=32768,
            cache_type="q8_0",
            ngl=0,
            batch=1024,
            threads=4,
            override_tensor=None,
        )
        assert "--override-tensor" not in cmd

    def test_no_mmproj_for_vision(self):
        cmd = ServerFlagBuilder.build_command(
            llama_server_path="./llama-server",
            model_path="/models/vision.gguf",
            context_size=4096,
            cache_type="f16",
            ngl=0,
            batch=512,
            threads=4,
            no_mmproj=True,
        )
        assert "--no-mmproj" in cmd

    def test_no_mmproj_default_false(self):
        cmd = ServerFlagBuilder.build_command(
            llama_server_path="./llama-server",
            model_path="/models/test.gguf",
            context_size=4096,
            cache_type="f16",
            ngl=0,
            batch=512,
            threads=4,
            no_mmproj=False,
        )
        assert "--no-mmproj" not in cmd

    def test_host_and_port(self):
        cmd = ServerFlagBuilder.build_command(
            llama_server_path="./llama-server",
            model_path="/models/test.gguf",
            context_size=32768,
            cache_type="f16",
            ngl=0,
            batch=1024,
            threads=4,
            host="0.0.0.0",
            port=9999,
        )
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "9999" in cmd


# ---------------------------------------------------------------------------
# ServerLifecycle — benchmark_throughput
# ---------------------------------------------------------------------------

class TestServerLifecycleBenchmark:
    def _make_lifecycle(self) -> ServerLifecycle:
        return ServerLifecycle(host="127.0.0.1", port=8765, benchmark_timeout=30)

    def test_successful_benchmark_with_timing(self):
        lc = self._make_lifecycle()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Generated text here",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 100,
        }

        with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_response):
            tps = lc.benchmark_throughput(prompt="hello", n_predict=100)

        # 100 tokens / 0.5s = 200 tps
        assert tps == 200.0

    def test_successful_benchmark_fallback_to_n_predict(self):
        lc = self._make_lifecycle()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        # tokens_predicted = 0 → falls back to len(content)//4
        mock_response.json.return_value = {
            "content": "abcd efgh",
            "timing": {"generation_ms": 1000},
            "tokens_predicted": 0,
        }

        with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_response):
            tps = lc.benchmark_throughput(prompt="hello", n_predict=100)

        # content = "abcd efgh" → 10 chars → max(1, 10//4) = 2 tokens
        # 2 tokens / 1.0s = 2.0 tps
        assert tps == 2.0

    def test_benchmark_non_200_status(self):
        lc = self._make_lifecycle()
        mock_response = mock.Mock()
        mock_response.status_code = 500

        with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_response):
            tps = lc.benchmark_throughput()

        assert tps == 0.0

    def test_benchmark_connection_error(self):
        lc = self._make_lifecycle()

        with mock.patch(
            "llama_optimus.context_tuning.requests.post",
            side_effect=Exception("Connection refused"),
        ):
            tps = lc.benchmark_throughput()

        assert tps == 0.0

    def test_default_prompt(self):
        lc = self._make_lifecycle()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "out",
            "timing": {"generation_ms": 100},
            "tokens_predicted": 10,
        }

        captured_payload = {}

        def _post(url, json=None, timeout=None):
            captured_payload.update(json)
            return mock_response

        with mock.patch("llama_optimus.context_tuning.requests.post", side_effect=_post):
            lc.benchmark_throughput()  # no prompt arg

        assert captured_payload["prompt"] == "Write a detailed technical analysis of " * 20

    def test_benchmark_fallback_timing_ms_when_missing(self):
        """When timing key is missing, fallback to elapsed wall-clock time."""
        lc = self._make_lifecycle()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        # No "timing" key at all → fallback uses elapsed * 1000
        mock_response.json.return_value = {
            "content": "some generated content",
            "tokens_predicted": 50,
        }

        with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_response):
            tps = lc.benchmark_throughput(prompt="hello", n_predict=50)

        # tps > 0 because it falls back to elapsed time
        assert tps > 0.0


# ---------------------------------------------------------------------------
# ServerLifecycle — start (mocked)
# ---------------------------------------------------------------------------

class TestServerLifecycleStart:
    def _make_lifecycle(self) -> ServerLifecycle:
        return ServerLifecycle(host="127.0.0.1", port=8766, startup_timeout=5)

    def test_start_success_on_first_health(self):
        lc = self._make_lifecycle()
        mock_response = mock.Mock()
        mock_response.status_code = 200

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen") as mock_popen:
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_response):
                ok, err = lc.start(["llama-server", "-c", "4096"])

        assert ok is True
        assert err == ""

    def test_start_timeout(self):
        lc = self._make_lifecycle()

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen"):
            with mock.patch(
                "llama_optimus.context_tuning.requests.get",
                side_effect=requests.ConnectionError("not ready"),
            ):
                with mock.patch("llama_optimus.context_tuning.time.sleep"):
                    ok, err = lc.start(["llama-server", "-c", "4096"])

        assert ok is False
        assert "did not start within" in err

    def test_start_exception(self):
        lc = self._make_lifecycle()

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen"):
            with mock.patch(
                "llama_optimus.context_tuning.requests.get",
                side_effect=Exception("DNS failure"),
            ):
                ok, err = lc.start(["llama-server", "-c", "4096"])

        assert ok is False
        assert "DNS failure" in err

    def test_start_success_after_multiple_retries(self):
        """Server health fails with ConnectionError first, then returns 200."""
        lc = self._make_lifecycle()

        mock_ok = mock.Mock()
        mock_ok.status_code = 200

        call_count = [0]

        def _health_get(url, timeout=None):
            call_count[0] += 1
            if call_count[0] < 3:
                raise requests.ConnectionError("not ready yet")
            return mock_ok

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen"):
            with mock.patch(
                "llama_optimus.context_tuning.requests.get",
                side_effect=_health_get,
            ):
                with mock.patch("llama_optimus.context_tuning.time.sleep"):
                    ok, err = lc.start(["llama-server", "-c", "4096"])

        assert ok is True
        assert err == ""
        assert call_count[0] >= 3

    def test_start_health_returns_non_200_status(self):
        """Health endpoint returns 503 (non-connection error status)."""
        lc = self._make_lifecycle()

        mock_resp = mock.Mock()
        mock_resp.status_code = 503

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen"):
            with mock.patch(
                "llama_optimus.context_tuning.requests.get",
                return_value=mock_resp,
            ):
                ok, err = lc.start(["llama-server", "-c", "4096"])

        # Non-200 from health check is a generic Exception path → returns False
        assert ok is False


# ---------------------------------------------------------------------------
# ServerLifecycle — stop
# ---------------------------------------------------------------------------

class TestServerLifecycleStop:
    def test_stop_no_process(self):
        lc = ServerLifecycle()
        code = lc.stop()
        assert code == 0

    def test_stop_graceful_success(self):
        lc = ServerLifecycle()
        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        lc._process = mock_proc

        mock_log = mock.Mock()
        lc._log_file = mock_log

        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
            with mock.patch("llama_optimus.context_tuning.os.unlink"):
                code = lc.stop(graceful=True)

        assert code == 0
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=10)

    def test_stop_force_kill_on_timeout(self):
        lc = ServerLifecycle()
        mock_proc = mock.Mock()
        mock_proc.returncode = 1
        mock_proc.pid = 12345
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 10),
            None,  # second wait after kill
        ]
        lc._process = mock_proc
        mock_log = mock.Mock()
        lc._log_file = mock_log

        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
            with mock.patch("llama_optimus.context_tuning.os.unlink"):
                code = lc.stop(graceful=True)

        assert code == 1
        mock_proc.kill.assert_called_once()

    def test_stop_graceful_windows_ctrl_break(self):
        """Graceful stop on Windows sends CTRL_BREAK_EVENT."""
        lc = ServerLifecycle()
        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.pid = 9999
        lc._process = mock_proc
        mock_log = mock.Mock()
        lc._log_file = mock_log

        ctrl_break_sent = [False]

        def _kill(pid, sig):
            import signal
            if sig == signal.CTRL_BREAK_EVENT:
                ctrl_break_sent[0] = True

        with mock.patch("llama_optimus.context_tuning.os.name", "nt"):
            with mock.patch("llama_optimus.context_tuning.os.kill", side_effect=_kill):
                with mock.patch("llama_optimus.context_tuning.os.unlink"):
                    code = lc.stop(graceful=True)

        assert code == 0
        assert ctrl_break_sent[0] is True
        mock_proc.wait.assert_called_once_with(timeout=10)

    def test_stop_non_graceful(self):
        """Non-graceful stop calls kill() directly without terminate."""
        lc = ServerLifecycle()
        mock_proc = mock.Mock()
        mock_proc.returncode = -9
        mock_proc.pid = 12345
        lc._process = mock_proc
        mock_log = mock.Mock()
        lc._log_file = mock_log

        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
            with mock.patch("llama_optimus.context_tuning.os.unlink"):
                code = lc.stop(graceful=False)

        assert code == -9
        mock_proc.kill.assert_called_once()
        # Non-graceful path calls kill() directly, no terminate() or wait()
        mock_proc.terminate.assert_not_called()

    def test_stop_log_file_oserror_cleanup(self):
        """OSError during log file unlink is silently swallowed."""
        lc = ServerLifecycle()
        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        lc._process = mock_proc
        mock_log = mock.Mock()
        lc._log_file = mock_log

        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
            with mock.patch(
                "llama_optimus.context_tuning.os.unlink",
                side_effect=OSError("file locked"),
            ):
                code = lc.stop(graceful=True)

        assert code == 0
        mock_log.close.assert_called_once()


# ---------------------------------------------------------------------------
# ServerLifecycle — run_trial
# ---------------------------------------------------------------------------

class TestServerLifecycleRunTrial:
    def test_run_trial_success(self):
        lc = ServerLifecycle(host="127.0.0.1", port=8767)
        cmd = ["llama-server", "-c", "65536", "-ctk", "q4_0"]

        mock_health = mock.Mock()
        mock_health.status_code = 200

        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        mock_bench.json.return_value = {
            "content": "output",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 200,
        }

        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            result = lc.run_trial(cmd, n_predict=200)

        assert result.context_size == 65536
        assert result.cache_type == "q4_0"
        assert result.tokens_per_sec == 400.0  # 200 / 0.5s
        assert result.error is None
        assert result.server_exit_code == 0

    def test_run_trial_start_failure(self):
        lc = ServerLifecycle(host="127.0.0.1", port=8768, startup_timeout=1)
        cmd = ["llama-server", "-c", "32768", "-ctk", "f16"]

        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch(
                "llama_optimus.context_tuning.requests.get",
                side_effect=requests.ConnectionError(),
            ):
                with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                    with mock.patch("llama_optimus.context_tuning.time.sleep"):
                        result = lc.run_trial(cmd)

        assert result.context_size == 32768
        assert result.cache_type == "f16"
        assert result.tokens_per_sec == 0.0
        assert result.error is not None

    def test_run_trial_extracts_context_and_cache(self):
        lc = ServerLifecycle(host="127.0.0.1", port=8769, startup_timeout=1)
        cmd = ["llama-server", "-c", "131072", "-ctk", "q8_0", "-ctv", "q8_0"]

        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch(
                "llama_optimus.context_tuning.requests.get",
                side_effect=requests.ConnectionError(),
            ):
                with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                    with mock.patch("llama_optimus.context_tuning.time.sleep"):
                        result = lc.run_trial(cmd)

        assert result.context_size == 131072
        assert result.cache_type == "q8_0"

    def test_run_trial_zero_tps_returns_proper_result(self):
        """Trial where benchmark returns 0 tps still returns a proper result."""
        lc = ServerLifecycle(host="127.0.0.1", port=8770)
        cmd = ["llama-server", "-c", "32768", "-ctk", "f16"]

        mock_health = mock.Mock()
        mock_health.status_code = 200

        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        # Server returns 0 tokens_predicted with non-zero timing → 0 tps
        mock_bench.json.return_value = {
            "content": "",
            "timing": {"generation_ms": 5000},
            "tokens_predicted": 0,
        }

        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            result = lc.run_trial(cmd, n_predict=0)

        assert result.context_size == 32768
        assert result.cache_type == "f16"
        assert result.error is None
        assert result.server_exit_code == 0
        # content is empty string → max(1, 0//4) = 1 token → 1 / 5.0 = 0.2 tps
        assert result.tokens_per_sec == 0.2


# ---------------------------------------------------------------------------
# run_context_optimization — grid search
# ---------------------------------------------------------------------------

class TestRunContextOptimization:
    def _make_metadata(self, max_ctx: int = 131072, has_vision: bool = False) -> FakeModelMetadata:
        return FakeModelMetadata(max_context=max_ctx, has_vision=has_vision)

    def test_contexts_pruned_by_max_context(self, capsys):
        metadata = self._make_metadata(max_ctx=65536)
        mock_health = mock.Mock()
        mock_health.status_code = 200

        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        mock_bench.json.return_value = {
            "content": "out",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 100,
        }

        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            with mock.patch("llama_optimus.hardware_probe.get_gpu_telemetry", return_value=None):
                                results = run_context_optimization(
                                    llama_server_path="./llama-server",
                                    model_path="/models/test.gguf",
                                    contexts=[32768, 65536, 131072],
                                    cache_types=["f16"],
                                    min_speed=10.0,
                                    model_metadata=metadata,
                                    best_phase1_config={"gpu_layers": 0, "batch": 2048, "threads": 4},
                                    no_telemetry=True,
                                )

        # 131072 should be pruned, only 32768 and 65536 remain
        assert len(results) == 2
        assert results[0].context_size == 65536  # sorted descending
        assert results[1].context_size == 32768

        captured = capsys.readouterr()
        assert "131072" in captured.out

    def test_grid_search_runs_all_combinations(self):
        metadata = self._make_metadata(max_ctx=131072)
        mock_health = mock.Mock()
        mock_health.status_code = 200

        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        mock_bench.json.return_value = {
            "content": "out",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 100,
        }

        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            results = run_context_optimization(
                                llama_server_path="./llama-server",
                                model_path="/models/test.gguf",
                                contexts=[32768, 65536],
                                cache_types=["f16", "q4_0"],
                                min_speed=10.0,
                                model_metadata=metadata,
                                best_phase1_config={"gpu_layers": 0, "batch": 2048, "threads": 4},
                                no_telemetry=True,
                            )

        # 2 contexts × 2 cache types = 4 results
        assert len(results) == 4
        # Sorted by context_size descending
        assert results[0].context_size >= results[1].context_size

    def test_no_mmproj_for_vision_model(self):
        metadata = self._make_metadata(max_ctx=4096, has_vision=True)
        mock_health = mock.Mock()
        mock_health.status_code = 200
        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        mock_bench.json.return_value = {
            "content": "out",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 50,
        }
        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345

        captured_cmds = []

        def _popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            return mock_proc

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", side_effect=_popen):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            run_context_optimization(
                                llama_server_path="./llama-server",
                                model_path="/models/vision.gguf",
                                contexts=[4096],
                                cache_types=["f16"],
                                min_speed=5.0,
                                model_metadata=metadata,
                                best_phase1_config={"gpu_layers": 0, "batch": 512, "threads": 4},
                                no_telemetry=True,
                            )

        assert len(captured_cmds) == 1
        assert "--no-mmproj" in captured_cmds[0]

    def test_results_sorted_descending_by_context(self):
        metadata = self._make_metadata(max_ctx=131072)
        mock_health = mock.Mock()
        mock_health.status_code = 200
        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        mock_bench.json.return_value = {
            "content": "out",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 100,
        }
        mock_proc = mock.Mock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            results = run_context_optimization(
                                llama_server_path="./llama-server",
                                model_path="/models/test.gguf",
                                contexts=[131072, 32768, 65536],
                                cache_types=["q4_0"],
                                min_speed=10.0,
                                model_metadata=metadata,
                                best_phase1_config={"gpu_layers": 0, "batch": 2048, "threads": 4},
                                no_telemetry=True,
                            )

        assert results[0].context_size == 131072
        assert results[1].context_size == 65536
        assert results[2].context_size == 32768

    def test_telemetry_populates_vram_and_temp(self):
        metadata = self._make_metadata(max_ctx=32768)
        snap = FakeGpuSnapshot(
            vram_used_mb=18000.0,
            vram_total_mb=24000.0,
            vram_headroom_pct=25.0,
            temperature_c=68.0,
        )

        mock_health = mock.Mock()
        mock_health.status_code = 200
        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        mock_bench.json.return_value = {
            "content": "out",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 100,
        }
        mock_proc = mock.Mock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            with mock.patch(
                                "llama_optimus.context_tuning.get_gpu_telemetry", return_value=snap
                            ):
                                results = run_context_optimization(
                                    llama_server_path="./llama-server",
                                    model_path="/models/test.gguf",
                                    contexts=[32768],
                                    cache_types=["q4_0"],
                                    min_speed=10.0,
                                    model_metadata=metadata,
                                    best_phase1_config={"gpu_layers": 0, "batch": 2048, "threads": 4},
                                    no_telemetry=False,
                                )

        assert len(results) == 1
        assert results[0].vram_used_mb == 18000.0
        assert results[0].gpu_temp == 68.0

    def test_vram_penalization_applied_when_headroom_tight(self, capsys):
        """VRAM penalization reduces tps when headroom is tight."""
        metadata = self._make_metadata(max_ctx=32768)
        snap = FakeGpuSnapshot(
            vram_used_mb=22000.0,
            vram_total_mb=24000.0,
            vram_headroom_pct=8.0,  # below threshold (12%)
            temperature_c=75.0,
        )

        mock_health = mock.Mock()
        mock_health.status_code = 200
        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        mock_bench.json.return_value = {
            "content": "out",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 100,
        }
        mock_proc = mock.Mock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            with mock.patch(
                                "llama_optimus.context_tuning.get_gpu_telemetry", return_value=snap
                            ):
                                results = run_context_optimization(
                                    llama_server_path="./llama-server",
                                    model_path="/models/test.gguf",
                                    contexts=[32768],
                                    cache_types=["q4_0"],
                                    min_speed=10.0,
                                    model_metadata=metadata,
                                    best_phase1_config={"gpu_layers": 0, "batch": 2048, "threads": 4},
                                    vram_headroom=0.12,
                                    no_telemetry=False,
                                )

        assert len(results) == 1
        # Original tps would be 200.0; with 0.7 penalty → 140.0
        assert results[0].tokens_per_sec == 140.0
        captured = capsys.readouterr()
        assert "VRAM penalization" in captured.out

    def test_empty_contexts_returns_empty_results(self):
        """Empty contexts list returns empty results."""
        metadata = self._make_metadata(max_ctx=131072)

        results = run_context_optimization(
            llama_server_path="./llama-server",
            model_path="/models/test.gguf",
            contexts=[],
            cache_types=["f16", "q4_0"],
            min_speed=10.0,
            model_metadata=metadata,
            best_phase1_config={"gpu_layers": 0, "batch": 2048, "threads": 4},
            no_telemetry=True,
        )

        assert len(results) == 0

    def test_telemetry_none_no_crash(self):
        """Telemetry returning None (no GPU) doesn't crash the optimization loop."""
        metadata = self._make_metadata(max_ctx=32768)

        mock_health = mock.Mock()
        mock_health.status_code = 200
        mock_bench = mock.Mock()
        mock_bench.status_code = 200
        mock_bench.json.return_value = {
            "content": "out",
            "timing": {"generation_ms": 500},
            "tokens_predicted": 100,
        }
        mock_proc = mock.Mock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0

        with mock.patch("llama_optimus.context_tuning.subprocess.Popen", return_value=mock_proc):
            with mock.patch("llama_optimus.context_tuning.requests.get", return_value=mock_health):
                with mock.patch("llama_optimus.context_tuning.requests.post", return_value=mock_bench):
                    with mock.patch("llama_optimus.context_tuning.os.unlink"):
                        with mock.patch("llama_optimus.context_tuning.os.name", "posix"):
                            with mock.patch(
                                "llama_optimus.context_tuning.get_gpu_telemetry", return_value=None
                            ):
                                results = run_context_optimization(
                                    llama_server_path="./llama-server",
                                    model_path="/models/test.gguf",
                                    contexts=[32768],
                                    cache_types=["f16"],
                                    min_speed=10.0,
                                    model_metadata=metadata,
                                    best_phase1_config={"gpu_layers": 0, "batch": 2048, "threads": 4},
                                    no_telemetry=False,
                                )

        assert len(results) == 1
        assert results[0].vram_used_mb is None
        assert results[0].gpu_temp is None


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------

class TestFormatResultsTable:
    def _make_results(self) -> List[ContextTrialResult]:
        return [
            ContextTrialResult(context_size=131072, cache_type="q4_0", tokens_per_sec=25.0, vram_used_mb=20000.0, gpu_temp=70.0),
            ContextTrialResult(context_size=65536, cache_type="q4_0", tokens_per_sec=30.0, vram_used_mb=16000.0, gpu_temp=65.0),
            ContextTrialResult(context_size=65536, cache_type="f16", tokens_per_sec=18.0, vram_used_mb=22000.0, gpu_temp=72.0),
            ContextTrialResult(context_size=32768, cache_type="f16", tokens_per_sec=0.0, error="timeout"),
        ]

    def test_table_has_header(self):
        table = format_results_table(self._make_results(), min_speed=20.0)
        assert "Context" in table
        assert "Cache" in table
        assert "Speed" in table
        assert "VRAM" in table
        assert "Temp" in table
        assert "Status" in table

    def test_pass_fail_indicators(self):
        table = format_results_table(self._make_results(), min_speed=20.0)
        lines = table.strip().split("\n")
        # header + separator + 4 data rows
        data_lines = lines[2:]
        pass_count = sum(1 for l in data_lines if "PASS" in l)
        fail_count = sum(1 for l in data_lines if "FAIL" in l)
        # 131072/q4_0=25→PASS, 65536/q4_0=30→PASS, 65536/f16=18→FAIL, 32768/f16=0→FAIL
        assert pass_count == 2
        assert fail_count == 2

    def test_sorted_by_context_descending(self):
        table = format_results_table(self._make_results(), min_speed=20.0)
        lines = table.strip().split("\n")
        data_lines = lines[2:]
        # First data row should have the largest context
        assert "131072" in data_lines[0]

    def test_missing_vram_and_temp_show_dash(self):
        results = [
            ContextTrialResult(context_size=32768, cache_type="f16", tokens_per_sec=15.0, vram_used_mb=None, gpu_temp=None),
        ]
        table = format_results_table(results, min_speed=10.0)
        assert "—" in table

    def test_empty_results(self):
        table = format_results_table([], min_speed=20.0)
        lines = table.strip().split("\n")
        assert len(lines) == 2  # header + separator only

    def test_boundary_speed_equals_min_speed(self):
        """When speed exactly equals min_speed, the trial should PASS."""
        results = [
            ContextTrialResult(context_size=32768, cache_type="f16", tokens_per_sec=20.0),
        ]
        table = format_results_table(results, min_speed=20.0)
        data_lines = table.strip().split("\n")[2:]
        assert "PASS" in data_lines[0]
        assert "FAIL" not in data_lines[0]
