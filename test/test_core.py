# test/test_core.py
# Tests for llama_optimus.core — real function calls with mocked I/O boundaries.

from unittest.mock import MagicMock, patch, call
import subprocess


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_trial(params):
    """Create a mock Optuna trial that returns predefined values.

    ``params`` is a dict mapping trial parameter names to the value that
    ``suggest_int`` / ``suggest_categorical`` should return.

    Uses ``side_effect`` so that ``call_args_list`` is available on the
    mock methods (required for assertion in context-constraint tests).
    """
    trial = MagicMock()

    def _suggest_int(name, low, high):
        return params.get(name, low)

    def _suggest_categorical(name, choices):
        return params.get(name, choices[0])

    trial.suggest_int = MagicMock(side_effect=_suggest_int)
    trial.suggest_categorical = MagicMock(side_effect=_suggest_categorical)
    return trial


def _tg_csv(avg_ts=50.0, stddev_ts=1.0):
    """Minimal CSV with a token-generation row (n_gen > 0, n_prompt == 0)."""
    return f"n_prompt,n_gen,avg_ts,stddev_ts\n0,128,{avg_ts},{stddev_ts}\n"


def _pp_csv(avg_ts=80.0, stddev_ts=2.0):
    """Minimal CSV with a prompt-processing row (n_prompt > 0, n_gen == 0)."""
    return f"n_prompt,n_gen,avg_ts,stddev_ts\n256,0,{avg_ts},{stddev_ts}\n"


def _both_csv(tg_avg=50.0, tg_std=1.0, pp_avg=80.0, pp_std=2.0):
    """CSV with both tg and pp rows."""
    return (
        f"n_prompt,n_gen,avg_ts,stddev_ts\n"
        f"0,128,{tg_avg},{tg_std}\n"
        f"256,0,{pp_avg},{pp_std}\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Category 4 — run_llama_bench_with_csv  (CSV parsing)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunLlamaBenchWithCsv:
    """Tests for run_llama_bench_with_csv() — the I/O boundary function."""

    @patch('llama_optimus.core.subprocess.run')
    def test_parses_tg_metric(self, mock_run):
        """tg metric extracts avg_ts from row where n_gen > 0."""
        mock_run.return_value = MagicMock(returncode=0, stdout=_tg_csv(avg_ts=55.5))
        from llama_optimus.core import run_llama_bench_with_csv
        result = run_llama_bench_with_csv(["dummy"], "tg")
        assert result == 55.5

    @patch('llama_optimus.core.subprocess.run')
    def test_parses_pp_metric(self, mock_run):
        """pp metric extracts avg_ts from row where n_prompt > 0."""
        mock_run.return_value = MagicMock(returncode=0, stdout=_pp_csv(avg_ts=92.3))
        from llama_optimus.core import run_llama_bench_with_csv
        result = run_llama_bench_with_csv(["dummy"], "pp")
        assert result == 92.3

    @patch('llama_optimus.core.subprocess.run')
    def test_parses_mean_metric(self, mock_run):
        """mean metric returns (tg + pp) / 2."""
        mock_run.return_value = MagicMock(returncode=0, stdout=_both_csv(tg_avg=50.0, pp_avg=90.0))
        from llama_optimus.core import run_llama_bench_with_csv
        result = run_llama_bench_with_csv(["dummy"], "mean")
        assert result == 70.0  # (50 + 90) / 2

    @patch('llama_optimus.core.subprocess.run')
    def test_returns_0_when_tg_row_empty(self, mock_run):
        """tg metric returns 0.0 when no row has n_gen > 0."""
        mock_run.return_value = MagicMock(returncode=0, stdout=_pp_csv())
        from llama_optimus.core import run_llama_bench_with_csv
        result = run_llama_bench_with_csv(["dummy"], "tg")
        assert result == 0.0

    @patch('llama_optimus.core.subprocess.run')
    def test_returns_0_when_pp_row_empty(self, mock_run):
        """pp metric returns 0.0 when no row has n_prompt > 0."""
        mock_run.return_value = MagicMock(returncode=0, stdout=_tg_csv())
        from llama_optimus.core import run_llama_bench_with_csv
        result = run_llama_bench_with_csv(["dummy"], "pp")
        assert result == 0.0

    @patch('llama_optimus.core.subprocess.run')
    def test_returns_0_when_mean_missing_row(self, mock_run):
        """mean metric returns 0.0 when either tg or pp row is missing."""
        mock_run.return_value = MagicMock(returncode=0, stdout=_tg_csv())  # only tg, no pp
        from llama_optimus.core import run_llama_bench_with_csv
        result = run_llama_bench_with_csv(["dummy"], "mean")
        assert result == 0.0

    @patch('llama_optimus.core.subprocess.run')
    def test_raises_runtime_error_on_nonzero_exit(self, mock_run):
        """RuntimeError raised when subprocess returns non-zero exit code."""
        import pytest
        mock_run.return_value = MagicMock(returncode=1, stderr="OOM killer")
        from llama_optimus.core import run_llama_bench_with_csv
        with pytest.raises(RuntimeError, match="OOM killer"):
            run_llama_bench_with_csv(["dummy"], "tg")


# ──────────────────────────────────────────────────────────────────────────────
# Category 1 — objective_1 command construction
# ──────────────────────────────────────────────────────────────────────────────

class TestObjective1CommandConstruction:
    """Verify that objective_1 builds the correct llama-bench command."""

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_base_command_contains_required_flags(self, mock_bench):
        """Base command must contain all required flags."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '/bin/llama-bench' in cmd
        assert '--batch-size' in cmd
        assert '--ubatch-size' in cmd
        assert '--threads' in cmd
        assert '-ngl' in cmd
        assert '--model' in cmd
        assert '-r' in cmd
        assert '-o' in cmd
        assert '--no-warmup' in cmd

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_cache_flags_added_for_q4_0(self, mock_bench):
        """Non-f16 cache_type adds -ctk and -ctv flags."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'q4_0', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '-ctk' in cmd
        assert '-ctv' in cmd
        idx_ctk = cmd.index('-ctk')
        assert cmd[idx_ctk + 1] == 'q4_0'
        idx_ctv = cmd.index('-ctv')
        assert cmd[idx_ctv + 1] == 'q4_0'

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_cache_flags_omitted_for_f16(self, mock_bench):
        """f16 (default) cache_type omits -ctk/-ctv flags."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '-ctk' not in cmd
        assert '-ctv' not in cmd

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_mmap_flag_added_when_mmap_0(self, mock_bench):
        """mmap=0 adds -mmp 0 flag."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 0,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '-mmp' in cmd
        idx = cmd.index('-mmp')
        assert cmd[idx + 1] == '0'

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_mmap_flag_omitted_when_mmap_1(self, mock_bench):
        """mmap=1 (default) omits -mmp flag."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '-mmp' not in cmd

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_tg_metric_flags(self, mock_bench):
        """metric='tg' adds -n <n_tokens> -p 0."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '-n' in cmd
        idx_n = cmd.index('-n')
        assert cmd[idx_n + 1] == '128'
        assert '-p' in cmd
        idx_p = cmd.index('-p')
        assert cmd[idx_p + 1] == '0'

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_pp_metric_flags(self, mock_bench):
        """metric='pp' adds -p <2*n_tokens> -n 0."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='pp', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '-p' in cmd
        idx_p = cmd.index('-p')
        assert cmd[idx_p + 1] == '256'  # 2 * 128
        assert '-n' in cmd
        idx_n = cmd.index('-n')
        assert cmd[idx_n + 1] == '0'

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_mean_metric_flags(self, mock_bench):
        """metric='mean' adds both -n and -p with token values."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='mean', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '-n' in cmd
        idx_n = cmd.index('-n')
        assert cmd[idx_n + 1] == '128'
        assert '-p' in cmd
        idx_p = cmd.index('-p')
        assert cmd[idx_p + 1] == '256'

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    @patch('llama_optimus.core.get_context_aware_batch_high')
    def test_context_size_constrains_batch(self, mock_batch_high, mock_bench):
        """When context_size is provided, batch sampling uses constrained high."""
        mock_batch_high.return_value = 4096
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf',
                    context_size=32768)
        # Verify get_context_aware_batch_high was called with context_size
        mock_batch_high.assert_called_with(32768)
        # Verify the high bound passed to suggest_int for 'batch' is 4096
        suggest_int_calls = [c for c in trial.suggest_int.call_args_list if c[0][0] == 'batch']
        assert len(suggest_int_calls) == 1
        assert suggest_int_calls[0][0][2] == 4096  # high bound


# ──────────────────────────────────────────────────────────────────────────────
# Category 2 — objective_2 / objective_3 specific tests
# ──────────────────────────────────────────────────────────────────────────────

class TestObjective2Specifics:
    """Verify objective_2 uses fixed numerical params and categorical scanning."""

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_fixed_numerical_params_used_directly(self, mock_bench):
        """batch, u_batch, threads, gpu_layers are passed as fixed values."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'flash_attn': 0, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_2
        objective_2(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf',
                    override_mode='scan', batch=8192, u_batch=1024,
                    threads=16, gpu_layers=80)
        cmd = mock_bench.call_args[0][0]
        # Verify the fixed values appear in the command
        batch_idx = cmd.index('--batch-size')
        assert cmd[batch_idx + 1] == '8192'
        ubatch_idx = cmd.index('--ubatch-size')
        assert cmd[ubatch_idx + 1] == '1024'
        threads_idx = cmd.index('--threads')
        assert cmd[threads_idx + 1] == '16'
        ngl_idx = cmd.index('-ngl')
        assert cmd[ngl_idx + 1] == '80'

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_flash_attn_flag_when_enabled(self, mock_bench):
        """flash_attn=1 adds --flash-attn flag."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'flash_attn': 1, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_2
        objective_2(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf',
                    override_mode='scan', batch=4096, u_batch=512,
                    threads=8, gpu_layers=50)
        cmd = mock_bench.call_args[0][0]
        assert '--flash-attn' in cmd

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_flash_attn_flag_omitted_when_disabled(self, mock_bench):
        """flash_attn=0 omits --flash-attn flag."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'flash_attn': 0, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_2
        objective_2(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf',
                    override_mode='scan', batch=4096, u_batch=512,
                    threads=8, gpu_layers=50)
        cmd = mock_bench.call_args[0][0]
        assert '--flash-attn' not in cmd


class TestObjective3Specifics:
    """Verify objective_3 re-samples numerical params with fixed categorical params."""

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_objective_3_resamples_numerical_params(self, mock_bench):
        """objective_3 samples batch, u_batch, threads, gpu_layers from trial."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 2048, 'u_batch': 256, 'threads': 4,
            'gpu_layers': 30, 'cache_type': 'q8_0', 'mmap': 0,
            'flash_attn': 1,
        })
        from llama_optimus.core import objective_3
        objective_3(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf',
                    override_pattern='none', flash_attn=1, override_mode='scan')
        cmd = mock_bench.call_args[0][0]
        # Verify the sampled values appear
        batch_idx = cmd.index('--batch-size')
        assert cmd[batch_idx + 1] == '2048'
        # Verify flash_attn flag is present
        assert '--flash-attn' in cmd


# ──────────────────────────────────────────────────────────────────────────────
# Category 3 — _run_with_telemetry wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestRunWithTelemetry:
    """Verify the telemetry wrapper applies VRAM penalties and sets attributes."""

    @patch('llama_optimus.core.get_gpu_telemetry')
    def test_calls_objective_and_returns_result(self, mock_telemetry):
        """Wrapper calls the objective and returns its result."""
        mock_telemetry.return_value = MagicMock(
            vram_used_mb=4000, vram_total_mb=8000,
            vram_headroom_pct=50.0, temperature_c=65.0,
            gpu_utilization_pct=80.0
        )
        obj_fn = MagicMock(return_value=55.0)
        trial = MagicMock()

        from llama_optimus.core import _run_with_telemetry
        result = _run_with_telemetry(obj_fn, trial, vram_headroom_threshold=0.12)

        obj_fn.assert_called_once_with(trial)
        # Result may be penalized, but the objective was called
        assert result is not None

    @patch('llama_optimus.core.get_gpu_telemetry')
    @patch('llama_optimus.core.penalize_vram_heavy')
    def test_applies_vram_penalty(self, mock_penalize, mock_telemetry):
        """penalize_vram_heavy is called with post-snapshot."""
        mock_telemetry.return_value = MagicMock(
            vram_used_mb=7000, vram_total_mb=8000,
            vram_headroom_pct=12.5, temperature_c=75.0,
            gpu_utilization_pct=95.0
        )
        mock_penalize.return_value = (40.0, "headroom_low")
        obj_fn = MagicMock(return_value=55.0)
        trial = MagicMock()

        from llama_optimus.core import _run_with_telemetry
        result = _run_with_telemetry(obj_fn, trial, vram_headroom_threshold=0.12)

        mock_penalize.assert_called_once()
        assert result == 40.0

    @patch('llama_optimus.core.get_gpu_telemetry')
    @patch('llama_optimus.core.penalize_vram_heavy')
    def test_sets_user_attributes_on_trial(self, mock_penalize, mock_telemetry):
        """Trial user attributes are set with VRAM metrics."""
        post = MagicMock(
            vram_used_mb=6000, vram_total_mb=8000,
            vram_headroom_pct=25.0, temperature_c=70.0,
            gpu_utilization_pct=90.0
        )
        mock_telemetry.return_value = post
        mock_penalize.return_value = (50.0, "none")
        obj_fn = MagicMock(return_value=55.0)
        trial = MagicMock()

        from llama_optimus.core import _run_with_telemetry
        _run_with_telemetry(obj_fn, trial, vram_headroom_threshold=0.12)

        set_user_attr_calls = trial.set_user_attr.call_args_list
        attr_names = [c[0][0] for c in set_user_attr_calls]
        assert 'vram_used_mb' in attr_names
        assert 'vram_total_mb' in attr_names
        assert 'vram_headroom_pct' in attr_names
        assert 'vram_penalty_reason' in attr_names

    @patch('llama_optimus.core.get_gpu_telemetry')
    def test_no_telemetry_skips_gpu_query(self, mock_telemetry):
        """When no_telemetry=True, get_gpu_telemetry is not called."""
        mock_telemetry.return_value = None
        obj_fn = MagicMock(return_value=55.0)
        trial = MagicMock()

        from llama_optimus.core import _run_with_telemetry
        result = _run_with_telemetry(obj_fn, trial, no_telemetry=True)

        # With no_telemetry=True, the code does:
        #   pre_snap = get_gpu_telemetry() if not no_telemetry else None
        #   post_snap = get_gpu_telemetry() if not no_telemetry else None
        # So get_gpu_telemetry is NEVER called.
        mock_telemetry.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# Category 5 — Error handling
# ──────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    """Verify objectives return 0.0 on exceptions (graceful degradation)."""

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_objective_1_returns_0_on_exception(self, mock_bench):
        """objective_1 catches exceptions and returns 0.0."""
        mock_bench.side_effect = RuntimeError("llama-bench crashed")
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        result = objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                             llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        assert result == 0.0

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_objective_2_returns_0_on_exception(self, mock_bench):
        """objective_2 catches exceptions and returns 0.0."""
        mock_bench.side_effect = RuntimeError("llama-bench crashed")
        trial = make_mock_trial({
            'flash_attn': 0, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_2
        result = objective_2(trial, n_tokens=128, metric='tg', repeat=3,
                             llama_bench_path='/bin/llama-bench', model_path='/m.gguf',
                             override_mode='scan', batch=4096, u_batch=512,
                             threads=8, gpu_layers=50)
        assert result == 0.0

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_objective_3_returns_0_on_exception(self, mock_bench):
        """objective_3 catches exceptions and returns 0.0."""
        mock_bench.side_effect = RuntimeError("llama-bench crashed")
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
            'flash_attn': 0,
        })
        from llama_optimus.core import objective_3
        result = objective_3(trial, n_tokens=128, metric='tg', repeat=3,
                             llama_bench_path='/bin/llama-bench', model_path='/m.gguf',
                             override_pattern='none', flash_attn=0, override_mode='scan')
        assert result == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Additional: Combined flag tests (replacing old overfit helpers)
# ──────────────────────────────────────────────────────────────────────────────

class TestCombinedFlags:
    """Verify that multiple non-default options combine correctly in commands."""

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_cache_and_mmap_combined(self, mock_bench):
        """q4_0 cache + mmap=0 both appear in command."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'q4_0', 'mmap': 0,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        assert '-ctk' in cmd
        assert '-ctv' in cmd
        assert '-mmp' in cmd

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_q8_0_cache_symmetric(self, mock_bench):
        """q8_0 cache type is symmetric (same for K and V)."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'q8_0', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        idx_ctk = cmd.index('-ctk')
        idx_ctv = cmd.index('-ctv')
        assert cmd[idx_ctk + 1] == cmd[idx_ctv + 1] == 'q8_0'

    @patch('llama_optimus.core.run_llama_bench_with_csv')
    def test_all_defaults_produces_minimal_command(self, mock_bench):
        """f16 cache + mmap=1 produces no extra quantization flags."""
        mock_bench.return_value = 50.0
        trial = make_mock_trial({
            'batch': 4096, 'u_batch': 512, 'threads': 8,
            'gpu_layers': 50, 'cache_type': 'f16', 'mmap': 1,
        })
        from llama_optimus.core import objective_1
        objective_1(trial, n_tokens=128, metric='tg', repeat=3,
                    llama_bench_path='/bin/llama-bench', model_path='/m.gguf')
        cmd = mock_bench.call_args[0][0]
        # No cache or mmap flags
        assert '-ctk' not in cmd
        assert '-ctv' not in cmd
        assert '-mmp' not in cmd
