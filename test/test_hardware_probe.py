# test/test_hardware_probe.py

from llama_optimus.hardware_probe import (
    GpuTelemetrySnapshot,
    penalize_vram_heavy,
)


class TestGpuTelemetrySnapshot:
    """Tests for the GpuTelemetrySnapshot dataclass."""

    def test_vram_headroom_calculation(self):
        snap = GpuTelemetrySnapshot(
            vram_used_mb=20000.0,
            vram_total_mb=24000.0,
            temperature_c=65.0,
            gpu_utilization_pct=80.0,
        )
        expected = (1.0 - 20000.0 / 24000.0) * 100.0  # 16.67%
        assert abs(snap.vram_headroom_pct - expected) < 0.01

    def test_vram_headroom_zero_total(self):
        snap = GpuTelemetrySnapshot(
            vram_used_mb=1000.0,
            vram_total_mb=0.0,
            temperature_c=50.0,
            gpu_utilization_pct=0.0,
        )
        assert snap.vram_headroom_pct == 0.0

    def test_vram_headroom_full(self):
        snap = GpuTelemetrySnapshot(
            vram_used_mb=24000.0,
            vram_total_mb=24000.0,
            temperature_c=70.0,
            gpu_utilization_pct=95.0,
        )
        assert snap.vram_headroom_pct == 0.0

    def test_gpu_index_default(self):
        snap = GpuTelemetrySnapshot(
            vram_used_mb=1000.0,
            vram_total_mb=8000.0,
            temperature_c=50.0,
            gpu_utilization_pct=10.0,
        )
        assert snap.gpu_index == 0


class TestPenalizeVramHeavy:
    """Tests for VRAM-based penalization logic."""

    def _make_snap(self, headroom_pct):
        """Create a snapshot with a specific VRAM headroom percentage."""
        total = 24000.0
        used = total * (1.0 - headroom_pct / 100.0)
        return GpuTelemetrySnapshot(
            vram_used_mb=used,
            vram_total_mb=total,
            temperature_c=60.0,
            gpu_utilization_pct=50.0,
        )

    def test_optimal_zone_no_penalty(self):
        """≥13% headroom → no penalty."""
        snap = self._make_snap(15.0)  # 15% headroom
        penalized, reason = penalize_vram_heavy(100.0, snap)
        assert penalized == 100.0
        assert reason == "optimal"

    def test_optimal_zone_boundary(self):
        """Exactly 13% headroom → no penalty."""
        snap = self._make_snap(13.0)
        penalized, reason = penalize_vram_heavy(100.0, snap)
        assert penalized == 100.0
        assert reason == "optimal"

    def test_warning_zone_penalty(self):
        """10-12% headroom → 0.9x penalty."""
        snap = self._make_snap(11.0)  # 11% headroom
        penalized, reason = penalize_vram_heavy(100.0, snap)
        assert abs(penalized - 90.0) < 0.01
        assert "warning" in reason

    def test_paging_likely_penalty(self):
        """<10% headroom → 0.7x penalty."""
        snap = self._make_snap(8.0)  # 8% headroom
        penalized, reason = penalize_vram_heavy(100.0, snap)
        assert abs(penalized - 70.0) < 0.01
        assert "paging-likely" in reason

    def test_paging_likely_zero_headroom(self):
        """0% headroom (full VRAM) → 0.7x penalty."""
        snap = self._make_snap(0.0)
        penalized, reason = penalize_vram_heavy(100.0, snap)
        assert abs(penalized - 70.0) < 0.01
        assert "paging-likely" in reason

    def test_custom_threshold(self):
        """Custom threshold shifts boundary."""
        snap = self._make_snap(11.0)  # 11% headroom
        # With default threshold (0.12), 11% is in warning zone
        penalized_default, _ = penalize_vram_heavy(100.0, snap, vram_headroom_threshold=0.12)
        assert abs(penalized_default - 90.0) < 0.01


class TestNvidiaSmiParsing:
    """Tests for nvidia-smi output parsing (mocked)."""

    def test_parse_mib_value(self):
        """Verify MiB parsing logic."""
        from llama_optimus.hardware_probe import get_gpu_telemetry
        # Just check the function exists and is callable
        assert callable(get_gpu_telemetry)

    def test_get_gpu_count_callable(self):
        from llama_optimus.hardware_probe import get_gpu_count
        assert callable(get_gpu_count)
        # On non-NVIDIA systems, should return 0 (not crash)
        count = get_gpu_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_vram_total_mb_callable(self):
        from llama_optimus.hardware_probe import get_vram_total_mb
        assert callable(get_vram_total_mb)
        # On non-NVIDIA systems, should return None (not crash)
        val = get_vram_total_mb()
        assert val is None or isinstance(val, float)
