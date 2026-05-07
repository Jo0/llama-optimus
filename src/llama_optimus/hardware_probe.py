# hardware_probe.py
# nvidia-smi telemetry integration and hardware detection

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GpuTelemetrySnapshot:
    """A point-in-time snapshot of GPU state."""
    vram_used_mb: float
    vram_total_mb: float
    temperature_c: float
    gpu_utilization_pct: float
    gpu_index: int = 0

    @property
    def vram_headroom_pct(self) -> float:
        """Percentage of VRAM remaining."""
        return (
            (1.0 - self.vram_used_mb / self.vram_total_mb) * 100.0
            if self.vram_total_mb > 0
            else 0.0
        )


def get_gpu_count() -> int:
    """Return the number of NVIDIA GPUs detected by nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return 0
        # Each line corresponds to one GPU
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0


def get_gpu_telemetry(gpu_index: int = 0) -> Optional[GpuTelemetrySnapshot]:
    """Capture a point-in-time GPU telemetry snapshot.

    Runs: nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu
          --format=csv,noheader --id=<gpu_index>

    Returns:
        GpuTelemetrySnapshot on success, None if nvidia-smi fails (e.g., AMD/Intel GPU).
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
        "--format=csv,noheader",
        f"--id={gpu_index}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None

        def parse_mib(s: str) -> float:
            """Parse a value like '21456 MiB' into a float."""
            return float(s.replace("MiB", "").strip())

        def parse_temp(s: str) -> float:
            """Parse a value like '68 C' into a float."""
            return float(s.replace("C", "").strip())

        def parse_pct(s: str) -> float:
            """Parse a value like '83 %' into a float."""
            return float(s.replace("%", "").strip())

        return GpuTelemetrySnapshot(
            vram_used_mb=parse_mib(parts[0]),
            vram_total_mb=parse_mib(parts[1]),
            temperature_c=parse_temp(parts[2]),
            gpu_utilization_pct=parse_pct(parts[3]),
            gpu_index=gpu_index,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
        return None


def get_vram_total_mb(gpu_index: int = 0) -> Optional[float]:
    """Return total GPU VRAM in megabytes. Convenience wrapper."""
    snap = get_gpu_telemetry(gpu_index)
    return snap.vram_total_mb if snap else None


def penalize_vram_heavy(
    tokens_per_sec: float,
    post_snap: GpuTelemetrySnapshot,
    vram_headroom_threshold: float = 0.12,
) -> tuple[float, str]:
    """Apply VRAM-based penalization to throughput score.

    Heuristic (from Qwen3.6-27B optimization):
    - ≥13% headroom: no penalty (optimal zone)
    - 10-12% headroom: 0.9x penalty (warning zone)
    - <10% headroom: 0.7x penalty (paging likely)

    Args:
        tokens_per_sec: Raw throughput measured by llama-bench.
        post_snap: GPU telemetry snapshot captured after the benchmark run.
        vram_headroom_threshold: Fraction (0.0-1.0) representing the warning
            threshold. Default 0.12 = 12%.

    Returns:
        (penalized_tokens_per_sec, reason_string)
    """
    headroom = post_snap.vram_headroom_pct / 100.0  # convert percentage to fraction

    if headroom >= 0.13:
        return tokens_per_sec, "optimal"
    elif headroom >= (vram_headroom_threshold - 0.02):  # warning zone: threshold - 2%
        penalized = tokens_per_sec * 0.9
        return penalized, f"warning (headroom={headroom:.1%})"
    else:
        penalized = tokens_per_sec * 0.7
        return penalized, f"paging-likely (headroom={headroom:.1%})"
