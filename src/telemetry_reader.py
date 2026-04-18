"""
telemetry_reader.py — Enhanced telemetry for Jetson Orin NX 8 GB

Additions vs original:
  - CPU temperature parsed from tegrastats (cpu@ pattern)
  - psutil fallback for CPU temp and RAM when tegrastats parse fails
  - System state classification: Normal / High Load / Critical
    matching thesis thresholds exactly
  - All values returned in a single unified dict
"""

import subprocess
import re
import time
import psutil
from datetime import datetime


# ─────────────────────────────────────────────
# Thesis-defined thresholds (Section: Phase 3)
# ─────────────────────────────────────────────
THRESHOLDS = {
    "gpu_temp_c":       {"high": 65.0,  "critical": 75.0},
    "cpu_temp_c":       {"high": 65.0,  "critical": 75.0},
    "gpu_util_percent": {"high": 85.0,  "critical": 95.0},
    "power_w":          {"high": 8.0,   "critical": 10.0},
    "ram_used_mb":      {"high": 5000.0,"critical": 6500.0},
}


def read_tegrastats() -> str:
    """Capture one line from tegrastats and terminate."""
    try:
        process = subprocess.Popen(
            ["tegrastats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        line = process.stdout.readline().strip()
        process.terminate()
        process.wait(timeout=2)
        return line
    except Exception as e:
        print(f"[WARN] tegrastats failed: {e}")
        return ""


def parse_tegrastats(output: str) -> dict:
    """
    Parse tegrastats output into a unified telemetry dict.
    Falls back to psutil for any missing values.
    """
    data = {}

    # RAM
    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", output)
    if ram_match:
        data["ram_used_mb"]  = int(ram_match.group(1))
        data["ram_total_mb"] = int(ram_match.group(2))

    # GPU utilization
    gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", output)
    if gpu_match:
        data["gpu_util_percent"] = float(gpu_match.group(1))

    # GPU temperature — Orin NX patterns
    for pattern in [r"gpu@([\d.]+)C", r"GPU@([\d.]+)C", r"SOC@([\d.]+)C"]:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            data["gpu_temp_c"] = float(m.group(1))
            break

    # CPU temperature — Orin NX patterns
    for pattern in [r"cpu@([\d.]+)C", r"CPU@([\d.]+)C", r"cpu0@([\d.]+)C"]:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            data["cpu_temp_c"] = float(m.group(1))
            break

    # Power — Orin NX may use VDD_IN or VDD_CPU_GPU_CV
    for pattern in [r"VDD_IN\s+(\d+)mW", r"VDD_CPU_GPU_CV\s+(\d+)/\d+mW"]:
        m = re.search(pattern, output)
        if m:
            data["power_w"] = int(m.group(1)) / 1000.0
            break

    # ── psutil fallbacks ──────────────────────────────────────────────
    if "ram_used_mb" not in data:
        vm = psutil.virtual_memory()
        data["ram_used_mb"]  = int(vm.used  / (1024 * 1024))
        data["ram_total_mb"] = int(vm.total / (1024 * 1024))

    if "gpu_util_percent" not in data:
        data["gpu_util_percent"] = 0.0   # psutil cannot read GPU util

    # CPU temp via psutil (reads /sys thermal zones on Jetson)
    if "cpu_temp_c" not in data:
        try:
            temps = psutil.sensors_temperatures()
            # Jetson thermal zones: "thermal-fan-est", "CPU-therm", etc.
            for key in ["thermal_fan_est", "CPU-therm", "cpu-thermal",
                        "coretemp", "k10temp"]:
                if key in temps:
                    data["cpu_temp_c"] = temps[key][0].current
                    break
            if "cpu_temp_c" not in data and temps:
                # Take first available sensor
                first_key = next(iter(temps))
                data["cpu_temp_c"] = temps[first_key][0].current
        except Exception:
            data["cpu_temp_c"] = 0.0

    if "power_w" not in data:
        data["power_w"] = 0.0

    # ── Warn if parse produced empty result ───────────────────────────
    if not output and not data.get("ram_used_mb"):
        print(f"[WARN] Empty tegrastats output and psutil fallback used.")

    return data


def classify_state(data: dict) -> str:
    """
    Classify system state as Normal / High Load / Critical.
    Matches thesis Phase 3 decision logic exactly.

    Critical:  any metric exceeds critical threshold
    High Load: any metric exceeds high threshold
    Normal:    all metrics within safe range
    """
    if not data:
        return "Unknown"

    # Critical conditions (thesis: power > 10W or temp > 75°C)
    if (data.get("gpu_temp_c",  0) > THRESHOLDS["gpu_temp_c"]["critical"] or
        data.get("cpu_temp_c",  0) > THRESHOLDS["cpu_temp_c"]["critical"] or
        data.get("power_w",     0) > THRESHOLDS["power_w"]["critical"]    or
        data.get("gpu_util_percent", 0) > THRESHOLDS["gpu_util_percent"]["critical"]):
        return "Critical"

    # High Load (thesis: GPU > 85% or temp > 65°C)
    if (data.get("gpu_temp_c",  0) > THRESHOLDS["gpu_temp_c"]["high"] or
        data.get("cpu_temp_c",  0) > THRESHOLDS["cpu_temp_c"]["high"] or
        data.get("power_w",     0) > THRESHOLDS["power_w"]["high"]    or
        data.get("gpu_util_percent", 0) > THRESHOLDS["gpu_util_percent"]["high"]):
        return "High Load"

    return "Normal"


def significant_change(prev: dict, curr: dict) -> bool:
    """Returns True if any telemetry value changed significantly."""
    delta_thresholds = {
        "gpu_temp_c":       2.0,
        "cpu_temp_c":       2.0,
        "gpu_util_percent": 10.0,
        "power_w":          1.5,
        "ram_used_mb":      100,
    }
    for key, threshold in delta_thresholds.items():
        if key in prev and key in curr:
            if abs(curr[key] - prev[key]) >= threshold:
                return True
    return False


# ─────────────────────────────────────────────
# Standalone event-driven monitor
# ─────────────────────────────────────────────
if __name__ == "__main__":
    last_state = None
    print("\nEvent-driven telemetry monitor — Jetson Orin NX")
    print("Tracking: GPU temp | CPU temp | GPU util | Power | RAM | State")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            raw     = read_tegrastats()
            current = parse_tegrastats(raw)
            state   = classify_state(current)

            if not current:
                time.sleep(1)
                continue

            if last_state is None or significant_change(last_state, current):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"STATE={state:10s} | "
                      f"GPU={current.get('gpu_temp_c',0):.1f}°C | "
                      f"CPU={current.get('cpu_temp_c',0):.1f}°C | "
                      f"GPU_UTIL={current.get('gpu_util_percent',0):.0f}% | "
                      f"PWR={current.get('power_w',0):.2f}W | "
                      f"RAM={current.get('ram_used_mb',0)}MB")
                last_state = current

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nTelemetry monitor stopped.")
