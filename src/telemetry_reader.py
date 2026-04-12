import subprocess
import re
import time
from datetime import datetime


# =============================
# READ ONE LINE OF tegrastats
# =============================
def read_tegrastats():
    """
    Runs tegrastats, captures one line of output,
    then immediately terminates the process.
    Compatible with Jetson tegrastats versions.
    """
    process = subprocess.Popen(
        ["tegrastats"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    line = process.stdout.readline().strip()
    process.terminate()
    return line


# =============================
# PARSE TELEMETRY VALUES
# =============================
def parse_tegrastats(output):
    """
    Parses RAM, GPU utilization, GPU temperature,
    and power from tegrastats output.
    """
    data = {}

    # RAM usage
    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", output)
    if ram_match:
        data["ram_used_mb"] = int(ram_match.group(1))
        data["ram_total_mb"] = int(ram_match.group(2))

    # GPU utilization
    gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", output)
    if gpu_match:
        data["gpu_util_percent"] = float(gpu_match.group(1))

    # GPU temperature
    temp_match = re.search(r"gpu@([\d.]+)C", output)
    if temp_match:
        data["gpu_temp_c"] = float(temp_match.group(1))

    # Power consumption
    power_match = re.search(r"VDD_IN\s+(\d+)mW", output)
    if power_match:
        data["power_w"] = int(power_match.group(1)) / 1000.0

    return data


# =============================
# EVENT THRESHOLDS
# =============================
THRESHOLDS = {
    "gpu_temp_c": 2.0,          # °C change
    "gpu_util_percent": 10.0,   # % change
    "power_w": 2.0,             # Watts change
    "ram_used_mb": 100          # MB change
}


# =============================
# CHECK FOR SIGNIFICANT CHANGE
# =============================
def significant_change(prev, curr):
    """
    Returns True if any telemetry value changes
    more than its defined threshold.
    """
    for key, threshold in THRESHOLDS.items():
        if key in prev and key in curr:
            if abs(curr[key] - prev[key]) >= threshold:
                return True
    return False


# =============================
# MAIN LOOP (EVENT-DRIVEN)
# =============================
if __name__ == "__main__":

    last_state = None

    print("\nStarting event-driven telemetry logging")
    print("Events triggered by changes in:")
    print("Temperature | GPU Utilization | Power | RAM")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            raw = read_tegrastats()
            current = parse_tegrastats(raw)

            if not current:
                time.sleep(1)
                continue

            if last_state is None or significant_change(last_state, current):

                # Timestamps
                human_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                epoch_time = time.time()

                print("EVENT DETECTED")
                print(f"Timestamp (human): {human_time}")
                # print(f"Timestamp (epoch): {epoch_time:.3f}")

                for k, v in current.items():
                    print(f"{k}: {v}")

                print("-" * 50)

                last_state = current

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nTelemetry logging stopped by user.")
