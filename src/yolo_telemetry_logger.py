import subprocess
import re
import time
import csv
from datetime import datetime

CSV_FILE = "yolo_telemetry_log.csv"

# -----------------------------
# READ ONE LINE OF tegrastats
# -----------------------------
def read_tegrastats():
    process = subprocess.Popen(
        ["tegrastats"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    line = process.stdout.readline().strip()
    process.terminate()
    return line


# -----------------------------
# PARSE TELEMETRY
# -----------------------------
def parse_tegrastats(output):
    data = {}

    ram = re.search(r"RAM\s+(\d+)/(\d+)MB", output)
    if ram:
        data["ram_used_mb"] = int(ram.group(1))
        data["ram_total_mb"] = int(ram.group(2))

    gpu = re.search(r"GR3D_FREQ\s+(\d+)%", output)
    if gpu:
        data["gpu_util_percent"] = float(gpu.group(1))

    temp = re.search(r"gpu@([\d\.]+)C", output)
    if temp:
        data["gpu_temp_c"] = float(temp.group(1))

    power = re.search(r"VDD_IN\s+(\d+)mW", output)
    if power:
        data["power_w"] = int(power.group(1)) / 1000.0

    return data


# -----------------------------
# EVENT THRESHOLDS
# -----------------------------
THRESHOLDS = {
    "gpu_temp_c": 2.0,
    "gpu_util_percent": 15.0,
    "power_w": 2.0,
    "ram_used_mb": 300
}


def significant_change(prev, curr):
    for k, th in THRESHOLDS.items():
        if k in prev and k in curr:
            if abs(curr[k] - prev[k]) >= th:
                return True
    return False


# -----------------------------
# RUN YOLO & MEASURE LATENCY
# -----------------------------
def run_yolo_and_measure():
    """
    Runs a single YOLO inference and extracts latency.
    """
    start = time.time()

    subprocess.run(
        [
            "yolo",
            "predict",
            "model=yolov8n.pt",
            "source=https://ultralytics.com/images/bus.jpg",
            "device=0",
            "show=False",
            "save=False"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    end = time.time()
    latency_ms = (end - start) * 1000
    return latency_ms


# -----------------------------
# MAIN LOGGER
# -----------------------------
if __name__ == "__main__":

    last_state = None

    # Create CSV with header if not exists
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_human",
            "timestamp_epoch",
            "gpu_temp_c",
            "gpu_util_percent",
            "power_w",
            "ram_used_mb",
            "yolo_latency_ms"
        ])

    print("\nLogging YOLO + telemetry to CSV")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            raw = read_tegrastats()
            telemetry = parse_tegrastats(raw)

            if not telemetry:
                time.sleep(1)
                continue

            if last_state is None or significant_change(last_state, telemetry):

                # Measure YOLO latency
                latency = run_yolo_and_measure()

                # Timestamps
                human_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                epoch_time = time.time()

                # Write CSV row
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        human_time,
                        epoch_time,
                        telemetry.get("gpu_temp_c"),
                        telemetry.get("gpu_util_percent"),
                        telemetry.get("power_w"),
                        telemetry.get("ram_used_mb"),
                        latency
                    ])

                print(f"[{human_time}] EVENT LOGGED | YOLO latency: {latency:.2f} ms")

                last_state = telemetry

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nLogging stopped. CSV saved as:", CSV_FILE)
