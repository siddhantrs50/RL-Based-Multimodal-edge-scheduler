import csv
import matplotlib.pyplot as plt
from datetime import datetime

CSV_FILE = "yolo_telemetry_log.csv"

# Lists to store data
timestamps = []
gpu_temp = []
gpu_util = []
power = []
latency = []

# -----------------------------
# READ CSV
# -----------------------------
with open(CSV_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamps.append(
            datetime.strptime(row["timestamp_human"], "%Y-%m-%d %H:%M:%S")
        )
        gpu_temp.append(float(row["gpu_temp_c"]))
        gpu_util.append(float(row["gpu_util_percent"]))
        power.append(float(row["power_w"]))
        latency.append(float(row["yolo_latency_ms"]))

# -----------------------------
# PLOT 1 — GPU TEMPERATURE
# -----------------------------
plt.figure()
plt.plot(timestamps, gpu_temp)
plt.xlabel("Time")
plt.ylabel("GPU Temperature (°C)")
plt.title("GPU Temperature vs Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT 2 — POWER
# -----------------------------
plt.figure()
plt.plot(timestamps, power)
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.title("Power Consumption vs Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT 3 — GPU UTILIZATION
# -----------------------------
plt.figure()
plt.plot(timestamps, gpu_util)
plt.xlabel("Time")
plt.ylabel("GPU Utilization (%)")
plt.title("GPU Utilization vs Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT 4 — YOLO LATENCY
# -----------------------------
plt.figure()
plt.plot(timestamps, latency)
plt.xlabel("Time")
plt.ylabel("YOLO Latency (ms)")
plt.title("YOLO Inference Latency vs Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
