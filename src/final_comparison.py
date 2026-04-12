import csv
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
def load(file):
    temp, gpu, power, yolo, bert, decision = [], [], [], [], [], []

    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            temp.append(float(row["gpu_temp"]))
            gpu.append(float(row["gpu_util"]))
            power.append(float(row["power"]))
            yolo.append(int(row["yolo_run"]))
            bert.append(int(row["bert_run"]))
            decision.append(row["decision"])

    return np.array(temp), np.array(gpu), np.array(power), np.array(yolo), np.array(bert), decision


# -----------------------------
# LOAD FILES
# -----------------------------
temp_n, gpu_n, power_n, _, _, _ = load("naive_log.csv")
temp_r, gpu_r, power_r, yolo_r, bert_r, decision_r = load("rule_based_log.csv")

# -----------------------------
# 1. TEMPERATURE TREND
# -----------------------------
plt.figure()
plt.plot(temp_n, label="Naive")
plt.plot(temp_r, label="Rule-Based")
plt.title("Temperature Trend")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 2. MOVING AVERAGE (SMOOTH)
# -----------------------------
def moving_avg(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure()
plt.plot(moving_avg(temp_n), label="Naive")
plt.plot(moving_avg(temp_r), label="Rule-Based")
plt.title("Smoothed Temperature")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 3. POWER COMPARISON
# -----------------------------
plt.figure()
plt.plot(power_n, label="Naive")
plt.plot(power_r, label="Rule-Based")
plt.title("Power Comparison")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 4. GPU UTILIZATION
# -----------------------------
plt.figure()
plt.plot(gpu_n, label="Naive")
plt.plot(gpu_r, label="Rule-Based")
plt.title("GPU Utilization")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 5. HISTOGRAM (DISTRIBUTION)
# -----------------------------
plt.figure()
plt.hist(temp_n, alpha=0.5, label="Naive")
plt.hist(temp_r, alpha=0.5, label="Rule-Based")
plt.title("Temperature Distribution")
plt.legend()
plt.show()

# -----------------------------
# 6. BOX PLOT (VERY IMPORTANT)
# -----------------------------
plt.figure()
plt.boxplot([temp_n, temp_r], labels=["Naive", "Rule-Based"])
plt.title("Temperature Variability (Box Plot)")
plt.show()

# -----------------------------
# 7. CUMULATIVE ENERGY
# -----------------------------
energy_n = np.cumsum(power_n)
energy_r = np.cumsum(power_r)

plt.figure()
plt.plot(energy_n, label="Naive")
plt.plot(energy_r, label="Rule-Based")
plt.title("Cumulative Energy Consumption")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 8. EXECUTION PATTERN
# -----------------------------
plt.figure()
plt.plot(yolo_r, label="YOLO")
plt.plot(bert_r, label="BERT")
plt.title("Execution Pattern")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 9. DECISION DISTRIBUTION
# -----------------------------
from collections import Counter

counts = Counter(decision_r)

plt.figure()
plt.bar(counts.keys(), counts.values())
plt.title("Scheduler Decision Distribution")
plt.xticks(rotation=30)
plt.show()

# -----------------------------
# 10. STATISTICS PRINT
# -----------------------------
print("\n===== STATISTICAL SUMMARY =====")

print("\nTemperature:")
print(f"Naive Mean: {temp_n.mean():.2f}")
print(f"Rule Mean: {temp_r.mean():.2f}")

print("\nPower:")
print(f"Naive Mean: {power_n.mean():.2f}")
print(f"Rule Mean: {power_r.mean():.2f}")

print("\nGPU Util:")
print(f"Naive Mean: {gpu_n.mean():.2f}")
print(f"Rule Mean: {gpu_r.mean():.2f}")

print("\nVariance (Stability):")
print(f"Temp Variance Naive: {np.var(temp_n):.2f}")
print(f"Temp Variance Rule: {np.var(temp_r):.2f}")
