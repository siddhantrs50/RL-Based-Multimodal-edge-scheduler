import csv
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# LOAD FUNCTION
# -----------------------------
def load(file):
    temp, gpu, power, yolo, bert = [], [], [], [], []

    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            temp.append(float(row["gpu_temp"]))
            gpu.append(float(row["gpu_util"]))
            power.append(float(row["power"]))
            yolo.append(int(row["yolo_run"]))
            bert.append(int(row["bert_run"]))

    return np.array(temp), np.array(gpu), np.array(power), np.array(yolo), np.array(bert)

# -----------------------------
# LOAD ALL DATA
# -----------------------------
temp_n, gpu_n, power_n, _, _ = load("naive_log.csv")
temp_r, gpu_r, power_r, yolo_r, bert_r = load("rule_based_log.csv")
temp_rl, gpu_rl, power_rl, yolo_rl, bert_rl = load("rl_log.csv")

# -----------------------------
# 1. TEMPERATURE COMPARISON
# -----------------------------
plt.figure()
plt.plot(temp_n, label="Naive")
plt.plot(temp_r, label="Rule-Based")
plt.plot(temp_rl, label="RL")
plt.title("Temperature Comparison")
plt.xlabel("Iteration")
plt.ylabel("°C")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 2. POWER COMPARISON
# -----------------------------
plt.figure()
plt.plot(power_n, label="Naive")
plt.plot(power_r, label="Rule-Based")
plt.plot(power_rl, label="RL")
plt.title("Power Consumption")
plt.xlabel("Iteration")
plt.ylabel("Watts")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 3. GPU UTILIZATION
# -----------------------------
plt.figure()
plt.plot(gpu_n, label="Naive")
plt.plot(gpu_r, label="Rule-Based")
plt.plot(gpu_rl, label="RL")
plt.title("GPU Utilization")
plt.xlabel("Iteration")
plt.ylabel("%")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 4. SMOOTHED TEMPERATURE
# -----------------------------
def moving_avg(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure()
plt.plot(moving_avg(temp_n), label="Naive")
plt.plot(moving_avg(temp_r), label="Rule-Based")
plt.plot(moving_avg(temp_rl), label="RL")
plt.title("Smoothed Temperature")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 5. ENERGY CONSUMPTION
# -----------------------------
energy_n = np.cumsum(power_n)
energy_r = np.cumsum(power_r)
energy_rl = np.cumsum(power_rl)

plt.figure()
plt.plot(energy_n, label="Naive")
plt.plot(energy_r, label="Rule-Based")
plt.plot(energy_rl, label="RL")
plt.title("Cumulative Energy")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 6. BOX PLOT (VERY IMPORTANT)
# -----------------------------
plt.figure()
plt.boxplot([temp_n, temp_r, temp_rl], labels=["Naive", "Rule", "RL"])
plt.title("Temperature Variability")
plt.show()

# -----------------------------
# 7. EXECUTION COMPARISON
# -----------------------------
plt.figure()
plt.plot(yolo_rl, label="YOLO (RL)")
plt.plot(bert_rl, label="BERT (RL)")
plt.title("RL Execution Pattern")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 8. HISTOGRAM
# -----------------------------
plt.figure()
plt.hist(temp_n, alpha=0.4, label="Naive")
plt.hist(temp_r, alpha=0.4, label="Rule")
plt.hist(temp_rl, alpha=0.4, label="RL")
plt.title("Temperature Distribution")
plt.legend()
plt.show()

# -----------------------------
# 9. STATISTICS (VERY IMPORTANT)
# -----------------------------
print("\n===== FINAL RESULTS =====")

def stats(name, temp, power, gpu):
    print(f"\n{name}:")
    print(f"Temp Mean: {np.mean(temp):.2f}")
    print(f"Power Mean: {np.mean(power):.2f}")
    print(f"GPU Mean: {np.mean(gpu):.2f}")
    print(f"Temp Variance: {np.var(temp):.2f}")

stats("Naive", temp_n, power_n, gpu_n)
stats("Rule-Based", temp_r, power_r, gpu_r)
stats("RL", temp_rl, power_rl, gpu_rl)
