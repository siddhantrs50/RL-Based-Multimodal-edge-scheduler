"""
rl_scheduler.py — RL precision scheduler
                  Jetson Orin NX 8 GB / AVerMedia D115

Logs per iteration:
  timestamp, gpu_temp, cpu_temp, gpu_util, power, ram,
  system_state, decision, yolo_precision, bert_precision,
  yolo_latency_ms, bert_latency_ms, total_latency_ms,
  fps, throughput_tasks_per_sec, energy_per_inf_j,
  scheduler_overhead_ms, cost_function, temp_trend
"""

import time
import csv
import random
import numpy as np
import cv2
import psutil
from datetime import datetime
from stable_baselines3 import PPO
from ultralytics import YOLO

from bert_infer import BertTRTInference
from telemetry_reader import read_tegrastats, parse_tegrastats, classify_state


# ─────────────────────────────────────────────
# Thesis cost function  C = α(1−A) + βL + γE + δT
# ─────────────────────────────────────────────
ALPHA, BETA, GAMMA, DELTA = 0.4, 0.3, 0.2, 0.1

ACCURACY = {
    "YOLO_FP16+BERT_FP16": 1.00,
    "YOLO_INT8+BERT_FP16": 0.88,
    "YOLO_FP16+BERT_INT8": 0.90,
    "YOLO_INT8+BERT_INT8": 0.80,
}

ACTION_NAMES = {
    0: "YOLO_FP16+BERT_FP16",
    1: "YOLO_INT8+BERT_FP16",
    2: "YOLO_FP16+BERT_INT8",
    3: "YOLO_INT8+BERT_INT8",
}

def cost_function(decision, latency_ms, energy_j, temp_c):
    acc = ACCURACY.get(decision, 0.85)
    return (ALPHA * (1 - acc) +
            BETA  * (latency_ms / 1000.0) +
            GAMMA * energy_j +
            DELTA * (temp_c / 100.0))


# ─────────────────────────────────────────────
# Load RL model
# ─────────────────────────────────────────────
try:
    rl_model = PPO.load("./best_model/best_model")
    print("Loaded best model.")
except FileNotFoundError:
    rl_model = PPO.load("ppo_scheduler_final")
    print("Loaded final model.")

# ─────────────────────────────────────────────
# Load engines
# ─────────────────────────────────────────────
print("Loading YOLO engines...")
yolo_engines = {
    "fp16": YOLO("engines/yolov8_fp16.engine"),
    "int8": YOLO("engines/yolov8_int8.engine"),
}
print("Loading BERT engines...")
bert_engines = {
    "fp16": BertTRTInference("engines/bert_fp16.engine"),
    "int8": BertTRTInference("engines/bert_int8.engine"),
}
print("All engines loaded.\n")

# ─────────────────────────────────────────────
# Webcam
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("[WARN] Webcam not available — using random frames")
    cap = None

def get_frame():
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            return frame
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

TEXT_POOL = [
    "RL scheduling under thermal constraints on edge devices.",
    "Jetson Orin NX enables efficient multimodal AI inference.",
    "Dynamic model compression reduces power consumption at edge.",
    "Real-time telemetry guides adaptive precision switching decisions.",
    "Edge AI requires balancing latency and energy efficiency carefully.",
    "TensorRT INT8 quantization accelerates transformer inference significantly.",
    "Thermal-aware scheduling improves edge device operational longevity.",
    "Reinforcement learning optimizes scheduling decisions dynamically online.",
    "YOLOv8 combined with NLP inference on a single Jetson Orin NX board.",
    "Precision switching between FP16 and INT8 based on system telemetry.",
]

# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────
CSV_FILE = "rl_log.csv"
FIELDNAMES = [
    "timestamp", "gpu_temp", "cpu_temp", "gpu_util", "power", "ram",
    "system_state", "temp_trend", "decision", "yolo_precision", "bert_precision",
    "yolo_latency_ms", "bert_latency_ms", "total_latency_ms",
    "fps", "throughput_tasks_per_sec", "energy_per_inf_j",
    "scheduler_overhead_ms", "cost_function"
]
with open(CSV_FILE, "w", newline="") as f:
    csv.writer(f).writerow(FIELDNAMES)

YOLO_RUNS = 8
BERT_RUNS = 8


def run_yolo(precision):
    latencies = []
    for _ in range(YOLO_RUNS):
        frame = get_frame()
        t0 = time.time()
        yolo_engines[precision].predict(
            source=frame, device=0, verbose=False, imgsz=640)
        latencies.append((time.time() - t0) * 1000)
    return float(np.mean(latencies))


def run_bert(precision):
    latencies = []
    for _ in range(BERT_RUNS):
        t0 = time.time()
        bert_engines[precision].infer(random.choice(TEXT_POOL))
        latencies.append((time.time() - t0) * 1000)
    return float(np.mean(latencies))


def build_obs(state, prev_temp):
    temp  = state.get("gpu_temp_c",       50.0)
    gpu   = state.get("gpu_util_percent",  0.0)
    power = state.get("power_w",           5.0)
    ram   = state.get("ram_used_mb",    2000.0)
    trend = float(np.clip(temp - prev_temp, -15.0, 15.0))
    return np.array([temp, gpu, power, ram, trend], dtype=np.float32)


MAX_ITERATIONS = 100
prev_temp = 50.0
proc = psutil.Process()

print(f"RL Precision Scheduler")
print(f"Intensity: {YOLO_RUNS} YOLO + {BERT_RUNS} BERT per iteration\n")

try:
    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{MAX_ITERATIONS} ---")

        raw   = read_tegrastats()
        state = parse_tegrastats(raw)
        if not state:
            time.sleep(0.2)
            continue

        sys_state = classify_state(state)
        obs       = build_obs(state, prev_temp)

        # Measure RL inference overhead
        t_sched = time.time()
        action, _ = rl_model.predict(obs, deterministic=True)
        action    = int(action)
        sched_overhead_ms = (time.time() - t_sched) * 1000

        decision  = ACTION_NAMES[action]
        yolo_prec = "int8" if action in (1, 3) else "fp16"
        bert_prec = "int8" if action in (2, 3) else "fp16"

        yolo_lat  = run_yolo(yolo_prec)
        bert_lat  = run_bert(bert_prec)
        total_lat = yolo_lat + bert_lat

        fps        = 1000.0 / yolo_lat
        throughput = (YOLO_RUNS + BERT_RUNS) / (total_lat / 1000.0)
        power      = state.get("power_w", 0.0)
        temp_c     = state.get("gpu_temp_c", 50.0)
        energy_j   = (power * total_lat) / 1000.0
        cost       = cost_function(decision, total_lat, energy_j, temp_c)

        print(f"State     : {sys_state} | "
              f"GPU={state.get('gpu_temp_c',0):.1f}°C "
              f"CPU={state.get('cpu_temp_c',0):.1f}°C "
              f"GPU_UTIL={state.get('gpu_util_percent',0):.0f}% "
              f"PWR={power:.2f}W")
        print(f"Decision  : {decision} | Overhead: {sched_overhead_ms:.3f}ms")
        print(f"YOLO({yolo_prec.upper()}): {yolo_lat:.1f}ms | "
              f"BERT({bert_prec.upper()}): {bert_lat:.1f}ms | "
              f"FPS: {fps:.1f} | Energy: {energy_j:.4f}J | Cost: {cost:.4f}")

        prev_temp = temp_c

        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%H:%M:%S"),
                state.get("gpu_temp_c",       ""),
                state.get("cpu_temp_c",        ""),
                state.get("gpu_util_percent",  ""),
                state.get("power_w",           ""),
                state.get("ram_used_mb",       ""),
                sys_state,
                f"{obs[4]:.3f}",
                decision,
                yolo_prec, bert_prec,
                f"{yolo_lat:.2f}",
                f"{bert_lat:.2f}",
                f"{total_lat:.2f}",
                f"{fps:.2f}",
                f"{throughput:.2f}",
                f"{energy_j:.4f}",
                f"{sched_overhead_ms:.4f}",
                f"{cost:.4f}",
            ])

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nRL scheduler stopped.")
finally:
    if cap is not None:
        cap.release()

print(f"\nDone. Log: {CSV_FILE}")
