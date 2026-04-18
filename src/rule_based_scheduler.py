"""
rule_based_scheduler.py — Threshold-driven precision scheduler
                           Jetson Orin NX 8 GB / AVerMedia D115

Logs per iteration:
  timestamp, gpu_temp, cpu_temp, gpu_util, power, ram,
  system_state, decision, yolo_precision, bert_precision,
  yolo_latency_ms, bert_latency_ms, total_latency_ms,
  fps, throughput_tasks_per_sec, energy_per_inf_j,
  scheduler_overhead_ms, cost_function
"""

import time
import csv
import random
import numpy as np
import cv2
import psutil
from datetime import datetime
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

def cost_function(decision, latency_ms, energy_j, temp_c):
    acc = ACCURACY.get(decision, 0.85)
    return (ALPHA * (1 - acc) +
            BETA  * (latency_ms / 1000.0) +
            GAMMA * energy_j +
            DELTA * (temp_c / 100.0))


# ─────────────────────────────────────────────
# Load all 4 engines
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
    "Rule-based scheduling uses fixed thresholds for precision decisions.",
    "Thermal limits trigger precision downgrade on Jetson Orin NX board.",
    "Edge AI systems need adaptive inference under resource constraints.",
    "INT8 quantization reduces power consumption significantly at edge.",
    "TensorRT engines enable fast switching between FP16 and INT8 modes.",
    "Dynamic model compression improves energy efficiency on embedded boards.",
    "Multimodal scheduling balances vision and language workloads well.",
    "Real-time telemetry guides adaptive precision switching decisions.",
    "YOLOv8 object detection runs efficiently with TensorRT optimization.",
    "DistilBERT provides lightweight NLP inference for edge deployment.",
]

# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────
CSV_FILE = "rule_based_log.csv"
FIELDNAMES = [
    "timestamp", "gpu_temp", "cpu_temp", "gpu_util", "power", "ram",
    "system_state", "decision", "yolo_precision", "bert_precision",
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


def decide(state):
    """
    Thesis Phase 3 rule-based decision logic.
    Returns (decision_label, yolo_precision, bert_precision)
    """
    temp  = state.get("gpu_temp_c",       0.0)
    cpu_t = state.get("cpu_temp_c",        0.0)
    gpu   = state.get("gpu_util_percent",  0.0)
    power = state.get("power_w",           0.0)

    # Critical state → full INT8
    if temp > 75.0 or cpu_t > 75.0 or power > 10.0:
        return "YOLO_INT8+BERT_INT8",  "int8", "int8"
    # High Load → reduce heavier model (YOLO) first
    elif temp > 65.0 or cpu_t > 65.0 or gpu > 85.0:
        return "YOLO_INT8+BERT_FP16", "int8", "fp16"
    # Moderate load → reduce lighter model (BERT)
    elif gpu > 70.0:
        return "YOLO_FP16+BERT_INT8", "fp16", "int8"
    # Normal → full FP16
    else:
        return "YOLO_FP16+BERT_FP16", "fp16", "fp16"


MAX_ITERATIONS = 100
proc = psutil.Process()

print(f"Rule-Based Precision Scheduler")
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

        # Measure scheduler overhead
        t_sched = time.time()
        decision, yolo_prec, bert_prec = decide(state)
        sched_overhead_ms = (time.time() - t_sched) * 1000

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

        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%H:%M:%S"),
                state.get("gpu_temp_c",       ""),
                state.get("cpu_temp_c",        ""),
                state.get("gpu_util_percent",  ""),
                state.get("power_w",           ""),
                state.get("ram_used_mb",       ""),
                sys_state,
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
    print("\nRule-based scheduler stopped.")
finally:
    if cap is not None:
        cap.release()

print(f"\nDone. Log: {CSV_FILE}")
