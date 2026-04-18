"""
naive_scheduler.py — Always FP32 baseline scheduler
                     Jetson Orin NX 8 GB / AVerMedia D115

Logs per iteration:
  timestamp, gpu_temp, cpu_temp, gpu_util, power, ram,
  system_state, decision, yolo_precision, bert_precision,
  yolo_latency_ms, bert_latency_ms, total_latency_ms,
  fps, throughput_tasks_per_sec, energy_per_inf_j,
  scheduler_overhead_ms
"""

import time
import csv
import random
import numpy as np
import cv2
import psutil
from datetime import datetime
import torch
from ultralytics import YOLO
from transformers import DistilBertTokenizer, DistilBertModel

from telemetry_reader import read_tegrastats, parse_tegrastats, classify_state


# ─────────────────────────────────────────────
# Thesis cost function weights (α, β, γ, δ)
# C(mi,t) = α(1−Ai) + β·Li + γ·Ei + δ·Ti
# ─────────────────────────────────────────────
ALPHA, BETA, GAMMA, DELTA = 0.4, 0.3, 0.2, 0.1

# Accuracy proxy for FP32 baseline
ACCURACY = {"fp32": 1.00}

def cost_function(accuracy, latency_ms, energy_j, temp_c):
    return (ALPHA * (1 - accuracy) +
            BETA  * (latency_ms / 1000.0) +
            GAMMA * energy_j +
            DELTA * (temp_c / 100.0))


# ─────────────────────────────────────────────
# Load FP32 PyTorch models
# ─────────────────────────────────────────────
print("Loading YOLOv8m FP32 (PyTorch)...")
yolo_model = YOLO("yolov8m.pt")

print("Loading DistilBERT FP32 (PyTorch)...")
tokenizer  = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()
print(f"Models loaded on {device}\n")

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
    "Naive scheduling always uses full precision FP32 models.",
    "Static FP32 deployment ignores runtime thermal conditions.",
    "Full precision inference provides the highest accuracy baseline.",
    "Jetson Orin NX runs FP32 PyTorch models without quantisation.",
    "Always-on FP32 is the unoptimised reference for comparison.",
    "Dynamic model compression improves energy efficiency significantly.",
    "Multimodal scheduling balances vision and language workloads.",
    "Real-time telemetry enables adaptive precision switching decisions.",
    "YOLOv8 object detection runs on live webcam feed from camera.",
    "DistilBERT provides lightweight NLP inference for edge devices.",
]

# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────
CSV_FILE = "naive_log.csv"
FIELDNAMES = [
    "timestamp", "gpu_temp", "cpu_temp", "gpu_util", "power", "ram",
    "system_state", "decision", "yolo_precision", "bert_precision",
    "yolo_latency_ms", "bert_latency_ms", "total_latency_ms",
    "fps", "throughput_tasks_per_sec", "energy_per_inf_j", "cost_function"
]
with open(CSV_FILE, "w", newline="") as f:
    csv.writer(f).writerow(FIELDNAMES)

YOLO_RUNS = 8
BERT_RUNS = 8


def run_yolo():
    latencies = []
    for _ in range(YOLO_RUNS):
        frame = get_frame()
        t0 = time.time()
        yolo_model.predict(source=frame, device=0, verbose=False, imgsz=640)
        latencies.append((time.time() - t0) * 1000)
    return float(np.mean(latencies))


def run_bert():
    latencies = []
    for _ in range(BERT_RUNS):
        text   = random.choice(TEXT_POOL)
        inputs = tokenizer(text, return_tensors="pt", max_length=128,
                           truncation=True, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            _ = bert_model(**inputs)
        latencies.append((time.time() - t0) * 1000)
    return float(np.mean(latencies))


MAX_ITERATIONS = 100
proc = psutil.Process()

print(f"NAIVE scheduler — always FP32")
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

        # Scheduler overhead — naive has no decision logic, measure baseline
        t_sched_start = time.time()
        decision   = "NAIVE_FP32+FP32"
        yolo_prec  = "fp32"
        bert_prec  = "fp32"
        sched_overhead_ms = (time.time() - t_sched_start) * 1000

        yolo_lat   = run_yolo()
        bert_lat   = run_bert()
        total_lat  = yolo_lat + bert_lat

        fps        = 1000.0 / yolo_lat
        throughput = (YOLO_RUNS + BERT_RUNS) / (total_lat / 1000.0)
        power      = state.get("power_w", 0.0)
        temp_c     = state.get("gpu_temp_c", 50.0)
        energy_j   = (power * total_lat) / 1000.0
        cost       = cost_function(ACCURACY["fp32"], total_lat, energy_j, temp_c)

        print(f"State     : {sys_state} | "
              f"GPU={state.get('gpu_temp_c',0):.1f}°C "
              f"CPU={state.get('cpu_temp_c',0):.1f}°C "
              f"GPU_UTIL={state.get('gpu_util_percent',0):.0f}% "
              f"PWR={power:.2f}W")
        print(f"YOLO(FP32): {yolo_lat:.1f}ms | "
              f"BERT(FP32): {bert_lat:.1f}ms | "
              f"FPS: {fps:.1f} | "
              f"Energy: {energy_j:.4f}J | "
              f"Cost: {cost:.4f}")

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
                f"{cost:.4f}",
            ])

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nNaive scheduler stopped.")
finally:
    if cap is not None:
        cap.release()

print(f"\nDone. Log: {CSV_FILE}")
