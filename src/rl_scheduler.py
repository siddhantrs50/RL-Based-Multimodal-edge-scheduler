import time
import threading
import subprocess
import csv
from datetime import datetime
import torch
from stable_baselines3 import PPO

from telemetry_reader import read_tegrastats, parse_tegrastats

# -----------------------------
# CSV SETUP
# -----------------------------
CSV_FILE = "rl_log.csv"

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "gpu_temp",
        "gpu_util",
        "power",
        "ram",
        "decision",
        "yolo_run",
        "bert_run"
    ])

# -----------------------------
# LOAD RL MODEL
# -----------------------------
model = PPO.load("ppo_scheduler")

# -----------------------------
# LOAD BERT
# -----------------------------
from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# -----------------------------
# YOLO FUNCTION
# -----------------------------
def run_yolo():
    process = subprocess.Popen(
        [
            "yolo",
            "predict",
            "model=yolov8n.pt",
            "source=0",
            "device=0",
            "show=False",
            "save=False",
            "stream=True"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3)
    process.terminate()

# -----------------------------
# BERT FUNCTION
# -----------------------------
def run_bert():
    text = "Adaptive scheduling improves system efficiency."
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = bert_model(**inputs)

# -----------------------------
# MAIN LOOP
# -----------------------------
MAX_ITERATIONS = 100

print("Starting RL Scheduler with Logging...\n")

for i in range(MAX_ITERATIONS):

    print(f"\n--- RL Iteration {i+1}/{MAX_ITERATIONS} ---")

    # -----------------------------
    # READ TELEMETRY
    # -----------------------------
    raw = read_tegrastats()
    state_dict = parse_tegrastats(raw)

    if not state_dict:
        time.sleep(1)
        continue

    # Convert state
    state = [
        state_dict.get("gpu_temp_c", 0),
        state_dict.get("gpu_util_percent", 0),
        state_dict.get("power_w", 0),
        state_dict.get("ram_used_mb", 0),
    ]

    print(f"State: {state}")

    # -----------------------------
    # RL DECISION
    # -----------------------------
    action, _ = model.predict(state)

    yolo_run = 0
    bert_run = 0

    if action == 0:
        decision = "RUN_BOTH"
        print("→ RL: RUN BOTH")

        t1 = threading.Thread(target=run_yolo)
        t2 = threading.Thread(target=run_bert)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        yolo_run = 1
        bert_run = 1

    elif action == 1:
        decision = "SKIP_YOLO"
        print("→ RL: SKIP YOLO")

        run_bert()
        bert_run = 1

    elif action == 2:
        decision = "DELAY_BERT"
        print("→ RL: DELAY BERT")

        run_yolo()
        yolo_run = 1

    # -----------------------------
    # LOG CSV
    # -----------------------------
    timestamp = datetime.now().strftime("%H:%M:%S")

    temp = state_dict.get("gpu_temp_c", 0)
    gpu = state_dict.get("gpu_util_percent", 0)
    power = state_dict.get("power_w", 0)
    ram = state_dict.get("ram_used_mb", 0)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            temp,
            gpu,
            power,
            ram,
            decision,
            yolo_run,
            bert_run
        ])

    time.sleep(1)

print("\nRL Scheduler completed.")
print(f"CSV saved as: {CSV_FILE}")
