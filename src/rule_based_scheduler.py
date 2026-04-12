import time
import threading
import subprocess
import csv
import random
from datetime import datetime
import torch
from transformers import DistilBertTokenizer, DistilBertModel

from telemetry_reader import read_tegrastats, parse_tegrastats

# -----------------------------
# CSV SETUP
# -----------------------------
CSV_FILE = "rule_based_log.csv"

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
# LOAD BERT (ONLY ONCE)
# -----------------------------
print("Loading DistilBERT...")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("BERT loaded successfully\n")

# -----------------------------
# TEXT POOL (RANDOM NLP LOAD)
# -----------------------------
TEXT_POOL = [
    "Edge AI systems require efficient scheduling under constraints.",
    "Real time object detection is computationally intensive.",
    "Transformers are widely used for natural language processing tasks.",
    "Jetson devices enable deployment of deep learning models at the edge.",
    "Thermal management is critical in embedded AI systems.",
    "Multimodal workloads combine vision and language processing.",
    "Reinforcement learning can optimize scheduling decisions dynamically.",
    "Latency and power consumption must be balanced carefully.",
    "Efficient inference pipelines are required for edge deployment.",
    "Resource constrained devices require intelligent workload management."
]

# -----------------------------
# YOLO (WEBCAM BURST MODE)
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
# BERT (RANDOMIZED LOAD)
# -----------------------------
def run_bert():
    text = random.choice(TEXT_POOL)
    seq_len = random.choice([32, 64, 128])

    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

# -----------------------------
# IMPROVED SCHEDULER LOGIC
# -----------------------------
def scheduler_decision(state):
    temp = state.get("gpu_temp_c", 0)
    gpu = state.get("gpu_util_percent", 0)
    ram = state.get("ram_used_mb", 0)

    # Exploration (prevents starvation)
    if random.random() < 0.2:
        return "RUN_BOTH"

    if temp > 75:
        return "SKIP_YOLO"
    elif gpu > 95:
        return "DELAY_BERT"
    elif ram > 5000:
        return "DELAY_BERT"
    else:
        return "RUN_BOTH"

# -----------------------------
# MAIN LOOP (100 ITERATIONS)
# -----------------------------
if __name__ == "__main__":

    MAX_ITERATIONS = 100

    print("Starting Rule-Based Scheduler (100 iterations)\n")

    try:
        for iteration in range(MAX_ITERATIONS):

            print(f"\n--- Iteration {iteration+1}/{MAX_ITERATIONS} ---")

            # -----------------------------
            # READ TELEMETRY
            # -----------------------------
            raw = read_tegrastats()
            state = parse_tegrastats(raw)

            if not state:
                time.sleep(1)
                continue

            print(f"State: {state}")

            # -----------------------------
            # DECISION
            # -----------------------------
            decision = scheduler_decision(state)
            print(f"Decision: {decision}")

            yolo_run = 0
            bert_run = 0

            # -----------------------------
            # APPLY DECISION
            # -----------------------------
            if decision == "SKIP_YOLO":
                print("→ Running BERT only")
                run_bert()
                bert_run = 1

            elif decision == "DELAY_BERT":
                print("→ Running YOLO only")
                run_yolo()
                yolo_run = 1

            elif decision == "RUN_BOTH":
                print("→ Running YOLO + BERT")

                yolo_thread = threading.Thread(target=run_yolo)
                bert_thread = threading.Thread(target=run_bert)

                yolo_thread.start()
                bert_thread.start()

                yolo_thread.join()
                bert_thread.join()

                yolo_run = 1
                bert_run = 1

            # -----------------------------
            # LOG CSV
            # -----------------------------
            timestamp = datetime.now().strftime("%H:%M:%S")

            temp = state.get("gpu_temp_c", 0)
            gpu = state.get("gpu_util_percent", 0)
            power = state.get("power_w", 0)
            ram = state.get("ram_used_mb", 0)

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

        print("\nCompleted 100 iterations successfully.")
        print(f"CSV saved as: {CSV_FILE}")

    except KeyboardInterrupt:
        print("\nScheduler stopped early.")
