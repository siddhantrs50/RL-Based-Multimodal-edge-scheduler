import threading
import time
import torch
import subprocess
import csv
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertModel

from telemetry_reader import read_tegrastats, parse_tegrastats

# -----------------------------
# CSV SETUP
# -----------------------------
CSV_FILE = "naive_log.csv"

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
# LOAD BERT ONCE
# -----------------------------
print("Loading BERT...")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("BERT loaded\n")

# -----------------------------
# YOLO (WEBCAM BURST)
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
# BERT
# -----------------------------
def run_bert():
    text = "Running multimodal workload on edge device."

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

# -----------------------------
# MAIN LOOP (100 ITERATIONS)
# -----------------------------
if __name__ == "__main__":

    MAX_ITERATIONS = 100

    print("Running NAIVE multimodal system...\n")

    try:
        for i in range(MAX_ITERATIONS):

            print(f"\n--- Iteration {i+1}/100 ---")

            # -----------------------------
            # READ TELEMETRY
            # -----------------------------
            raw = read_tegrastats()
            state = parse_tegrastats(raw)

            if not state:
                continue

            print(f"State: {state}")
            print("Decision: NAIVE (RUN BOTH)")

            # -----------------------------
            # RUN BOTH ALWAYS
            # -----------------------------
            yolo_thread = threading.Thread(target=run_yolo)
            bert_thread = threading.Thread(target=run_bert)

            yolo_thread.start()
            bert_thread.start()

            yolo_thread.join()
            bert_thread.join()

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
                    "NAIVE",
                    1,
                    1
                ])

            time.sleep(1)

        print("\nCompleted 100 iterations.")

    except KeyboardInterrupt:
        print("\nStopped early.")
