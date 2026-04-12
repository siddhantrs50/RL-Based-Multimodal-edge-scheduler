import threading
import time
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import subprocess

# -----------------------------
# YOLO THREAD
# -----------------------------
def run_yolo():
    print("Starting YOLO inference...")
    subprocess.run(
        [
            "yolo",
            "predict",
            "model=yolov8n.pt",
            "source=https://ultralytics.com/images/bus.jpg",
            "device=0",
            "show=False",
            "save=False"
        ]
    )

# -----------------------------
# BERT THREAD
# -----------------------------
def run_bert():
    print("Starting DistilBERT inference...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text = "Running vision and language models together stresses edge devices."

    for _ in range(5):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            start = time.time()
            _ = model(**inputs)
            end = time.time()
            print(f"BERT latency: {(end - start) * 1000:.2f} ms")
        time.sleep(1)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    yolo_thread = threading.Thread(target=run_yolo)
    bert_thread = threading.Thread(target=run_bert)

    yolo_thread.start()
    bert_thread.start()

    yolo_thread.join()
    bert_thread.join()

    print("Naive multimodal inference completed.")
