import time
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -----------------------------
# SAMPLE INPUT
# -----------------------------
text = "Jetson Orin NX enables efficient edge AI inference."

# -----------------------------
# RUN INFERENCE & MEASURE LATENCY
# -----------------------------
with torch.no_grad():
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
    outputs = model(**inputs)
    end = time.time()

latency_ms = (end - start) * 1000

print(f"DistilBERT inference latency: {latency_ms:.2f} ms")
