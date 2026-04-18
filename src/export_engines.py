"""
export_engines.py — Export YOLOv8 and DistilBERT to TensorRT .engine files
                    for Jetson Orin NX 8 GB

Precision strategy:
    Naive scheduler   → FP32 PyTorch directly (NO engine needed)
    Rule/RL scheduler → FP16 and INT8 TensorRT engines

Engines produced in engines/:
    yolov8_fp16.engine   — YOLO TensorRT FP16
    yolov8_int8.engine   — YOLO TensorRT INT8
    bert_fp16.engine     — BERT TensorRT FP16
    bert_int8.engine     — BERT TensorRT INT8

NOTE: FP32 models (yolov8m.pt, distilbert-base-uncased) are loaded
      directly by naive_scheduler.py — no export needed for them.

Requirements:
    pip install ultralytics transformers torch onnx
    TensorRT + trtexec installed via JetPack SDK
    export PATH=$PATH:/usr/src/tensorrt/bin

Expected time: ~10–20 minutes (INT8 calibration is the slow step).
"""

import os
import sys
import numpy as np
import torch

ENGINE_DIR  = "engines"
os.makedirs(ENGINE_DIR, exist_ok=True)

MAX_SEQ_LEN = 128   # fixed sequence length for BERT TRT engines

BERT_ONNX  = os.path.join(ENGINE_DIR, "distilbert.onnx")
BERT_FP16  = os.path.join(ENGINE_DIR, "bert_fp16.engine")
BERT_INT8  = os.path.join(ENGINE_DIR, "bert_int8.engine")


# ══════════════════════════════════════════════
# PART 1 — YOLOv8m → TensorRT FP16 and INT8
# ══════════════════════════════════════════════
def export_yolo():
    from ultralytics import YOLO

    # ── FP16 ──────────────────────────────────
    print("\n" + "="*55)
    print("  Exporting YOLOv8m → TensorRT FP16")
    print("="*55)

    model_fp16 = YOLO("yolov8m.pt")
    model_fp16.export(
        format="engine",
        half=True,
        int8=False,
        imgsz=640,
        device=0,
        workspace=4,
    )
    _move_engine("yolov8m.engine", "yolov8_fp16.engine")

    # ── INT8 ──────────────────────────────────
    print("\n" + "="*55)
    print("  Exporting YOLOv8m → TensorRT INT8")
    print("="*55)

    model_int8 = YOLO("yolov8m.pt")
    model_int8.export(
        format="engine",
        int8=True,
        half=False,
        imgsz=640,
        device=0,
        workspace=4,
        data="coco128.yaml",   # calibration data — downloads automatically
    )
    _move_engine("yolov8m.engine", "yolov8_int8.engine")


def _move_engine(src: str, dst_name: str):
    """Move Ultralytics-generated engine to engines/ with a clean name."""
    dst = os.path.join(ENGINE_DIR, dst_name)
    if os.path.exists(src):
        os.rename(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"Saved: {dst}  ({size_mb:.1f} MB)")
    else:
        # Ultralytics sometimes saves next to the .pt file
        alt = os.path.join(os.path.dirname("yolov8m.pt"), "yolov8m.engine")
        if os.path.exists(alt):
            os.rename(alt, dst)
            print(f"Saved: {dst}")
        else:
            print(f"[WARN] Engine not found at {src} — check Ultralytics output path")


# ══════════════════════════════════════════════
# PART 2 — DistilBERT → ONNX (shared base)
# ══════════════════════════════════════════════
def export_bert_to_onnx():
    """
    Export DistilBERT from PyTorch to ONNX with fixed input shape.
    This ONNX file is the input for both FP16 and INT8 trtexec builds.
    The model is exported in FP32 — trtexec handles precision conversion.
    """
    from transformers import DistilBertTokenizer, DistilBertModel

    print("\n" + "="*55)
    print("  Exporting DistilBERT → ONNX (FP32 base)")
    print("="*55)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model     = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    dummy_ids  = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        BERT_ONNX,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes=None,        # fixed shape — required for TensorRT
        opset_version=13,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(BERT_ONNX) / (1024 * 1024)
    print(f"Saved ONNX: {BERT_ONNX}  ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════
# PART 3 — ONNX → TensorRT FP16 / INT8
# ══════════════════════════════════════════════
def export_bert_trt(precision: str):
    """
    Convert distilbert.onnx → TensorRT engine using trtexec.

    precision: "fp16" or "int8"

    trtexec is shipped with JetPack. If not in PATH, run:
        export PATH=$PATH:/usr/src/tensorrt/bin
    """
    import subprocess

    assert precision in ("fp16", "int8")
    out_engine = BERT_FP16 if precision == "fp16" else BERT_INT8

    print("\n" + "="*55)
    print(f"  ONNX → TensorRT {precision.upper()}")
    print("="*55)

    cmd = [
        "trtexec",
        f"--onnx={BERT_ONNX}",
        f"--saveEngine={out_engine}",
        "--explicitBatch",
        "--workspace=2048",   # MB
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    else:
        cmd += ["--int8", "--fp16"]   # FP16 fallback for unsupported INT8 layers

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, text=True)

    if result.returncode == 0:
        size_mb = os.path.getsize(out_engine) / (1024 * 1024)
        print(f"Saved: {out_engine}  ({size_mb:.1f} MB)")
    else:
        print(f"[ERROR] trtexec failed for {precision.upper()}")
        print("Ensure trtexec is in PATH:")
        print("  export PATH=$PATH:/usr/src/tensorrt/bin")
        sys.exit(1)


# ══════════════════════════════════════════════
# PART 4 — Verify all engines + FP32 PyTorch
# ══════════════════════════════════════════════
def verify_engines():
    print("\n" + "="*55)
    print("  Verification")
    print("="*55)

    from ultralytics import YOLO
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    # FP32 PyTorch YOLO (used by naive scheduler)
    try:
        m = YOLO("yolov8m.pt")
        m.predict(source=dummy, device=0, verbose=False)
        print("[OK] yolov8m.pt  (FP32 PyTorch — naive baseline)")
    except Exception as e:
        print(f"[FAIL] yolov8m.pt: {e}")

    # TRT engines for rule/RL schedulers
    for name in ["yolov8_fp16.engine", "yolov8_int8.engine"]:
        path = os.path.join(ENGINE_DIR, name)
        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            continue
        try:
            m = YOLO(path)
            m.predict(source=dummy, device=0, verbose=False)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[OK] {name}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    # FP32 PyTorch BERT (used by naive scheduler)
    try:
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch
        tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        mdl = DistilBertModel.from_pretrained("distilbert-base-uncased")
        mdl.eval()
        inp = tok("test", return_tensors="pt", max_length=128,
                  truncation=True, padding="max_length")
        with torch.no_grad():
            mdl(**inp)
        print("[OK] distilbert-base-uncased  (FP32 PyTorch — naive baseline)")
    except Exception as e:
        print(f"[FAIL] DistilBERT PyTorch FP32: {e}")

    # TRT BERT engines
    for name in ["bert_fp16.engine", "bert_int8.engine"]:
        path = os.path.join(ENGINE_DIR, name)
        if not os.path.exists(path):
            print(f"[MISSING] {path}")
        elif os.path.getsize(path) < 1024:
            print(f"[FAIL] {name} — file too small, export likely failed")
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[OK] {name}  ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("\nJetson Orin NX — TensorRT Engine Export")
    print(f"Output directory: {os.path.abspath(ENGINE_DIR)}")
    print("\nPrecision plan:")
    print("  Naive     → FP32 PyTorch  (no export needed)")
    print("  Rule / RL → FP16 + INT8 TensorRT engines\n")

    export_yolo()

    export_bert_to_onnx()
    export_bert_trt("fp16")
    export_bert_trt("int8")

    verify_engines()

    print("\nExport complete!")
    print(f"Engines saved to: {os.path.abspath(ENGINE_DIR)}/")
    print("\nRun order:")
    print("  1. python3 train_rl.py")
    print("  2. python3 naive_scheduler.py")
    print("  3. python3 rule_based_scheduler.py")
    print("  4. python3 rl_scheduler.py")
    print("  5. python3 final_results.py")
