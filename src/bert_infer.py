"""
bert_infer.py — TensorRT 10 inference wrapper for DistilBERT
               Uses PyTorch tensors directly — no pycuda required.

TensorRT 10 API changes from TRT 8/9:
  - get_binding_shape()  → get_tensor_shape()
  - execute_async_v2()   → execute_async_v3()
  - Bindings list        → set_tensor_address() per tensor name
"""

import numpy as np
import torch
import tensorrt as trt
from transformers import DistilBertTokenizer

MAX_SEQ_LEN = 128
TRT_LOGGER  = trt.Logger(trt.Logger.WARNING)


class BertTRTInference:
    """
    Loads a TensorRT 10 .engine file for DistilBERT and runs inference
    using PyTorch CUDA tensors as input/output buffers.
    No pycuda dependency.
    """

    def __init__(self, engine_path: str):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        # ── Load engine ───────────────────────────────────────────────
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine  = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # ── Get tensor names ──────────────────────────────────────────
        self.input_names  = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # ── Pre-allocate PyTorch CUDA tensors ─────────────────────────
        self.input_tensors  = {}
        self.output_tensors = {}

        for name in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            # Replace -1 batch dim with 1
            shape = tuple(1 if s == -1 else s for s in shape)
            # BERT inputs are int32
            self.input_tensors[name] = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )

        for name in self.output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            shape = tuple(1 if s == -1 else s for s in shape)
            self.output_tensors[name] = torch.zeros(
                shape, dtype=torch.float32, device="cuda"
            )

        # ── Bind tensor addresses to context ──────────────────────────
        for name, tensor in {**self.input_tensors,
                             **self.output_tensors}.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

    # ─────────────────────────────────────────
    def infer(self, text: str) -> np.ndarray:
        """Tokenize text and run one TensorRT forward pass."""

        encoded = self.tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Copy tokenized inputs into pre-allocated CUDA tensors
        self.input_tensors["input_ids"].copy_(
            encoded["input_ids"].to(torch.int32)
        )
        self.input_tensors["attention_mask"].copy_(
            encoded["attention_mask"].to(torch.int32)
        )

        # Run inference (TRT 10 API)
        self.context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream
        )

        # Sync and return output as numpy
        torch.cuda.synchronize()
        return self.output_tensors[self.output_names[0]].cpu().numpy()
