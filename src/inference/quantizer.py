import os
import time
import torch
import numpy as np
from transformers import BertTokenizerFast
from configs.config import config


def quantize_onnx_model(
    onnx_path: str = None,
    output_path: str = None,
):
    """
    Apply INT8 dynamic quantization to the ONNX model.
    Reduces model size by ~4x and speeds up CPU inference.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_path = onnx_path or os.path.join(config.BASE_DIR, "models", "onnx", "model.onnx")
    output_path = output_path or os.path.join(config.BASE_DIR, "models", "onnx", "model_quantized.onnx")

    print(f"Quantizing model: {onnx_path}")

    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"\nQuantization successful!")
    print(f"Original ONNX size:   {original_size:.1f} MB")
    print(f"Quantized ONNX size:  {quantized_size:.1f} MB")
    print(f"Size reduction:       {reduction:.1f}%")

    return output_path


def benchmark_quantized(quantized_path: str = None):
    """
    Benchmark quantized ONNX vs original ONNX inference.
    """
    import onnxruntime as ort

    onnx_path = os.path.join(config.BASE_DIR, "models", "onnx", "model.onnx")
    quantized_path = quantized_path or os.path.join(config.BASE_DIR, "models", "onnx", "model_quantized.onnx")

    tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)
    encoding = tokenizer(
        ["customer id"],
        is_split_into_words=True,
        max_length=config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    ort_inputs = {
        "input_ids": encoding["input_ids"].numpy(),
        "attention_mask": encoding["attention_mask"].numpy(),
        "token_type_ids": encoding["token_type_ids"].numpy(),
    }

    num_runs = 20
    print(f"\nBenchmarking quantized model ({num_runs} runs each)...")

    # Original ONNX
    ort_session = ort.InferenceSession(onnx_path)
    start = time.time()
    for _ in range(num_runs):
        ort_session.run(None, ort_inputs)
    onnx_time = (time.time() - start) / num_runs * 1000

    # Quantized ONNX
    ort_session_q = ort.InferenceSession(quantized_path)
    start = time.time()
    for _ in range(num_runs):
        ort_session_q.run(None, ort_inputs)
    quantized_time = (time.time() - start) / num_runs * 1000

    speedup = onnx_time / quantized_time

    print(f"\n  Original ONNX:  {onnx_time:.2f} ms")
    print(f"  Quantized ONNX: {quantized_time:.2f} ms")
    print(f"  Speedup:        {speedup:.2f}x")

    return {
        "onnx_ms": onnx_time,
        "quantized_ms": quantized_time,
        "speedup": speedup,
    }


if __name__ == "__main__":
    quantized_path = quantize_onnx_model()
    benchmark_quantized(quantized_path)