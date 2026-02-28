import os
import torch
import numpy as np
from transformers import BertTokenizerFast
from src.models.ner_model import BertNERModel
from configs.config import config


def export_to_onnx(
    model_path: str = None,
    output_path: str = None,
):
    """
    Export the trained BERT NER model to ONNX format
    for optimized inference.
    """
    model_path = model_path or os.path.join(config.BASE_DIR, "models", "best_model")
    output_path = output_path or os.path.join(config.BASE_DIR, "models", "onnx", "model.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading model from: {model_path}")
    tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)
    model = BertNERModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
    model.eval()

    # Create dummy input for tracing
    dummy_input = tokenizer(
        ["customer id"],
        is_split_into_words=True,
        max_length=config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    token_type_ids = dummy_input["token_type_ids"]

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        output_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
        do_constant_folding=True,
    )

    # Verify the export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    original_size = os.path.getsize(os.path.join(model_path, "model.safetensors")) / (1024 * 1024)
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nONNX Export successful!")
    print(f"Output path:      {output_path}")
    print(f"Original size:    {original_size:.1f} MB")
    print(f"ONNX size:        {onnx_size:.1f} MB")

    return output_path


def benchmark_onnx(onnx_path: str = None):
    """
    Benchmark ONNX inference vs PyTorch inference.
    """
    import onnxruntime as ort
    import time

    onnx_path = onnx_path or os.path.join(config.BASE_DIR, "models", "onnx", "model.onnx")
    model_path = os.path.join(config.BASE_DIR, "models", "best_model")

    tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)

    # Prepare input
    encoding = tokenizer(
        ["customer id"],
        is_split_into_words=True,
        max_length=config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]

    # PyTorch benchmark
    model = BertNERModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
    model.eval()

    num_runs = 20
    print(f"\nBenchmarking ({num_runs} runs each)...")

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    pytorch_time = (time.time() - start) / num_runs * 1000

    # ONNX benchmark
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy(),
    }

    start = time.time()
    for _ in range(num_runs):
        ort_session.run(None, ort_inputs)
    onnx_time = (time.time() - start) / num_runs * 1000

    speedup = pytorch_time / onnx_time

    print(f"\n  PyTorch inference: {pytorch_time:.2f} ms")
    print(f"  ONNX inference:    {onnx_time:.2f} ms")
    print(f"  Speedup:           {speedup:.2f}x")

    return {"pytorch_ms": pytorch_time, "onnx_ms": onnx_time, "speedup": speedup}


if __name__ == "__main__":
    onnx_path = export_to_onnx()
    benchmark_onnx(onnx_path)