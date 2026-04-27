"""
uv run main.py --model gemma --quantize 4bit --device cpu
uv run main.py --model vit --device gpu
uv run main.py --model both --quantize 4bit --device cpu
"""

import argparse
import gc
import io
import time

import numpy as np
import psutil
import requests
import torch
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _mem_snapshot(use_gpu):
    stats = {"cpu_ram_mb": psutil.Process().memory_info().rss / 1024**2}
    if use_gpu:
        stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
    return stats


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _build_quant_config(quantize, use_gpu):
    if quantize != "4bit":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 if use_gpu else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=not use_gpu,
    )


def _print_mem_load(mem_pre, mem_post, use_gpu):
    print(f"  cpu_ram_mb:      {round(mem_post['cpu_ram_mb'] - mem_pre['cpu_ram_mb'], 2)}")
    if use_gpu:
        print(f"  gpu_allocated_mb:{round(mem_post['gpu_allocated_mb'] - mem_pre['gpu_allocated_mb'], 2)}")


def _print_mem_inference(mem_before, mem_after, use_gpu):
    print(f"  cpu_ram_delta_mb:    {round(mem_after['cpu_ram_mb'] - mem_before['cpu_ram_mb'], 2)}")
    if use_gpu:
        print(f"  gpu_peak_mb:         {round(torch.cuda.max_memory_allocated() / 1024**2, 2)}")
        print(f"  gpu_allocated_delta: {round(mem_after['gpu_allocated_mb'] - mem_before['gpu_allocated_mb'], 2)}")


# ---------------------------------------------------------------------------
# Gemma benchmark
# ---------------------------------------------------------------------------

def benchmark_gemma(device, quantize, prompt, max_new_tokens=200):
    use_gpu = device == "gpu"
    model_id = "google/gemma-4-E2B-it"

    quant_config = _build_quant_config(quantize, use_gpu)
    dtype = None if quant_config else (torch.float16 if use_gpu else torch.bfloat16)
    label = "int4 (nf4, double quant)" if quantize == "4bit" else str(dtype)

    print(f"\n{'='*60}")
    print(f"  Gemma benchmark | device={device.upper()} | quant={label}")
    print(f"{'='*60}")
    _clear_memory()

    mem_pre_load = _mem_snapshot(use_gpu)
    t_load = time.perf_counter()

    pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="cuda" if use_gpu else "cpu",
        dtype=dtype,
        model_kwargs={"quantization_config": quant_config} if quant_config else {},
    )

    load_time_s = time.perf_counter() - t_load
    mem_post_load = _mem_snapshot(use_gpu)

    print(f"\n--- Model load ---")
    print(f"  load_time_s: {round(load_time_s, 2)}")
    _print_mem_load(mem_pre_load, mem_post_load, use_gpu)

    tokenizer = pipe.tokenizer
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(tokenizer.encode(prompt_text))

    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    mem_before = _mem_snapshot(use_gpu)
    t0 = time.perf_counter()

    result = pipe(messages, max_new_tokens=max_new_tokens)

    latency_s = time.perf_counter() - t0
    mem_after = _mem_snapshot(use_gpu)

    output_text = result[0]["generated_text"][-1]["content"]
    output_tokens = len(tokenizer.encode(output_text))

    print(f"\n--- Output ---")
    print(output_text)

    print(f"\n--- Inference metrics ---")
    print(f"  latency_s:           {round(latency_s, 3)}")
    print(f"  prompt_tokens:       {prompt_tokens}")
    print(f"  output_tokens:       {output_tokens}")
    print(f"  throughput_tok_per_s:{round(output_tokens / latency_s, 2)}")
    _print_mem_inference(mem_before, mem_after, use_gpu)

    del pipe
    _clear_memory()


# ---------------------------------------------------------------------------
# ViT benchmark
# ---------------------------------------------------------------------------

def benchmark_vit(device, quantize, num_iterations=50):
    use_gpu = device == "gpu"
    model_id = "google/vit-base-patch16-224"
    label = "int4 (nf4)" if quantize == "4bit" else ("float16" if use_gpu else "bfloat16")

    print(f"\n{'='*60}")
    print(f"  ViT benchmark | device={device.upper()} | quant={label}")
    print(f"{'='*60}")
    _clear_memory()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(io.BytesIO(requests.get(url).content))

    quant_config = _build_quant_config(quantize, use_gpu)
    torch_device = "cuda:0" if use_gpu else "cpu"

    mem_pre_load = _mem_snapshot(use_gpu)
    t_load = time.perf_counter()

    try:
        pipe = pipeline(
            "image-classification",
            model=model_id,
            device=torch_device,
            model_kwargs={"quantization_config": quant_config} if quant_config else {},
        )
    except ValueError:
        pipe = pipeline(
            "image-classification",
            model=model_id,
            device_map={"": torch_device},
            model_kwargs={"quantization_config": quant_config} if quant_config else {},
        )

    load_time_s = time.perf_counter() - t_load
    mem_post_load = _mem_snapshot(use_gpu)

    print(f"\n--- Model load ---")
    print(f"  load_time_s: {round(load_time_s, 2)}")
    _print_mem_load(mem_pre_load, mem_post_load, use_gpu)

    print("Warming up (3 iterations)...")
    for i in range(3):
        output = pipe(image)
        if i == 0:
            print(f"  Sample output: {output[0]}")

    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    mem_before = _mem_snapshot(use_gpu)

    print(f"Running {num_iterations} iterations...")
    latencies = []
    t_total = time.perf_counter()
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        pipe(image)
        latencies.append(time.perf_counter() - t0)
    total_time = time.perf_counter() - t_total

    mem_after = _mem_snapshot(use_gpu)

    print(f"\n--- Inference metrics ---")
    print(f"  throughput_img_per_s:{round(num_iterations / total_time, 2)}")
    print(f"  avg_latency_ms:      {round(np.mean(latencies) * 1000, 2)}")
    print(f"  p95_latency_ms:      {round(np.percentile(latencies, 95) * 1000, 2)}")
    _print_mem_inference(mem_before, mem_after, use_gpu)

    del pipe
    _clear_memory()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gemma", "vit", "both"],
        default="gemma",
        help="Which model to benchmark (default: gemma)",
    )
    parser.add_argument(
        "--quantize",
        choices=["4bit", "none"],
        default="none",
        help="Quantization: 4bit (nf4) or none (bf16/fp16)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Prompt for the Gemma model",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of inference iterations for ViT benchmark (default: 50)",
    )
    args = parser.parse_args()

    if args.device == "gpu" and not torch.cuda.is_available():
        raise SystemExit("--device gpu requested but no CUDA GPU is available.")

    if args.model in ("gemma", "both"):
        benchmark_gemma(args.device, args.quantize, args.prompt)

    if args.model in ("vit", "both"):
        benchmark_vit(args.device, args.quantize, args.iterations)


if __name__ == "__main__":
    main()
