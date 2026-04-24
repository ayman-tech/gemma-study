"""
uv run main.py --quantize 4bit --device cpu
!uv run main.py --quantize 8bit --device gpu
"""

import argparse
import time

import psutil
import torch
from transformers import BitsAndBytesConfig, pipeline


def _mem_snapshot(use_gpu):
    stats = {"cpu_ram_mb": psutil.Process().memory_info().rss / 1024**2}
    if use_gpu:
        stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
    return stats


def run_inference(pipe, messages, use_gpu, max_new_tokens=200):
    tokenizer = pipe.tokenizer

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

    metrics = {
        "latency_s": round(latency_s, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "throughput_tok_per_s": round(output_tokens / latency_s, 2),
        "cpu_ram_delta_mb": round(
            mem_after["cpu_ram_mb"] - mem_before["cpu_ram_mb"], 2
        ),
    }

    if use_gpu:
        metrics["gpu_peak_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024**2, 2
        )
        metrics["gpu_allocated_delta_mb"] = round(
            mem_after["gpu_allocated_mb"] - mem_before["gpu_allocated_mb"], 2
        )

    return output_text, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize",
        choices=["4bit", "8bit", "none"],
        default="none",
        help="Quantization: 4bit (nf4), 8bit (int8), or none",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Prompt to send to the model",
    )
    args = parser.parse_args()

    use_gpu = args.device == "gpu"
    if use_gpu and not torch.cuda.is_available():
        raise SystemExit("--device gpu requested but no CUDA GPU is available.")

    model_id = "google/gemma-4-E2B-it"
    # model_id = "google/gemma-3-270m-it"

    if args.quantize == "4bit":
        compute_dtype = torch.float16 if use_gpu else torch.bfloat16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,                # nested quantization saves ~0.4 bits/param
            bnb_4bit_quant_type="nf4",                     # NormalFloat4 is optimal for LLM weights
            llm_int8_enable_fp32_cpu_offload=not use_gpu,  # required for CPU-only inference
        )
        dtype = None
        label = "int4 (nf4, double quant)"
    elif args.quantize == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=not use_gpu,  # required for CPU-only inference
        )
        dtype = None
        label = "int8"
    else:
        quant_config = None
        # fp16 is only numerically stable on CUDA; CPU requires bfloat16
        dtype = torch.float16 if use_gpu else torch.bfloat16
        label = str(dtype)

    print(f"Loading {model_id} [{label}] on {args.device.upper()} ...")
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

    model_memory = {
        "device": args.device,
        "quantization": label,
        "load_time_s": round(load_time_s, 2),
        "cpu_ram_mb": round(mem_post_load["cpu_ram_mb"] - mem_pre_load["cpu_ram_mb"], 2),
    }
    if use_gpu:
        model_memory["gpu_allocated_mb"] = round(
            mem_post_load["gpu_allocated_mb"] - mem_pre_load["gpu_allocated_mb"], 2
        )

    print("\n--- Model load ---")
    for k, v in model_memory.items():
        print(f"  {k}: {v}")

    messages = [{"role": "user", "content": args.prompt}]

    output, metrics = run_inference(pipe, messages, use_gpu)

    print("\n--- Output ---")
    print(output)

    print("\n--- Inference metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
