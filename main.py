import argparse
import time

import psutil
import torch
from transformers import BitsAndBytesConfig, pipeline


def _mem_snapshot():
    stats = {"cpu_ram_mb": psutil.Process().memory_info().rss / 1024**2}
    if torch.cuda.is_available():
        stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
    return stats


def run_inference(pipe, messages, max_new_tokens=200):
    tokenizer = pipe.tokenizer

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(tokenizer.encode(prompt_text))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mem_before = _mem_snapshot()
    t0 = time.perf_counter()

    result = pipe(messages, max_new_tokens=max_new_tokens)

    latency_s = time.perf_counter() - t0
    mem_after = _mem_snapshot()

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

    if torch.cuda.is_available():
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
        choices=["4bit", "none"],
        default="none",
        help="Load model in 4-bit quantization (requires CUDA)",
    )
    parser.add_argument(
        "--prompt",
        default="none",
        help="Load model in 4-bit quantization (requires CUDA)",
    )
    args = parser.parse_args()

    model_id = "google/gemma-4-E2B-it"
    # model_id = "google/gemma-3-270m-it"

    if args.quantize == "4bit":
        on_gpu = torch.cuda.is_available()
        compute_dtype = torch.float16 if on_gpu else torch.bfloat16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 bits/param
            bnb_4bit_quant_type="nf4",        # NormalFloat4 is optimal for LLM weights
            llm_int8_enable_fp32_cpu_offload=not on_gpu,  # required for CPU-only inference
        )
        dtype = None  # bitsandbytes controls the dtype
        label = "int4 (nf4, double quant)"
    else:
        quant_config = None
        # fp16 is only numerically stable on CUDA; CPU requires bfloat16
        dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        label = str(dtype)

    print(f"Loading {model_id} [{label}] ...")
    mem_pre_load = _mem_snapshot()
    t_load = time.perf_counter()

    pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        dtype=dtype,
        model_kwargs={"quantization_config": quant_config} if quant_config else {},
    )

    load_time_s = time.perf_counter() - t_load
    mem_post_load = _mem_snapshot()

    model_memory = {
        "quantization": label,
        "load_time_s": round(load_time_s, 2),
        "cpu_ram_mb": round(mem_post_load["cpu_ram_mb"] - mem_pre_load["cpu_ram_mb"], 2),
    }
    if torch.cuda.is_available():
        model_memory["gpu_allocated_mb"] = round(
            mem_post_load["gpu_allocated_mb"] - mem_pre_load["gpu_allocated_mb"], 2
        )

    print("\n--- Model load ---")
    for k, v in model_memory.items():
        print(f"  {k}: {v}")

    messages = [{"role": "user", "content": "What is the capital of France?"}]

    output, metrics = run_inference(pipe, messages)

    print("\n--- Output ---")
    print(output)

    print("\n--- Inference metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
