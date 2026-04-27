# Gemma Inference & Benchmarking Suite

A complete CLI-based inference and benchmarking suite for Gemma text generation and ViT image classification across CPU and GPU.

---

## Repository

```bash
git clone https://github.com/ayman-tech/gemma-study.git
```

---

## Project Structure

```
gemma-study/
├── main.py               # Gemma LLM inference + benchmarking script
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependency versions
└── output.txt            # Sample output from a previous run

```

---

## Features

- Run text generation with `google/gemma-4-E2B-it`
- Two quantization modes: **4-bit (NF4)** or **full precision**
- CPU and GPU (CUDA) support for both LLM and vision models
- ViT image classification benchmark (`google/vit-base-patch16-224`)
- Detailed metrics: latency, throughput, token counts, CPU/GPU memory usage
- Reproducible environment via [`uv`](https://github.com/astral-sh/uv)

---

## Requirements

- Python >= 3.12
- CUDA-capable GPU (optional, for GPU inference)
- [`uv`](https://github.com/astral-sh/uv) package manager

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | >= 2.11.0 | Deep learning backend |
| `transformers` | >= 5.6.2 | HuggingFace model pipelines |
| `accelerate` | >= 1.13.0 | Device dispatch |
| `bitsandbytes` | >= 0.49.2 | Quantization support |
| `psutil` | >= 7.2.2 | Memory tracking |
| `Pillow` | latest | Image loading for ViT benchmark |
| `numpy` | latest | Latency statistics |
| `requests` | latest | Sample image download |

---

## Usage (`main.py`)

### Installation

```bash
git clone https://github.com/ayman-tech/gemma-study.git
cd gemma-study
uv sync
```

### Running Inference

```bash
# Gemma only — CPU, no quantization (default)
uv run main.py

# Gemma — CPU with 4-bit
uv run main.py --model gemma --device cpu --quantize 4bit

# Gemma — GPU with 4-bit
uv run main.py --model gemma --device gpu --quantize 4bit

# ViT only — CPU
uv run main.py --model vit --device cpu

# ViT only — GPU
uv run main.py --model vit --device gpu

# Both models — GPU, 4-bit
uv run main.py --model both --device gpu --quantize 4bit

# Custom prompt
uv run main.py --model gemma --prompt "Explain attention in one sentence"

# ViT with more iterations
uv run main.py --model vit --device gpu --iterations 100

```

### CLI Arguments

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--quantize` | `4bit`, `none` | `none` | Quantization mode |
| `--device` | `cpu`, `gpu` | `cpu` | Compute device |
| `--model` | `gemma`, `vit`, `both` | `gemma` | Which benchmark(s) to run |
| `--iterations` | any integer | `50` | ViT benchmark iterations |
| `--prompt` | any string | `"What is the capital of France?"` | Input prompt |

### Quantization Modes

| Mode | Format | Best For |
|---|---|---|
| `none` | `bfloat16` (CPU) / `float16` (GPU) | Highest accuracy |
| `4bit` | NF4 + double quantization | Lowest memory, good quality |
> **Note:** In CPU mode (`--device cpu`), 4-bit loading sets `llm_int8_enable_fp32_cpu_offload=True` in the bitsandbytes config.

### LLM Output

```
--- Model load ---
  device: cpu
  quantization: int4 (nf4, double quant)
  load_time_s: 12.34
  cpu_ram_mb: 2048.5

--- Output ---
<model response>

--- Inference metrics ---
  latency_s: 3.141
  prompt_tokens: 12
  output_tokens: 87
  throughput_tok_per_s: 27.7
  cpu_ram_delta_mb: 15.2
  gpu_peak_mb: 3200.0           # GPU only
  gpu_allocated_delta_mb: 50.1  # GPU only
```

---

## ViT Image Classification Benchmark

`main.py` also benchmarks `google/vit-base-patch16-224` on a COCO sample image.

### Model & Settings

| Setting | Value |
|---|---|
| Model | `google/vit-base-patch16-224` |
| Task | `image-classification` |
| Warmup iterations | 3 |
| Benchmark iterations | 50 |
| Sample image | COCO val2017 `000000039769.jpg` |

### Benchmark Flow

```python
from transformers import pipeline

pipe = pipeline("image-classification", model="google/vit-base-patch16-224", device=device)

# Warmup
for i in range(3):
    output = pipe(image)

# Benchmark
latencies = []
for _ in range(50):
    start = time.time()
    pipe(image)
    latencies.append(time.time() - start)

# Metrics
throughput      = 50 / total_time
avg_latency_ms  = np.mean(latencies) * 1000
p95_latency_ms  = np.percentile(latencies, 95) * 1000
```

### Metrics Collected

| Metric | Description |
|---|---|
| **Throughput** | Images processed per second |
| **Average Latency** | Mean inference time in ms |
| **P95 Latency** | 95th-percentile inference time in ms |
| **Delta CPU Memory** | RAM consumed during inference (MB) |
| **Delta GPU Memory** | VRAM consumed during inference (MB) — GPU only |

### CPU vs GPU

Use `--device cpu` or `--device gpu`. If `--device gpu` is requested without CUDA availability, the script exits with:

```text
--device gpu requested but no CUDA GPU is available.
```

---

## Models Used

| Model | Task | Source |
|---|---|---|
| `google/gemma-4-E2B-it` | Text generation | [HuggingFace](https://huggingface.co/google/gemma-4-E2B-it) |
| `google/vit-base-patch16-224` | Image classification | [HuggingFace](https://huggingface.co/google/vit-base-patch16-224) |

To swap the Gemma model, edit `main.py`:
```python
model_id = "google/gemma-3-270m-it"  # lighter alternative
```

---

## License

Not specified. Contact the repository author for usage terms.
