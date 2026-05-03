"""
ViT fine-tuning (LoRA / QLoRA) on keremberke/chest-xray-classification.

Usage:
    uv run vit-ft.py --device gpu --quantize none --epochs 50   # LoRA bf16
    uv run vit-ft.py --device gpu --quantize 4bit --epochs 50   # QLoRA 4-bit
    uv run vit-ft.py --device cpu --quantize none --epochs 1    # smoke test

Adapter weights saved to:
    ./output/vit-lora/   (LoRA)
    ./output/vit-qlora/  (QLoRA)

To reload:
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "./output/vit-lora/")
"""

import argparse
import gc
import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    BitsAndBytesConfig,
    ViTForImageClassification,
)


# ---------------------------------------------------------------------------
# Shared utilities (duplicated from main.py — main.py is not a module)
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


def _build_quant_config(quantize, use_gpu, skip_modules=None):
    if quantize != "4bit":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 if use_gpu else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=not use_gpu,
        llm_int8_skip_modules=skip_modules,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ram_poller(stop_evt, holder):
    """Background thread: records the peak RSS seen during an epoch."""
    proc = psutil.Process()
    peak = proc.memory_info().rss
    while not stop_evt.is_set():
        peak = max(peak, proc.memory_info().rss)
        time.sleep(0.1)
    holder["peak_mb"] = peak / 1024**2


def _make_transform(processor, label_col):
    def transform(batch):
        inputs = processor(
            images=[img.convert("RGB") for img in batch["image"]],
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor(batch[label_col], dtype=torch.long)
        return inputs
    return transform


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def _plot_metrics(output_dir, method, train_times, throughputs, cpu_rams, gpu_rams, losses, val_accs):
    epochs = list(range(1, len(losses) + 1))
    metrics = [
        ("Train Loss",          losses,      "Cross-Entropy Loss", "tab:blue"),
        ("Validation Accuracy", val_accs,    "Accuracy",           "tab:green"),
        ("Train Time / Epoch",  train_times, "Seconds",            "tab:orange"),
        ("Throughput",          throughputs, "Samples / s",        "tab:purple"),
        ("Peak CPU RAM",        cpu_rams,    "MB",                 "tab:red"),
    ]
    if gpu_rams:
        metrics.append(("Peak GPU RAM", gpu_rams, "MB", "tab:brown"))

    ncols = 2
    nrows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = axes.flatten()

    for i, (title, data, ylabel, color) in enumerate(metrics):
        ax = axes[i]
        ax.plot(epochs, data, marker="o", linewidth=1.5, markersize=4, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"ViT Fine-tuning ({method})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {path}")


# ---------------------------------------------------------------------------
# Fine-tuning (LoRA and QLoRA share everything except model loading)
# ---------------------------------------------------------------------------

def finetune_vit(device, quantize, epochs):
    use_gpu = device == "gpu"
    is_qlora = quantize == "4bit"
    method = "QLoRA" if is_qlora else "LoRA"
    output_dir = f"./output/{device}_{method.lower()}"
    torch_device = "cuda:0" if use_gpu else "cpu"

    print(f"\n{'='*60}")
    print(f"  ViT fine-tuning | method={method} | device={device.upper()} | epochs={epochs}")
    print(f"{'='*60}")
    _clear_memory()

    # --- Dataset ---
    print("\nLoading dataset...")
    raw = load_dataset("keremberke/chest-xray-classification", name="full", trust_remote_code=True)
    val_split = "test" if "test" in raw else list(raw.keys())[-1]

    label_col = "labels" if "labels" in raw["train"].features else "label"
    label_feat = raw["train"].features[label_col]
    num_labels = label_feat.num_classes
    id2label = {i: label_feat.int2str(i) for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}
    print(f"  Classes ({num_labels}): {list(id2label.values())}")

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    transform = _make_transform(processor, label_col)
    raw["train"].set_transform(transform)
    raw[val_split].set_transform(transform)

    # num_workers=0: Windows spawn-based multiprocessing can't pickle dataset closures;
    # on Linux HPC this can be increased safely
    num_workers = 0
    train_loader = DataLoader(
        raw["train"], batch_size=16, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=use_gpu,
    )
    val_loader = DataLoader(
        raw[val_split], batch_size=16, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=use_gpu,
    )

    # --- Model ---
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if is_qlora:
        quant_config = _build_quant_config("4bit", use_gpu, skip_modules=["classifier"])
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            config=config,
            ignore_mismatched_sizes=True,
            quantization_config=quant_config,
            device_map={"": torch_device},
        )
        # ViT uses pixel_values (not input_ids) as main input — base class does not
        # auto-enable input grads for non-LM models, so do it manually for QLoRA
        model.enable_input_require_grads()
    else:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            config=config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
        )
        model = model.to(torch_device)

    # ignore_mismatched_sizes doesn't reliably reinit the head when quantization_config
    # + device_map are used — the pretrained 1000-class weights get loaded, causing
    # pooled_logits.view(-1, num_labels) to reshape (16,1000) → (8000,2) and crash.
    # Explicit reinit guarantees the right output size in both paths.
    model.classifier = nn.Linear(model.config.hidden_size, num_labels).to(torch_device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # classifier head is randomly re-init (1000→N classes); unfreeze it directly
    # instead of modules_to_save to avoid PEFT wrapper reshaping logits incorrectly
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad_(True)
    model.print_trainable_parameters()

    # --- Training ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    print(f"\nStarting training for {epochs} epochs...")

    train_times, throughputs, cpu_rams, gpu_rams = [], [], [], []
    train_losses, val_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()

        # Peak CPU RAM: background poller thread
        stop_evt = threading.Event()
        holder = {}
        ram_thread = threading.Thread(target=_ram_poller, args=(stop_evt, holder), daemon=True)
        ram_thread.start()

        if use_gpu:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        total_loss = 0.0
        num_samples = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(torch_device)
            labels = batch["labels"].to(torch_device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            outputs.loss.backward()
            optimizer.step()

            total_loss += outputs.loss.item() * labels.size(0)
            num_samples += labels.size(0)

        epoch_time_s = time.perf_counter() - t0
        stop_evt.set()
        ram_thread.join()

        throughput = num_samples / epoch_time_s
        peak_cpu_mb = holder.get("peak_mb", 0.0)
        avg_loss = total_loss / num_samples

        # Validation
        model.eval()
        correct = total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(torch_device)
                labels = batch["labels"].to(torch_device)
                preds = model(pixel_values=pixel_values).logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc = correct / total_val

        train_times.append(epoch_time_s)
        throughputs.append(throughput)
        cpu_rams.append(peak_cpu_mb)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        if use_gpu:
            gpu_rams.append(torch.cuda.max_memory_allocated() / 1024**2)

        gpu_str = f" | gpu_ram={gpu_rams[-1]:.0f}MB" if use_gpu else ""
        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | t={epoch_time_s:.1f}s{gpu_str}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Summary ({method} | {device.upper()} | {epochs} epochs)")
    print(f"{'='*60}")
    print(f"  avg_train_time_s:   {round(float(np.mean(train_times)), 2)}")
    print(f"  avg_throughput_s:   {round(float(np.mean(throughputs)), 2)}")
    print(f"  avg_cpu_ram_mb:     {round(float(np.mean(cpu_rams)), 2)}")
    if use_gpu:
        print(f"  avg_gpu_ram_mb:     {round(float(np.mean(gpu_rams)), 2)}")
    print(f"  final_train_loss:   {round(train_losses[-1], 4)}")
    print(f"  final_val_accuracy: {round(val_accs[-1], 4)}")

    # --- Plot & Save ---
    os.makedirs(output_dir, exist_ok=True)
    _plot_metrics(output_dir, method, train_times, throughputs, cpu_rams,
                  gpu_rams if use_gpu else None, train_losses, val_accs)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nAdapter weights saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ViT fine-tuning with LoRA / QLoRA")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="Device to train on (default: cpu)")
    parser.add_argument("--quantize", choices=["none", "4bit"], default="none",
                        help="none = LoRA bf16 | 4bit = QLoRA nf4 (default: none)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    args = parser.parse_args()

    if args.device == "gpu" and not torch.cuda.is_available():
        raise SystemExit("--device gpu requested but no CUDA GPU is available.")

    finetune_vit(device=args.device, quantize=args.quantize, epochs=args.epochs)


if __name__ == "__main__":
    main()
