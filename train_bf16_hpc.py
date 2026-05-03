# =============================================================================
# train_bf16_hpc.py
# Finetune google/gemma-3-270m-it with LoRA (BF16/FP16/FP32 auto-selected)
# Optimized for HPC (Zaratan) — A100 GPU, no internet on compute node.
# Dataset : gbharti/finance-alpaca
# Metrics : training time/epoch, throughput (samples/sec), peak RAM (CPU+GPU)
# HPC     : compatible with SLURM — see job_bf16.sh
#
# Usage:
#   python train_bf16_hpc.py                        # full dataset, 3 epochs
#   python train_bf16_hpc.py --num_rows 500         # 500 rows
#   python train_bf16_hpc.py --epochs 5             # 5 epochs
#   python train_bf16_hpc.py --gpu                  # force GPU (error if unavailable)
#   python train_bf16_hpc.py --no_gpu               # force CPU
# =============================================================================

import os
import time
import threading
import warnings
import argparse

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# HPC FIX 1: Block ALL internet calls — compute node has no internet
# Must be set BEFORE any HuggingFace imports
# -----------------------------------------------------------------------------
os.environ["HF_OFFLINE"]             = "1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["DATASETS_OFFLINE"]       = "1"

# -----------------------------------------------------------------------------
# HPC FIX 2: Redirect HF cache to scratch (avoids filling 19GB home quota)
# -----------------------------------------------------------------------------
HF_CACHE = "/home/ziyadh10/scratch/gemma_project/hf_cache"
os.environ["HF_HOME"]                = HF_CACHE
os.environ["TRANSFORMERS_CACHE"]     = HF_CACHE
os.environ["HF_DATASETS_CACHE"]      = HF_CACHE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import psutil
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# =============================================================================
# 1. ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="Fine-tune Gemma-3 with LoRA (BF16/FP32) on HPC")
parser.add_argument("--num_rows", type=int, default=None,
                    help="Number of training rows (default: None = full dataset)")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs (default: 50)")
gpu_group = parser.add_mutually_exclusive_group()
gpu_group.add_argument("--gpu",    action="store_true", help="Force GPU (error if unavailable)")
gpu_group.add_argument("--no_gpu", action="store_true", help="Force CPU even if GPU available")
args = parser.parse_args()

# =============================================================================
# 2. CONSTANTS — HPC absolute paths
# =============================================================================
MODEL_ID     = "google/gemma-3-270m-it"
DATASET_NAME = "gbharti/finance-alpaca"

# -----------------------------------------------------------------------------
# HPC FIX 4: Absolute paths instead of relative "./" paths
# -----------------------------------------------------------------------------
OUTPUT_DIR   = "/home/ziyadh10/scratch/gemma_project/output_bf16"
DATASET_DIR  = "/home/ziyadh10/scratch/gemma_project/dataset"

MAX_SEQ_LEN  = 128
NUM_ROWS     = args.num_rows
NUM_EPOCHS   = args.epochs
BATCH_SIZE   = 1
LOG_STEPS    = 10

# Create output dir if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 3. DEVICE & DTYPE DETECTION
# =============================================================================
print("\n[1/6] Detecting hardware ...")

if args.no_gpu:
    HAS_GPU = False
    print("   --no_gpu flag set — forcing CPU")
elif args.gpu:
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu flag set but no CUDA GPU was found.")
    HAS_GPU = True
else:
    HAS_GPU = torch.cuda.is_available()

print(f"   NUM_ROWS={NUM_ROWS or 'full'}  |  NUM_EPOCHS={NUM_EPOCHS}")

if HAS_GPU:
    gpu_name   = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"   GPU detected : {gpu_name}  ({gpu_mem_gb:.1f} GB VRAM)")

    if torch.cuda.is_bf16_supported():
        COMPUTE_DTYPE = torch.bfloat16
        USE_BF16, USE_FP16 = True, False
        print("   Dtype        : bfloat16 (Ampere+ GPU — A100 ✓)")
    else:
        COMPUTE_DTYPE = torch.float16
        USE_BF16, USE_FP16 = False, True
        print("   Dtype        : float16 (pre-Ampere GPU)")

    DEVICE_MAP = "auto"
    NO_CUDA    = False
else:
    try:
        cpu_bf16 = torch.backends.cpu.get_default_dtype() == torch.bfloat16
    except AttributeError:
        cpu_bf16 = False

    COMPUTE_DTYPE = torch.bfloat16 if cpu_bf16 else torch.float32
    USE_BF16  = cpu_bf16
    USE_FP16  = False
    DEVICE_MAP = "cpu"
    NO_CUDA    = True
    print(f"   No GPU — running on CPU (dtype: {COMPUTE_DTYPE})")

# =============================================================================
# 4. RAM / VRAM MONITOR
# =============================================================================
class PeakRAMMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval     = interval
        self._proc        = psutil.Process(os.getpid())
        self.peak_cpu_mb  = 0.0
        self.peak_gpu_mb  = 0.0
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if HAS_GPU:
            torch.cuda.reset_peak_memory_stats()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / 1024 ** 2
            if rss > self.peak_cpu_mb:
                self.peak_cpu_mb = rss
            if HAS_GPU:
                gpu_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                if gpu_mb > self.peak_gpu_mb:
                    self.peak_gpu_mb = gpu_mb
            self._stop.wait(self.interval)

# =============================================================================
# 5. PER-EPOCH METRICS CALLBACK
# =============================================================================
class EpochMetricsCallback(TrainerCallback):
    def __init__(self, num_samples: int):
        self.num_samples    = num_samples
        self.epoch_start    = 0.0
        self.epoch_records  = []

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed    = time.time() - self.epoch_start
        throughput = self.num_samples / elapsed if elapsed > 0 else 0.0
        record = {
            "epoch":                       int(state.epoch),
            "time_s":                      round(elapsed, 2),
            "throughput_samples_per_sec":  round(throughput, 4),
        }
        self.epoch_records.append(record)
        print(
            f"\n  [Epoch {record['epoch']}]  "
            f"time={record['time_s']}s  |  "
            f"throughput={record['throughput_samples_per_sec']} samples/sec"
        )

# =============================================================================
# 6. LOAD TOKENIZER
# =============================================================================
print("[2/6] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# 7. LOAD MODEL
# =============================================================================
print("[3/6] Loading model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=COMPUTE_DTYPE,
    device_map=DEVICE_MAP,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False
print(f"   Model loaded — dtype: {COMPUTE_DTYPE}  |  device_map: {DEVICE_MAP}")

# =============================================================================
# 8. LORA ADAPTER CONFIG
# =============================================================================
print("[4/6] Attaching LoRA adapters ...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =============================================================================
# 9. LOAD & FORMAT DATASET
# =============================================================================
print("[5/6] Loading and formatting dataset ...")

split_str   = "train" if NUM_ROWS is None else f"train[:{NUM_ROWS}]"
raw_dataset = load_dataset(DATASET_NAME, split=split_str)

def format_row(example):
    instruction  = example.get("instruction", "")
    inp          = example.get("input", "")
    output       = example.get("output", "")
    user_content = f"{instruction}\n{inp}" if inp and inp.strip() else instruction
    return {
        "text": (
            f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n{output}<end_of_turn>"
        )
    }

formatted_dataset = raw_dataset.map(format_row, remove_columns=raw_dataset.column_names)
print(f"   Total samples : {len(formatted_dataset)}")
print(f"   Sample row    :\n   {formatted_dataset[0]['text'][:200]} ...")

# =============================================================================
# 10. TRAINING
# =============================================================================
print("[6/6] Starting training ...\n")

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    fp16=USE_FP16,
    bf16=USE_BF16,
    # -------------------------------------------------------------------------
    # HPC FIX 5: gradient_checkpointing saves VRAM significantly on large jobs
    # -------------------------------------------------------------------------
    gradient_checkpointing=True,
    logging_steps=LOG_STEPS,
    save_steps=500,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    # -------------------------------------------------------------------------
    # HPC FIX 6: adamw_torch — safe across all PyTorch versions
    # -------------------------------------------------------------------------
    optim="adamw_torch",
    no_cuda=NO_CUDA,
    dataloader_pin_memory=HAS_GPU,
)

_num_samples = len(formatted_dataset)
epoch_cb     = EpochMetricsCallback(num_samples=_num_samples)
ram_monitor  = PeakRAMMonitor(interval=0.5)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=sft_config,
    processing_class=tokenizer,
    callbacks=[epoch_cb],
)

ram_monitor.start()
start_time   = time.time()
train_result = trainer.train()
total_elapsed = time.time() - start_time
ram_monitor.stop()

# =============================================================================
# 11. SAVE ADAPTER
# =============================================================================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nAdapter saved to: {OUTPUT_DIR}")

# =============================================================================
# 12. RESULTS SUMMARY
# =============================================================================
final_loss         = train_result.training_loss
minutes, seconds   = divmod(int(total_elapsed), 60)
overall_throughput = (_num_samples * NUM_EPOCHS) / total_elapsed

print("\n" + "=" * 60)
print("  TRAINING METRICS SUMMARY  —  BF16")
print("=" * 60)
print(f"  Device              : {'GPU (' + torch.cuda.get_device_name(0) + ')' if HAS_GPU else 'CPU'}")
print(f"  Dtype used          : {COMPUTE_DTYPE}")
print(f"  Total training time : {minutes}m {seconds}s  ({total_elapsed:.1f}s)")
print(f"  Final training loss : {final_loss:.4f}")
print(f"  Overall throughput  : {overall_throughput:.4f} samples/sec")
print()
print("  Per-epoch breakdown:")
for r in epoch_cb.epoch_records:
    print(f"    Epoch {r['epoch']}  |  {r['time_s']}s  |  {r['throughput_samples_per_sec']} samples/sec")
print()
print(f"  Peak CPU RAM        : {ram_monitor.peak_cpu_mb:.1f} MB")
if HAS_GPU:
    print(f"  Peak GPU VRAM       : {ram_monitor.peak_gpu_mb:.1f} MB")
else:
    print(f"  Peak GPU VRAM       : N/A (CPU-only run)")
print("=" * 60)

# =============================================================================
# 13. QUICK INFERENCE TEST
# =============================================================================
print("\n[Inference Test] Loading saved adapter ...\n")
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=COMPUTE_DTYPE,
    device_map=DEVICE_MAP,
    low_cpu_mem_usage=True,
)
inf_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
inf_model.eval()

question = "What is the difference between a stock and a bond?"
prompt   = (
    f"<start_of_turn>user\n{question}<end_of_turn>\n"
    f"<start_of_turn>model\n"
)
inputs = tokenizer(prompt, return_tensors="pt")
if HAS_GPU:
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    output_ids = inf_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Question : {question}")
print(f"Answer   : {response}")
print("\n[Done] BF16 HPC script finished.")
