# =============================================================================
# train_bf16_local.py
# Finetune google/gemma-3-270m-it with LoRA (BF16/FP16/FP32 auto-selected)
# Runs on GPU if available, falls back to CPU automatically.
# Dataset : gbharti/finance-alpaca
# Metrics : training time/epoch, throughput (samples/sec), peak RAM (CPU+GPU)
# HPC     : compatible with SLURM — see run_bf16.slurm
#
# Usage:
#   python train_bf16_local.py                        # full dataset, 50 epochs, auto device
#   python train_bf16_local.py --num_rows 500         # 500 rows
#   python train_bf16_local.py --epochs 3             # 3 epochs
#   python train_bf16_local.py --gpu                  # force GPU (error if unavailable)
#   python train_bf16_local.py --no_gpu               # force CPU
# =============================================================================

import os
import time
import threading
import warnings
import argparse

warnings.filterwarnings("ignore")
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
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# =============================================================================
# 1. ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="Fine-tune Gemma-3 with LoRA (BF16/FP32)")
parser.add_argument("--num_rows", type=int, default=None,
                    help="Number of training rows to use (default: None = full dataset)")
parser.add_argument("--epochs",   type=int, default=50,
                    help="Number of training epochs (default: 50)")
gpu_group = parser.add_mutually_exclusive_group()
gpu_group.add_argument("--gpu",    action="store_true",  help="Force GPU (error if unavailable)")
gpu_group.add_argument("--no_gpu", action="store_true",  help="Force CPU even if GPU is available")
args = parser.parse_args()

# =============================================================================
# 2. CONSTANTS
# =============================================================================
MODEL_ID     = "google/gemma-3-270m-it"
DATASET_NAME = "gbharti/finance-alpaca"
OUTPUT_DIR   = "./output_bf16"
MAX_SEQ_LEN  = 128
NUM_ROWS     = args.num_rows      # None = full dataset
NUM_EPOCHS   = args.epochs        # default 50
BATCH_SIZE   = 1
LOG_STEPS    = 10

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
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"   GPU detected : {gpu_name}  ({gpu_mem_gb:.1f} GB VRAM)")

    # BF16 requires Ampere (sm_80) or newer; else fall back to FP16
    if torch.cuda.is_bf16_supported():
        COMPUTE_DTYPE = torch.bfloat16
        USE_BF16, USE_FP16 = True, False
        print("   Dtype        : bfloat16 (Ampere+ GPU)")
    else:
        COMPUTE_DTYPE = torch.float16
        USE_BF16, USE_FP16 = False, True
        print("   Dtype        : float16 (pre-Ampere GPU)")

    DEVICE_MAP = "auto"   # let HF place layers on available GPUs
    NO_CUDA    = False
else:
    # Check for CPU-native BF16 (AVX512_BF16)
    try:
        cpu_bf16 = torch.backends.cpu.get_default_dtype() == torch.bfloat16
    except AttributeError:
        cpu_bf16 = False

    COMPUTE_DTYPE = torch.bfloat16 if cpu_bf16 else torch.float32
    USE_BF16  = cpu_bf16
    USE_FP16  = False
    DEVICE_MAP = "cpu"
    NO_CUDA    = True
    print(f"   No GPU found — running on CPU  (dtype: {COMPUTE_DTYPE})")

# =============================================================================
# 3. RAM / VRAM MONITOR
# =============================================================================
class PeakRAMMonitor:
    """Polls process RSS (CPU) and torch CUDA peak alloc (GPU) every 0.5 s."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._proc = psutil.Process(os.getpid())
        self.peak_cpu_mb: float = 0.0
        self.peak_gpu_mb: float = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

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
# 4. PER-EPOCH METRICS CALLBACK
# =============================================================================
class EpochMetricsCallback(TrainerCallback):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.epoch_start: float = 0.0
        self.epoch_records: list[dict] = []

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - self.epoch_start
        throughput = self.num_samples / elapsed if elapsed > 0 else 0.0
        record = {
            "epoch": int(state.epoch),
            "time_s": round(elapsed, 2),
            "throughput_samples_per_sec": round(throughput, 4),
        }
        self.epoch_records.append(record)
        print(
            f"\n  [Epoch {record['epoch']}]  "
            f"time={record['time_s']}s  |  "
            f"throughput={record['throughput_samples_per_sec']} samples/sec"
        )


# =============================================================================
# 5. LOAD TOKENIZER
# =============================================================================
print("[2/6] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# 6. LOAD MODEL
# =============================================================================
print("[3/6] Loading model ...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=COMPUTE_DTYPE,
    device_map=DEVICE_MAP,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False
print(f"   Model loaded  — dtype: {COMPUTE_DTYPE}  |  device_map: {DEVICE_MAP}")

# =============================================================================
# 7. LORA ADAPTER CONFIG
# =============================================================================
print("[4/6] Attaching LoRA adapters ...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =============================================================================
# 8. LOAD & FORMAT DATASET
# =============================================================================
print("[5/6] Loading and formatting dataset ...")

raw_dataset = load_dataset(DATASET_NAME, split="train" if NUM_ROWS is None else f"train[:{NUM_ROWS}]")

def format_row(example):
    instruction = example.get("instruction", "")
    inp         = example.get("input", "")
    output      = example.get("output", "")
    user_content = f"{instruction}\n{inp}" if inp and inp.strip() else instruction
    return {
        "text": (
            f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n{output}<end_of_turn>"
        )
    }

formatted_dataset = raw_dataset.map(format_row, remove_columns=raw_dataset.column_names)
print(f"   Sample row:\n   {formatted_dataset[0]['text'][:200]} ...")

# =============================================================================
# 9. TRAINING
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
    logging_steps=LOG_STEPS,
    save_steps=500,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    optim="adamw_torch",
    no_cuda=NO_CUDA,
    dataloader_pin_memory=HAS_GPU,   # pin memory only helps on GPU
)

_num_samples = len(formatted_dataset)
epoch_cb    = EpochMetricsCallback(num_samples=_num_samples)
ram_monitor = PeakRAMMonitor(interval=0.5)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=sft_config,
    processing_class=tokenizer,
    callbacks=[epoch_cb],
)

ram_monitor.start()
start_time = time.time()
train_result = trainer.train()
total_elapsed = time.time() - start_time
ram_monitor.stop()

# =============================================================================
# 10. SAVE ADAPTER
# =============================================================================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nAdapter saved to: {OUTPUT_DIR}")

# =============================================================================
# 11. RESULTS SUMMARY
# =============================================================================
final_loss = train_result.training_loss
minutes, seconds = divmod(int(total_elapsed), 60)
overall_throughput = (_num_samples * NUM_EPOCHS) / total_elapsed

print("\n" + "=" * 60)
print("  TRAINING METRICS SUMMARY")
print("=" * 60)
print(f"  Device                  : {'GPU (' + torch.cuda.get_device_name(0) + ')' if HAS_GPU else 'CPU'}")
print(f"  Dtype used              : {COMPUTE_DTYPE}")
print(f"  Total training time     : {minutes}m {seconds}s  ({total_elapsed:.1f}s)")
print(f"  Final training loss     : {final_loss:.4f}")
print(f"  Overall throughput      : {overall_throughput:.4f} samples/sec")
print()
print("  Per-epoch breakdown:")
for r in epoch_cb.epoch_records:
    print(f"    Epoch {r['epoch']}  |  {r['time_s']}s  |  {r['throughput_samples_per_sec']} samples/sec")
print()
print(f"  Peak CPU RAM usage      : {ram_monitor.peak_cpu_mb:.1f} MB")
if HAS_GPU:
    print(f"  Peak GPU VRAM usage     : {ram_monitor.peak_gpu_mb:.1f} MB")
else:
    print(f"  Peak GPU VRAM usage     : N/A (CPU-only run)")
print("=" * 60)

# =============================================================================
# 12. QUICK INFERENCE TEST
# =============================================================================
print("\n[Inference Test] Loading saved adapter for a quick finance question ...\n")

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
prompt = (
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
print("\n[Done] BF16/float32 script finished.")
