# =============================================================================
# train_colab.py
# Finetune google/gemma-3-270m-it on Google Colab
# Supports: INT4 + BF16 quantization, CPU + GPU device
#
# Install first:
#   !pip install transformers datasets peft accelerate bitsandbytes trl sentencepiece psutil
#
# Usage:
#   !python train_colab.py --quant int4 --device cpu
#   !python train_colab.py --quant int4 --device gpu
#   !python train_colab.py --quant bf16 --device cpu
#   !python train_colab.py --quant bf16 --device gpu
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
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# =============================================================================
# 1. ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="Gemma 3 270M Finetuning — Colab")
parser.add_argument("--quant",     type=str, choices=["int4", "bf16"], required=True,
                    help="Quantization: int4 or bf16")
parser.add_argument("--device",    type=str, choices=["cpu", "gpu"],   required=True,
                    help="Device: cpu or gpu")
parser.add_argument("--num_rows",  type=int, default=100,
                    help="Number of training rows (default: 100)")
parser.add_argument("--epochs",    type=int, default=3,
                    help="Number of epochs (default: 3)")
args = parser.parse_args()

# =============================================================================
# 2. CONSTANTS
# =============================================================================
MODEL_ID     = "google/gemma-3-270m-it"
DATASET_NAME = "gbharti/finance-alpaca"
OUTPUT_DIR   = f"./output_{args.quant}_{args.device}_colab"
MAX_SEQ_LEN  = 64
NUM_ROWS     = args.num_rows
NUM_EPOCHS   = args.epochs
BATCH_SIZE   = 1
LOG_STEPS    = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 3. DEVICE SETUP
# =============================================================================
print("\n" + "=" * 60)
print(f"  Gemma 3 270M | quant={args.quant.upper()} | device={args.device.upper()}")
print("=" * 60)

CUDA_AVAILABLE = torch.cuda.is_available()

if args.device == "gpu":
    if not CUDA_AVAILABLE:
        print("  ⚠️  WARNING: No GPU found!")
        print("  Go to: Runtime → Change runtime type → T4 GPU")
        print("  Falling back to CPU ...\n")
        HAS_GPU    = False
        DEVICE_MAP = "cpu"
        NO_CUDA    = True
    else:
        HAS_GPU    = True
        DEVICE_MAP = "auto"
        NO_CUDA    = False
        gpu_name   = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  GPU : {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")
else:
    HAS_GPU    = False
    DEVICE_MAP = "cpu"
    NO_CUDA    = True
    print(f"  Device : CPU")

# =============================================================================
# 4. DTYPE SETUP
# =============================================================================
if args.quant == "bf16":
    if HAS_GPU:
        COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        USE_BF16      = torch.cuda.is_bf16_supported()
        USE_FP16      = not USE_BF16
    else:
        # Test if CPU supports true BF16
        try:
            _x = torch.randn(2, 2).to(torch.bfloat16)
            _y = torch.matmul(_x, _x)
            assert _y.dtype == torch.bfloat16
            COMPUTE_DTYPE = torch.bfloat16
            USE_BF16      = True
            USE_FP16      = False
            print("  ✅ True BF16 supported on this CPU!")
        except Exception:
            COMPUTE_DTYPE = torch.float32
            USE_BF16      = False
            USE_FP16      = False
            print("  ⚠️  BF16 not supported on CPU → falling back to float32")
else:  # int4
    COMPUTE_DTYPE = torch.float32 if not HAS_GPU else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    USE_BF16 = False
    USE_FP16 = False

print(f"  Quant  : {args.quant.upper()}")
print(f"  Dtype  : {COMPUTE_DTYPE}")
print(f"  Rows   : {NUM_ROWS}")
print(f"  Epochs : {NUM_EPOCHS}")
print("=" * 60 + "\n")

# =============================================================================
# 5. RAM / VRAM MONITOR
# =============================================================================
class PeakRAMMonitor:
    def __init__(self, interval=0.5):
        self.interval    = interval
        self._proc       = psutil.Process(os.getpid())
        self.peak_cpu_mb = 0.0
        self.peak_gpu_mb = 0.0
        self._stop       = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)

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
# 6. PER-EPOCH CALLBACK
# =============================================================================
class EpochMetricsCallback(TrainerCallback):
    def __init__(self, num_samples):
        self.num_samples   = num_samples
        self.epoch_start   = 0.0
        self.epoch_records = []

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed    = time.time() - self.epoch_start
        throughput = self.num_samples / elapsed if elapsed > 0 else 0.0
        record = {
            "epoch":      int(state.epoch),
            "time_s":     round(elapsed, 2),
            "throughput": round(throughput, 4),
        }
        self.epoch_records.append(record)
        print(
            f"\n  [Epoch {record['epoch']}] "
            f"time={record['time_s']}s | "
            f"throughput={record['throughput']} samples/sec"
        )

# =============================================================================
# 7. LOAD TOKENIZER
# =============================================================================
print("[1/5] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("      Tokenizer loaded ✓")

# =============================================================================
# 8. LOAD MODEL
# =============================================================================
print(f"[2/5] Loading model ({args.quant.upper()}) ...")

quant_used = args.quant.upper()

if args.quant == "int4":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map=DEVICE_MAP,
            low_cpu_mem_usage=True,
        )
        print("      Model loaded with INT4 (nf4) ✓")
    except Exception as e:
        print(f"      ⚠️  INT4 failed ({e}), falling back to float32")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map=DEVICE_MAP,
            low_cpu_mem_usage=True,
        )
        quant_used = "float32 fallback"

    model = prepare_model_for_kbit_training(model)

else:  # bf16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=COMPUTE_DTYPE,
        device_map=DEVICE_MAP,
        low_cpu_mem_usage=True,
    )
    print(f"      Model loaded with {COMPUTE_DTYPE} ✓")

model.config.use_cache = False

# =============================================================================
# 9. LORA CONFIG
# =============================================================================
print("[3/5] Attaching LoRA adapters ...")
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
# 10. DATASET
# =============================================================================
print(f"[4/5] Loading dataset ({NUM_ROWS} rows) ...")
raw_dataset = load_dataset(DATASET_NAME, split=f"train[:{NUM_ROWS}]")

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
print(f"      Dataset ready: {len(formatted_dataset)} rows ✓")

# =============================================================================
# 11. TRAIN
# =============================================================================
print("[5/5] Starting training ...\n")

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    fp16=USE_FP16,
    bf16=USE_BF16,
    gradient_checkpointing=True,
    logging_steps=LOG_STEPS,
    save_steps=500,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    optim="adamw_torch",
    use_cpu=NO_CUDA,
    dataloader_pin_memory=HAS_GPU,
)

_num_samples = len(formatted_dataset)
epoch_cb     = EpochMetricsCallback(num_samples=_num_samples)
ram_monitor  = PeakRAMMonitor()

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=sft_config,
    processing_class=tokenizer,
    callbacks=[epoch_cb],
)

ram_monitor.start()
start_time    = time.time()
train_result  = trainer.train()
total_elapsed = time.time() - start_time
ram_monitor.stop()

# =============================================================================
# 12. SAVE
# =============================================================================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nAdapter saved to: {OUTPUT_DIR}")

# =============================================================================
# 13. RESULTS
# =============================================================================
final_loss         = train_result.training_loss
minutes, seconds   = divmod(int(total_elapsed), 60)
overall_throughput = (_num_samples * NUM_EPOCHS) / total_elapsed

print("\n" + "=" * 60)
print(f"  RESULTS — {args.quant.upper()} on {args.device.upper()}")
print("=" * 60)
print(f"  Device              : {'GPU (' + torch.cuda.get_device_name(0) + ')' if HAS_GPU else 'CPU'}")
print(f"  Quantization        : {quant_used}")
print(f"  Compute dtype       : {COMPUTE_DTYPE}")
print(f"  Total training time : {minutes}m {seconds}s ({total_elapsed:.1f}s)")
print(f"  Final training loss : {final_loss:.4f}")
print(f"  Overall throughput  : {overall_throughput:.4f} samples/sec")
print()
print("  Per-epoch breakdown:")
for r in epoch_cb.epoch_records:
    print(f"    Epoch {r['epoch']} | {r['time_s']}s | {r['throughput']} samples/sec")
print()
print(f"  Peak CPU RAM        : {ram_monitor.peak_cpu_mb:.1f} MB")
if HAS_GPU:
    print(f"  Peak GPU VRAM       : {ram_monitor.peak_gpu_mb:.1f} MB")
else:
    print(f"  Peak GPU VRAM       : N/A")
print("=" * 60)

# =============================================================================
# 14. INFERENCE TEST
# =============================================================================
print("\n[Inference Test]\n")
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
prompt   = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
inputs   = tokenizer(prompt, return_tensors="pt")
if HAS_GPU:
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    output_ids = inf_model.generate(
        **inputs, max_new_tokens=80,
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Q: {question}")
print(f"A: {response}")
print(f"\n[Done] {args.quant.upper()} on {args.device.upper()} finished.")
