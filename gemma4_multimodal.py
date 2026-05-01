import argparse
import time
import os

# Prevent Windows socket access violations by forcing offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import psutil
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import faulthandler
faulthandler.enable()

# Utility to track memory based on the active device
def get_memory_usage(device):
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)  
    else:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)  

# Callback to measure Latency, Throughput, and Memory
class BenchmarkCallback(TrainerCallback):
    def __init__(self, device):
        self.device = device
        self.start_time = 0
        self.step_times = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.step_times.append(time.time() - self.start_time)

    def on_train_end(self, args, state, control, **kwargs):
        avg_time_per_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        throughput = (args.train_batch_size * args.gradient_accumulation_steps) / avg_time_per_step if avg_time_per_step > 0 else 0
        mem_gb = get_memory_usage(self.device)
        
        print("\n" + "="*40)
        print("📊 TRAINING BENCHMARK RESULTS")
        print("="*40)
        print(f"Hardware        : {self.device.upper()}")
        print(f"Step Latency    : {avg_time_per_step:.4f} seconds/step")
        print(f"Throughput      : {throughput:.2f} samples/second")
        print(f"Peak Memory     : {mem_gb:.2f} GB")
        print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmarking Gemma 4 E2B Multimodal")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--quantization", type=str, choices=["bf16", "int4"], default="int4")
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--max_steps", type=int, default=30) 
    args = parser.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    quantization = args.quantization
    
    if device == "cpu" and quantization == "int4":
        print("\n⚠️ WARNING: CPU training does not support 4-bit quantization. Forcing fallback to BF16.")
        quantization = "bf16"

    print(f"\n🚀 Initializing Benchmark on {device.upper()} with {quantization.upper()} precision...\n")

    model_id = "google/gemma-4-e2b-it" 
    
    print("⏳ Configuring Quantization Engine...")
    if quantization == "int4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True 
        )
        torch_dtype = torch.bfloat16
    else:
        quant_config = None
        torch_dtype = torch.bfloat16

    print("⏳ Downloading/Loading Processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    target_device_map = "auto" if device == "cuda" else {"": "cpu"}
    
    print("⏳ Downloading/Loading Model Weights (This takes time!)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map=target_device_map, 
        quantization_config=quant_config,
        dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    print("✅ Model successfully loaded into memory!")

    print("⏳ Configuring LoRA Adapters...")
    if quantization == "int4":
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("⏳ Loading SLAKE Dataset...")
    dataset = load_dataset("mdwiratathya/SLAKE-vqa-english", split="train[:500]")

    def collate_fn(examples):
        texts = []
        images = []
        for ex in examples:
            # THE FIX: Use official Chat Templates so the processor injects the exact correct image tokens
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex['question']}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ex['answer']}
                    ]
                }
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False)
            texts.append(prompt)
            images.append(ex['image'].convert("RGB"))
            
        batch = processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        labels = batch["input_ids"].clone()
        if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
        batch["labels"] = labels
        return {k: v.to(device) for k, v in batch.items()}

    training_args = TrainingArguments(
        output_dir="./gemma4-e2b-benchmark",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        bf16=(device == "cuda"), 
        logging_steps=5,
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        remove_unused_columns=False 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[BenchmarkCallback(device=device)]
    )

    print("🔥 Starting Training Benchmark...")
    trainer.train()

    print("\n🔍 Testing Generation Latency & Output...")
    model.eval()
    sample = dataset[0]
    
    # THE FIX: Apply the same Chat Template logic to the inference step
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample['question']}
            ]
        }
    ]
    # add_generation_prompt=True tells the model to prepare for the assistant's turn
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"User Question: {sample['question']}")
    print(f"True Answer from Dataset: {sample['answer']}")
    
    inputs = processor(
        text=prompt_text, 
        images=sample['image'].convert("RGB"), 
        return_tensors="pt"
    ).to(device)

    start_infer = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    end_infer = time.time()
    
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n⏱️ Inference Latency: {end_infer - start_infer:.4f} seconds")
    print("="*40)
    print(f"🤖 Model Output:\n{generated_text}")
    print("="*40)

if __name__ == "__main__":
    main()