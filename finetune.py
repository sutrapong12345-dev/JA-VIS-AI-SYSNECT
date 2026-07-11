"""J.A.R.V.I.S. quality-gated LoRA fine-tuning entrypoint.

Run validation on Windows/backend without GPU dependencies:
    python finetune.py --check-only

Run training inside the dedicated WSL/Conda Unsloth environment:
    python finetune.py --train
"""

from __future__ import annotations

import argparse
import json
import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(ROOT_DIR, "training", "dataset_train.jsonl")
VALIDATION_PATH = os.path.join(ROOT_DIR, "training", "dataset_validation.jsonl")
FULL_DATASET_PATH = os.path.join(ROOT_DIR, "training", "dataset_sysnect.jsonl")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "jarvis_lora")
GGUF_DIR = os.path.join(ROOT_DIR, "outputs", "jarvis_finetuned")


def check_quality() -> dict:
    from backend.training_quality import inspect_jsonl

    report = inspect_jsonl(FULL_DATASET_PATH)
    report.pop("results", None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def train(args: argparse.Namespace) -> None:
    report = check_quality()
    if not report.get("ready_for_training") and not args.force:
        raise SystemExit(
            "Dataset ยังไม่ผ่าน Quality Gate: ต้องมีอย่างน้อย 100 ตัวอย่าง "
            "คะแนนเฉลี่ย >= 80 และอัตราผ่าน >= 80% (ใช้ --force เฉพาะเมื่อยอมรับความเสี่ยงแล้ว)"
        )
    if not os.path.isfile(TRAIN_PATH):
        raise SystemExit("ไม่พบ training/dataset_train.jsonl กรุณาสร้างชุดฝึกจากหน้า Admin ก่อน")

    try:
        import torch
        from datasets import load_dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "ไม่พบ dependency สำหรับ GPU training กรุณาใช้ WSL/Conda environment "
            "และติดตั้ง Unsloth, TRL, PEFT, Accelerate และ BitsAndBytes"
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    train_dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")
    eval_dataset = None
    if os.path.isfile(VALIDATION_PATH) and os.path.getsize(VALIDATION_PATH) > 0:
        eval_dataset = load_dataset("json", data_files=VALIDATION_PATH, split="train")

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=training_args,
    )
    trainer.train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
    print(f"Training complete: {GGUF_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. quality-gated LoRA trainer")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="ตรวจ dataset โดยไม่โหลด GPU libraries")
    mode.add_argument("--train", action="store_true", help="เริ่ม LoRA fine-tuning หลังผ่าน Quality Gate")
    parser.add_argument("--force", action="store_true", help="ข้าม readiness gate (ไม่แนะนำ)")
    parser.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.check_only:
        quality = check_quality()
        sys.exit(0 if quality.get("ready_for_training") else 2)
    train(arguments)
