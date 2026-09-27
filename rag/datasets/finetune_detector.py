"""LoRA-tune Llama 3.1 8B as a one-call RAGTruth detector, then save a merged model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from rag.datasets.detector import detector_messages
from rag.datasets.ragtruth import export_split, read_examples


def _rows(tokenizer, examples: List[dict], max_length: int) -> List[dict]:
    rows = []
    for example in examples:
        messages = detector_messages(example, with_label=True)
        prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            continue
        labels = list(full_ids)
        for index in range(len(prompt_ids)):
            labels[index] = -100
        rows.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune an 8B RAGTruth detector with LoRA")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args(argv)

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Fine-tuning needs peft. On the GPU node run: pip install peft") from exc

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.jsonl"
    if not train_path.exists():
        counts = export_split(args.root, train_path, "train")
        print(counts, flush=True)
    examples = read_examples(train_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.getenv("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = _rows(tokenizer, examples, args.max_length)
    print(f"Training examples: {len(encoded)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        token=os.getenv("HF_TOKEN"),
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    class Encoded(torch.utils.data.Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=20,
            save_strategy="epoch",
            report_to=[],
            remove_unused_columns=False,
            gradient_checkpointing=True,
        ),
        train_dataset=Encoded(),
        data_collator=_pad_collator(tokenizer),
    )
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged_dir = args.out / "merged"
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model: {merged_dir}", flush=True)


def _pad_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(features):
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    return collate


if __name__ == "__main__":
    main()
