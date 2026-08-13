#!/usr/bin/env python3
"""Overfit a deterministic tiny ToolACE subset and verify train-set AST accuracy."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.evaluation.metrics import evaluate_predictions
from fc_class_kd.prompting.templates import encode_sft_pair
from fc_class_kd.training.model_utils import attach_lora, load_student_model, max_memory_mb
from fc_class_kd.training.sft import (
    collate_sft_examples,
    encode_sft_records,
    generate_function_call,
)
from fc_class_kd.types import FCRecord


def select_tiny_records(
    records: list[FCRecord],
    tokenizer,
    count: int,
    max_length: int,
) -> tuple[list[FCRecord], list[dict[str, int | str]]]:
    """Select the first deterministic records that fit; never truncate a gold call."""
    selected: list[FCRecord] = []
    skipped: list[dict[str, int | str]] = []
    for record in records:
        try:
            input_ids, _ = encode_sft_pair(tokenizer, record, max_length=max_length)
        except ValueError as exc:
            skipped.append({"sample_id": record.id, "reason": str(exc)})
            continue
        selected.append(record)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} records fit max_length={max_length}; requested {count}"
        )
    return selected, skipped


def evaluate_train_ast(model, tokenizer, records: list[FCRecord], max_new_tokens: int):
    model.eval()
    predictions = [
        generate_function_call(model, tokenizer, record, max_new_tokens=max_new_tokens)
        for record in records
    ]
    metrics = evaluate_predictions(records, predictions)
    model.train()
    return metrics.rates(), predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rtx4090")
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--required-ast", type=float, default=0.90)
    parser.add_argument("--out", default="outputs/tiny_overfit")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("TINY_OVERFIT_FAIL: CUDA not available")
        return 1
    if not (1 <= args.samples <= 32):
        raise ValueError("--samples must be between 1 and 32")
    if args.steps <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")

    cfg = load_config(args.config)
    max_length = args.max_length or int(cfg.training.max_length)
    model_id = args.model or cfg.models.debug_student
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    split_path = Path(cfg.paths.splits) / f"{args.split}.jsonl"
    all_records = read_split(split_path)
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_student_model(
        model_id,
        load_in_4bit=True,
        fp16=bool(cfg.hardware.get("fp16", False)),
        bf16=bool(cfg.hardware.get("bf16", True)),
    )
    model = attach_lora(
        model,
        r=int(cfg.models.qlora.r),
        lora_alpha=int(cfg.models.qlora.lora_alpha),
        lora_dropout=float(cfg.models.qlora.lora_dropout),
        target_modules=list(cfg.models.qlora.target_modules),
    )
    model.train()

    records, skipped = select_tiny_records(all_records, tokenizer, args.samples, max_length)
    examples = encode_sft_records(tokenizer, records, max_length=max_length)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must have a pad_token_id")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    initial_rates, initial_predictions = evaluate_train_ast(
        model, tokenizer, records, args.max_new_tokens
    )
    losses: list[float] = []
    start = time.time()
    order = list(range(len(examples)))
    rng = random.Random(args.seed)
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        example = examples[order[step % len(order)]]
        batch = collate_sft_examples([example], tokenizer.pad_token_id, device=model.device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % 25 == 0:
            print(f"step={step + 1}/{args.steps} loss={losses[-1]:.6f}")

    final_rates, final_predictions = evaluate_train_ast(
        model, tokenizer, records, args.max_new_tokens
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")
    report = {
        "kind": "tiny_overfit_engineering_gate",
        "model": model_id,
        "split": str(split_path),
        "sample_ids": [record.id for record in records],
        "skipped_before_selection": skipped,
        "seed": args.seed,
        "samples": args.samples,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "max_length": max_length,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "initial_metrics": initial_rates,
        "final_metrics": final_rates,
        "required_ast": args.required_ast,
        "passed": final_rates.get("ast_accuracy", 0.0) >= args.required_ast,
        "elapsed_seconds": time.time() - start,
        "peak_gpu_memory_mb": max_memory_mb(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record, before, after in zip(records, initial_predictions, final_predictions):
            handle.write(
                json.dumps(
                    {"sample_id": record.id, "before": before, "after": after},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        print("TINY_OVERFIT_FAIL: train AST gate not reached")
        return 1
    print("TINY_OVERFIT_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
