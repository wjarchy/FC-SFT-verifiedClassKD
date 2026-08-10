#!/usr/bin/env python3
"""PRD §11 smoke test: 4 samples, verifier, 0.6B forward/backward, Class KD toy loss."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from fc_class_kd.candidates.rules import build_rule_candidates
from fc_class_kd.config import load_config
from fc_class_kd.data.splits import read_split
from fc_class_kd.losses.kd_losses import class_kd_loss
from fc_class_kd.prompting.templates import encode_sft_pair
from fc_class_kd.training.model_utils import attach_lora, load_student_model, max_memory_mb
from fc_class_kd.types import FCRecord, PRIMARY_CLASSES
from fc_class_kd.verifier.classifier import classify_function_call

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "smoke_4.jsonl"


def load_smoke_records(source: str, split_path: Path, samples: int) -> tuple[list[FCRecord], str]:
    if source == "fixture":
        path = FIXTURE_PATH
        records = read_split(path)[:samples]
        return records, str(path)

    if source == "split":
        if not split_path.exists():
            raise FileNotFoundError(f"Split not found: {split_path}")
        records = read_split(split_path)[:samples]
        return records, str(split_path)

    # auto: prefer real train split, fallback to bundled fixtures
    if split_path.exists():
        records = read_split(split_path)[:samples]
        return records, str(split_path)
    if FIXTURE_PATH.exists():
        records = read_split(FIXTURE_PATH)[:samples]
        print(f"Note: {split_path} missing — using bundled fixtures (no full dataset needed)")
        return records, str(FIXTURE_PATH)
    raise FileNotFoundError(
        f"No data: neither {split_path} nor {FIXTURE_PATH}. "
        "Run build_data.py or use --data-source fixture"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rtx2080ti")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--data-source",
        choices=["auto", "split", "fixture"],
        default="auto",
        help="auto: train.jsonl if present, else data/fixtures/smoke_4.jsonl",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SMOKE_FAIL: CUDA not available — need a machine with NVIDIA GPU")
        return 1

    cfg = load_config(args.config)
    model_id = args.model or cfg.models.debug_student
    split_path = Path(cfg.paths.splits) / "train.jsonl"

    records, data_path = load_smoke_records(args.data_source, split_path, args.samples)
    print(f"Loaded {len(records)} samples from {data_path}")

    for rec in records:
        cands = build_rule_candidates(rec, seed=42)
        print(f"  [{rec.id}] rule candidates: {len(cands)}")
        for c in cands[:3]:
            res = classify_function_call(c, rec.tools, rec.gold_call)
            print(f"    -> {res.class_label.value}")

    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_student_model(
        model_id,
        load_in_4bit=True,
        fp16=bool(cfg.hardware.get("fp16", True)),
        bf16=bool(cfg.hardware.get("bf16", False)),
    )
    model = attach_lora(
        model,
        r=cfg.models.qlora.r,
        lora_alpha=cfg.models.qlora.lora_alpha,
        lora_dropout=cfg.models.qlora.lora_dropout,
        target_modules=list(cfg.models.qlora.target_modules),
    )
    model.train()

    rec = records[0]
    input_ids, labels = encode_sft_pair(tokenizer, rec, max_length=int(cfg.training.max_length))
    batch = {
        "input_ids": torch.tensor([input_ids], device=model.device),
        "labels": torch.tensor([labels], device=model.device),
    }
    out = model(**batch)
    loss_sft = out.loss
    print(f"SFT loss: {loss_sft.item():.4f}")

    t_probs = torch.tensor([[0.5, 0.25, 0.15, 0.1]], device=model.device)
    s_log = torch.log(t_probs * 0.9 + 0.025)
    loss_kd = class_kd_loss(s_log, t_probs, temperature=1.0)
    loss_kd.backward()
    print(f"Class KD toy loss: {loss_kd.item():.4f}")

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
    )
    if not has_grad:
        print("SMOKE_FAIL: no LoRA gradients")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        adapter_dir = Path(tmp) / "adapter"
        model.save_pretrained(adapter_dir)
        meta = {"model_id": model_id, "samples": args.samples, "data_path": data_path}
        (adapter_dir / "smoke_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert (adapter_dir / "adapter_config.json").exists()
        print(f"Adapter saved OK ({adapter_dir})")

    peak_mb = max_memory_mb()
    print(f"Peak GPU memory: {peak_mb:.1f} MB")
    print("SMOKE_OK=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
