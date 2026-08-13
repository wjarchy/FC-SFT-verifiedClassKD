#!/usr/bin/env python3
"""GPU smoke test: fixtures, 0.6B SFT backward, and standalone Class-KD math."""

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
from fc_class_kd.scoring.aggregation import class_distribution_tensor
from fc_class_kd.training.candidate_scoring import score_candidate_sequences
from fc_class_kd.training.model_utils import attach_lora, load_student_model, max_memory_mb
from fc_class_kd.types import PRIMARY_CLASSES, FCRecord
from fc_class_kd.verifier.classifier import classify_call

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
            res = classify_call(c.serialized_call, rec.tools, rec.gold_call)
            print(f"    {c.source} -> {res.class_label.value}")

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

    if not torch.isfinite(loss_sft):
        print("SMOKE_FAIL: SFT loss is NaN/Inf")
        return 1
    loss_sft.backward()

    has_lora_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    frozen_has_grad = any(p.grad is not None for p in model.parameters() if not p.requires_grad)
    if not has_lora_grad:
        print("SMOKE_FAIL: no LoRA gradients after SFT backward")
        return 1
    if frozen_has_grad:
        print("SMOKE_FAIL: frozen base parameters received gradients")
        return 1

    model.zero_grad(set_to_none=True)
    kd_candidates = build_rule_candidates(rec, seed=42)[:4]
    class_lookup = {label: index for index, label in enumerate(PRIMARY_CLASSES)}
    class_indices = torch.tensor(
        [[class_lookup.get(classify_call(c.serialized_call, rec.tools, rec.gold_call).class_label, -1)
          for c in kd_candidates]],
        device=model.device,
    )
    student_scores = score_candidate_sequences(
        model,
        tokenizer,
        rec,
        kd_candidates,
        max_length=int(cfg.training.max_length),
        alpha=float(cfg.scoring.alpha),
    ).unsqueeze(0)
    student_candidate_probs = torch.softmax(student_scores, dim=-1)
    student_class_probs = class_distribution_tensor(
        student_candidate_probs,
        class_indices,
        len(PRIMARY_CLASSES),
    )
    # A fixed, deliberately different teacher distribution keeps this an engineering
    # gradient test; formal teacher probabilities are cached in PRD Step 6.
    teacher_candidate_probs = torch.softmax(
        student_scores.detach() + torch.tensor([[0.4, -0.2, 0.3, -0.1]], device=model.device),
        dim=-1,
    )
    teacher_class_probs = class_distribution_tensor(
        teacher_candidate_probs,
        class_indices,
        len(PRIMARY_CLASSES),
    )
    loss_kd = class_kd_loss(
        torch.log(student_class_probs.clamp_min(1e-12)),
        teacher_class_probs,
        temperature=1.0,
    )
    loss_kd.backward()
    has_end_to_end_lora_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    if not torch.isfinite(loss_kd) or not has_end_to_end_lora_grad:
        print("SMOKE_FAIL: end-to-end Class KD → LoRA gradient check failed")
        return 1
    print(f"End-to-end Class KD loss: {loss_kd.item():.6f}")
    print("End-to-end Class KD → candidate scores → LoRA gradient: OK")

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
