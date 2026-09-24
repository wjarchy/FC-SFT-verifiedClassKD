from __future__ import annotations

import math

import torch

from fc_class_kd.scoring.aggregation import decompose_kl_by_class_ids
from fc_class_kd.types import FCRecord, FunctionCall, ToolParameter, ToolSchema
from fc_class_kd.v7.candidates import build_k4_pool, candidate_class_ids
from fc_class_kd.v7.equivalence import build_strict_equivalent, canonical_signature


def _record(arguments, parameters, *, metadata=None):
    return FCRecord(
        id="v7",
        source="fixture",
        query="test",
        tools=[
            ToolSchema(name="target", parameters=parameters),
            ToolSchema(
                name="other",
                parameters={"q": ToolParameter(type="string", required=True)},
            ),
        ],
        gold_call=FunctionCall(name="target", arguments=arguments),
        metadata=metadata or {},
    )


def test_v7_category_a_object_order():
    record = _record(
        {"city": "Singapore", "unit": "C"},
        {
            "city": ToolParameter(type="string", required=True),
            "unit": ToolParameter(type="string", required=True),
        },
    )
    variant = build_strict_equivalent(record)
    assert variant is not None and variant.category == "A"
    assert list(variant.call.arguments) != list(record.gold_call.arguments)
    assert canonical_signature(record, variant.call) == canonical_signature(
        record, record.gold_call
    )


def test_v7_category_b_number_spelling():
    record = _record(
        {"temperature": 20},
        {"temperature": ToolParameter(type="number", required=True)},
    )
    variant = build_strict_equivalent(record)
    assert variant is not None and variant.category == "B"
    assert isinstance(variant.call.arguments["temperature"], float)
    assert canonical_signature(record, variant.call) == canonical_signature(
        record, record.gold_call
    )


def test_v7_category_c_explicit_runtime_default():
    record = _record(
        {"city": "Singapore", "unit": "C"},
        {
            "city": ToolParameter(type="string", required=True),
            "unit": ToolParameter(type="string", required=False),
        },
        metadata={"v7_safe_defaults": {"unit": "C"}},
    )
    variant = build_strict_equivalent(record)
    assert variant is not None and variant.category == "C"
    assert "unit" not in variant.call.arguments
    assert canonical_signature(record, variant.call) == canonical_signature(
        record, record.gold_call
    )


def test_v7_k4_has_one_equivalent_and_two_functional_negatives():
    record = _record(
        {"city": "Singapore", "unit": "C"},
        {
            "city": ToolParameter(type="string", required=True),
            "unit": ToolParameter(type="string", required=True),
        },
    )
    pool = build_k4_pool(record)
    assert [candidate.source.split(":", 1)[0] for candidate in pool] == [
        "gold",
        "equivalent_A",
        "wrong_argument",
        "wrong_function",
    ]
    class_ids = candidate_class_ids(record, pool)
    assert class_ids[0] == class_ids[1]
    assert len(set(class_ids)) == 3


def test_candidate_kl_decomposition_with_functional_class_ids():
    teacher = [0.50, 0.20, 0.20, 0.10]
    student = [0.25, 0.35, 0.25, 0.15]
    class_ids = ["correct", "correct", "wrong_arg", "wrong_fn"]
    candidate, inter, intra = decompose_kl_by_class_ids(teacher, student, class_ids)
    assert math.isclose(candidate, inter + intra, rel_tol=1e-9, abs_tol=1e-9)


def test_candidate_and_class_losses_backpropagate():
    logits = torch.tensor([[0.2, -0.1, 0.3, -0.4]], requires_grad=True)
    teacher = torch.tensor([[0.5, 0.2, 0.2, 0.1]])
    student = torch.softmax(logits / 2.0, dim=-1)
    candidate_loss = torch.sum(teacher * (torch.log(teacher) - torch.log(student))) * 4.0
    candidate_loss.backward(retain_graph=True)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    logits.grad.zero_()
    student_class = torch.stack((student[:, :2].sum(-1), student[:, 2], student[:, 3]), dim=-1)
    teacher_class = torch.stack((teacher[:, :2].sum(-1), teacher[:, 2], teacher[:, 3]), dim=-1)
    class_loss = (
        torch.sum(teacher_class * (torch.log(teacher_class) - torch.log(student_class))) * 4.0
    )
    class_loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
