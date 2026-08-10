"""Internal evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from fc_class_kd.types import FCRecord, VerifierClass
from fc_class_kd.verifier.classifier import classify_call


@dataclass
class EvalMetrics:
    total: int = 0
    parse_success: int = 0
    function_name_correct: int = 0
    argument_correct: int = 0
    ast_correct: int = 0
    wrong_function: int = 0
    wrong_arguments: int = 0
    invalid: int = 0

    def rates(self) -> Dict[str, float]:
        if self.total == 0:
            return {}
        return {
            "parse_success_rate": self.parse_success / self.total,
            "function_name_accuracy": self.function_name_correct / self.total,
            "argument_accuracy": self.argument_correct / self.total,
            "ast_accuracy": self.ast_correct / self.total,
            "wrong_function_rate": self.wrong_function / self.total,
            "wrong_argument_rate": self.wrong_arguments / self.total,
            "invalid_rate": self.invalid / self.total,
        }


def evaluate_predictions(
    records: List[FCRecord],
    predictions: List[str],
) -> EvalMetrics:
    metrics = EvalMetrics()
    for rec, pred in zip(records, predictions):
        metrics.total += 1
        result = classify_call(pred, rec.tools, rec.gold_call)
        if result.parse_ok:
            metrics.parse_success += 1
        if result.class_label == VerifierClass.CORRECT:
            metrics.ast_correct += 1
            metrics.function_name_correct += 1
            metrics.argument_correct += 1
        elif result.class_label == VerifierClass.WRONG_FUNCTION:
            metrics.wrong_function += 1
        elif result.class_label == VerifierClass.WRONG_ARGUMENTS:
            metrics.wrong_arguments += 1
            if result.canonical_call and result.canonical_call.name == rec.gold_call.name:
                metrics.function_name_correct += 1
        elif result.class_label == VerifierClass.INVALID:
            metrics.invalid += 1
    return metrics


def robustness_drop(ast_original: float, ast_equiv: float) -> float:
    return ast_original - ast_equiv
