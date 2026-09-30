import importlib.util
import math
from pathlib import Path

import pytest
import torch

from fc_class_kd.v7.gradients import gradient_geometry
from fc_class_kd.v7.kd import fixed_support_kd_loss


def load_gradient_script():
    path = Path("scripts/v7_analyze_gradients.py")
    spec = importlib.util.spec_from_file_location("v7_analyze_gradients", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gradient_geometry_reports_exact_decomposition():
    inter = [torch.tensor([1.0, 2.0])]
    intra = [torch.tensor([2.0, 2.0])]
    candidate = [inter[0] + intra[0]]
    result = gradient_geometry(candidate, inter, intra)
    assert result["grad_candidate_norm"] == pytest.approx(5.0)
    assert result["grad_class_norm"] == pytest.approx(math.sqrt(5.0))
    assert result["grad_inter_norm"] == pytest.approx(math.sqrt(5.0))
    assert result["grad_intra_norm"] == pytest.approx(math.sqrt(8.0))
    assert result["cosine_inter_intra"] == pytest.approx(6.0 / math.sqrt(40.0))
    assert result["decomposition_residual_norm"] == pytest.approx(0.0)


def test_gradient_geometry_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        gradient_geometry([torch.ones(2)], [], [torch.ones(2)])


def test_gradient_geometry_uses_none_for_zero_norm_cosine():
    result = gradient_geometry([torch.zeros(1)], [torch.zeros(1)], [torch.zeros(1)])
    assert result["cosine_inter_intra"] is None


def test_score_weight_chain_rule_matches_direct_parameter_gradients():
    module = load_gradient_script()
    parameters = torch.tensor([0.2, -0.3], dtype=torch.float32, requires_grad=True)
    design = torch.tensor([[1.0, 0.5, -0.2, 0.7], [-0.4, 0.8, 1.2, -0.6]], dtype=torch.float32)
    scores = (parameters @ design).unsqueeze(0)
    teacher = torch.tensor([[0.45, 0.35, 0.15, 0.05]], dtype=torch.float32)
    candidate = fixed_support_kd_loss(scores, teacher, "candidate", temperature=2.0)
    between = fixed_support_kd_loss(scores, teacher, "class", temperature=2.0)
    losses = {"candidate": candidate, "class": between, "intra": candidate - between}
    direct = {
        key: torch.autograd.grad(value, parameters, retain_graph=True)[0]
        for key, value in losses.items()
    }

    values, score_weights = module.loss_values_and_score_weights(scores, teacher, 2.0)
    for key, loss in losses.items():
        chained = torch.autograd.grad(
            scores,
            parameters,
            grad_outputs=score_weights[key].unsqueeze(0),
            retain_graph=True,
        )[0]
        assert values[key] == pytest.approx(float(loss.detach()), abs=1e-7)
        assert torch.allclose(chained, direct[key], atol=1e-7, rtol=1e-6)
