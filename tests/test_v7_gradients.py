import math

import pytest
import torch

from fc_class_kd.v7.gradients import gradient_geometry


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
