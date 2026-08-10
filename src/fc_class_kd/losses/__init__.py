"""Loss package."""

from fc_class_kd.losses.kd_losses import (
    candidate_kd_loss,
    class_kd_loss,
    combined_sft_kd_loss,
    masked_mean,
)

__all__ = [
    "candidate_kd_loss",
    "class_kd_loss",
    "combined_sft_kd_loss",
    "masked_mean",
]
