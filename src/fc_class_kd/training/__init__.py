"""Training package."""

from fc_class_kd.training.candidate_scoring import (
    score_candidate_sequences,
    sequence_scores_from_logits,
)
from fc_class_kd.training.formal_sft import (
    SFTBudget,
    build_epoch_groups,
    checkpoint_name,
    compute_sft_budget,
    select_best_checkpoint,
    select_s0_by_validation_ast,
    sha256_file,
    supervised_token_count,
)
from fc_class_kd.training.model_utils import (
    attach_lora,
    load_student_model,
    load_tokenizer,
    max_memory_mb,
)
from fc_class_kd.training.sft import (
    EncodedSFTExample,
    collate_sft_examples,
    encode_sft_records,
    generate_function_call,
)

__all__ = [
    "EncodedSFTExample",
    "SFTBudget",
    "attach_lora",
    "build_epoch_groups",
    "checkpoint_name",
    "collate_sft_examples",
    "compute_sft_budget",
    "encode_sft_records",
    "generate_function_call",
    "load_student_model",
    "load_tokenizer",
    "max_memory_mb",
    "score_candidate_sequences",
    "select_best_checkpoint",
    "select_s0_by_validation_ast",
    "sequence_scores_from_logits",
    "sha256_file",
    "supervised_token_count",
]
