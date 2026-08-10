"""Shared types for FC Class KD Fast Pilot."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerifierClass(str, Enum):
    CORRECT = "correct"
    WRONG_FUNCTION = "wrong_function"
    WRONG_ARGUMENTS = "wrong_arguments"
    INVALID = "invalid"
    UNKNOWN = "unknown"


# Fast Pilot primary classes used in Class KD (unknown excluded from aggregation)
PRIMARY_CLASSES: tuple[VerifierClass, ...] = (
    VerifierClass.CORRECT,
    VerifierClass.WRONG_FUNCTION,
    VerifierClass.WRONG_ARGUMENTS,
    VerifierClass.INVALID,
)


class ToolParameter(BaseModel):
    type: str
    description: Optional[str] = None
    required: bool = False
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    unordered: bool = False  # schema-declared unordered collection


class ToolSchema(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, ToolParameter] = Field(default_factory=dict)


class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class FCRecord(BaseModel):
    """Unified internal record after adapter + filter."""

    id: str
    source: str
    query: str
    tools: List[ToolSchema]
    gold_call: FunctionCall
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerifierResult(BaseModel):
    class_label: VerifierClass
    sub_label: Optional[str] = None
    canonical_call: Optional[FunctionCall] = None
    parse_ok: bool = False
    details: List[str] = Field(default_factory=list)
