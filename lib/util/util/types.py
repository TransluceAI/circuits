from typing import Any, Literal, NamedTuple, NotRequired, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

NDFloatArray = NDArray[np.floating[Any]]
NDIntArray = NDArray[np.integer[Any]]
NDBoolArray = NDArray[np.bool_]


class ToolCallView(TypedDict):
    title: str
    format: str
    content: str


class ToolCall(TypedDict):
    id: str
    function: str
    arguments: dict[str, str]
    view: NotRequired[ToolCallView | None]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_call_id: NotRequired[str | None]
    tool_calls: NotRequired[list[ToolCall] | None]


class GenerateOutput(NamedTuple):
    output_ids_BT: NDIntArray
    logits_BV: torch.Tensor
    tokenwise_log_probs: list[tuple[NDIntArray, NDFloatArray]]
    continuations: list[str]
    acts: NDFloatArray


class TopKResult(NamedTuple):
    indices: list[int]
    probs: list[float]


Primitive = str | int | float | bool | None
