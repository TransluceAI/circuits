from typing import TYPE_CHECKING, Callable, List, Tuple, TypedDict

import datasets  # type: ignore
from util.chat_input import ModelInput

# For type checking
if TYPE_CHECKING:
    from util.subject import Subject


class DatasetElement(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def construct_dataset(
    subject: "Subject",
    samples: List[Tuple[ModelInput, ModelInput]],
    max_len: int | None = None,
    shift_labels: bool = True,
    prompt_attn_mask: list[list[int]] | None = None,
    completion_attn_mask: list[list[int]] | None = None,
) -> datasets.Dataset:
    """
    :param samples: List of tuples containing a prompt and a completion
    :param shift_labels: Okay hear me out, this is stupid.
        By default, label tokens are obtained by shifting the input sequence left by 1. This is the "correct" behavior.
        But HF autoshifts labels during training automatically and does not clearly document this.
        If False, label tokens will match the input tokens exactly.
    """

    for _, ci in samples:
        assert not ci.use_chat_format, "Completion tokens should not be in chat format"

    # Get tokens
    prompt_tokens = [ci.tokenize(subject) for ci, _ in samples]
    prompt_attn_mask = (
        [[1] * len(x) for x in prompt_tokens] if prompt_attn_mask is None else prompt_attn_mask
    )
    completion_tokens = [ci.tokenize(subject, trim_bos=True) for _, ci in samples]
    completion_attn_mask = (
        [[1] * len(x) for x in completion_tokens]
        if completion_attn_mask is None
        else completion_attn_mask
    )

    # Set max_len if None
    if max_len is None:
        max_len = max(
            max(
                len(q) + len(a) - (1 if shift_labels else 0)
                for q, a in zip(prompt_tokens, completion_tokens)
            ),
            0,
        )

    def _pad_and_truncate(l: list[int], pad_int: int):
        assert pad_int is not None, f"Padding token must be an int, got {pad_int}"
        return [pad_int] * (max_len - len(l)) + l[:max_len]

    concat = {
        "input_ids": [
            _pad_and_truncate(q + a[: (-1 if shift_labels else None)], subject.pad_token_id)
            for q, a in zip(prompt_tokens, completion_tokens)
        ],
        "attention_mask": [
            _pad_and_truncate(q + a[: (-1 if shift_labels else None)], 0)
            for q, a in zip(prompt_attn_mask, completion_attn_mask)
        ],
        "labels": [
            _pad_and_truncate([-100] * (len(q) - (1 if shift_labels else 0)) + a, -100)
            for q, a in zip(prompt_tokens, completion_tokens)
        ],
    }

    return datasets.Dataset.from_dict(concat)  # type: ignore


def pretty_print_dataset_element(subject: "Subject", x: DatasetElement):
    d: Callable[[int], str] = lambda f: (repr(subject.decode(f)) if f != -100 else "").rjust(20)
    for i in range(min(len(x["input_ids"]), len(x["labels"]))):
        print(f"{str(i).zfill(3)} | {d(x['input_ids'][i])} -> {d(x['labels'][i])}")
