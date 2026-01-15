from typing import cast

from transformers import (  # type: ignore
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_huggingface_tokenizer(hf_model_id: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)  # type: ignore

    special_tokens_dict: dict[str, str] = dict()
    if tokenizer.eos_token is None:  # type: ignore
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:  # type: ignore
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:  # type: ignore
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def get_huggingface_model_and_tokenizer(
    hf_model_id: str, device_map: str = "auto"
) -> tuple[PreTrainedModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    config = AutoConfig.from_pretrained(hf_model_id)  # type: ignore
    model = AutoModelForCausalLM.from_pretrained(  # type: ignore
        hf_model_id,
        torch_dtype=config.torch_dtype,  # type: ignore
        device_map=device_map,
    )

    return cast(PreTrainedModel, model), get_huggingface_tokenizer(hf_model_id)
