import os
import shutil
from typing import Callable, List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor
from transformers import AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

SPECIAL_TOKENS = [
    ("<|begin_of_text|>", "<||begin_of_text||>"),
    ("<|start_header_id|>", "<||start_header_id||>"),
    ("<|end_header_id|>", "<||end_header_id||>"),
    ("<|eot_id|>", "<||eot_id||>"),
]
UPD_MAPPING = {
    128000: 128257,  # <|begin_of_text|>
    128006: 128258,  # <|start_header_id|>
    128007: 128259,  # <|end_header_id|>
    128009: 128260,  # <|eot_id|>
}


def is_llamadecoder_layer(module):
    return isinstance(module, LlamaDecoderLayer)


def param_init_fn(module: torch.nn.Module) -> None:
    assert hasattr(module, "reset_parameters") and isinstance(module.reset_parameters, Callable)
    module.reset_parameters()


def masked_mean(tensor: torch.Tensor, mask: torch.BoolTensor, dim: int) -> torch.Tensor:
    """Returns the mean of the non-masked elements over the given dim.

    tensor and mask should be the same shape."""
    return (tensor * mask.float()).sum(dim) / mask.float().sum(dim)


class Llama3TokenizerWrapper:
    def __init__(self, model_path: str, add_special_tokens: bool = False):
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        if add_special_tokens:
            tokenizer.add_tokens(
                [
                    "<||begin_of_text||>",
                    "<||start_header_id||>",
                    "<||end_header_id||>",
                    "<||eot_id||>",
                ]
            )

        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def update_input(self, seq: str) -> str:
        """Convert special tokens to their updated versions."""
        if not self.add_special_tokens:
            return seq
        for tok, upd_tok in SPECIAL_TOKENS:
            seq = seq.replace(tok, upd_tok)
        return seq

    def update_output(self, seq: str) -> str:
        """Convert updated special tokens to their original versions."""
        if not self.add_special_tokens:
            return seq
        for tok, upd_tok in SPECIAL_TOKENS:
            seq = seq.replace(upd_tok, tok)
        return seq

    def apply_chat_template(self, messages, *args, **kwargs):
        upd_messages = []
        for msg in messages:
            seq = msg["content"]
            upd_seq = self.update_input(seq)
            upd_messages.append({"role": msg["role"], "content": upd_seq})
        return self.tokenizer.apply_chat_template(upd_messages, *args, **kwargs)

    def __call__(self, seq: str | List[str], *args, **kwargs):
        if isinstance(seq, str):
            upd_seq = self.update_input(seq)
        else:
            upd_seq = [self.update_input(s) for s in seq]
        return self.tokenizer(upd_seq, *args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


def get_tokenizer(model_path: str, add_special_tokens: bool = False):
    return Llama3TokenizerWrapper(model_path, add_special_tokens)


class Llama3Model(torch.nn.Module):
    def __init__(
        self,
        model_path: str = "/home/ubuntu/llama3.1_8b_instruct_hf/",
        add_special_tokens: bool = False,
    ) -> None:
        super().__init__()

        # use_cache must be off training
        hf_config = LlamaConfig.from_pretrained(model_path, use_cache=False)

        self.model = LlamaForCausalLM.from_pretrained(model_path, config=hf_config)

        original_vocab_size = 128256

        # Resize the token embeddings because we had to fix the tokenizer padding
        additional_tokens = 1
        if add_special_tokens:
            additional_tokens += 4
        self.model.resize_token_embeddings(
            original_vocab_size + additional_tokens, mean_resizing=False
        )
        self.model.config.pad_token_id = original_vocab_size

        self.model.fsdp_wrap_fn = is_llamadecoder_layer

        self.model.activation_checkpointing_fn = is_llamadecoder_layer

        self.model.fsdp_param_init_fn = param_init_fn

    def forward(self, batch: dict):
        input_ids = batch["input_ids"]

        attention_mask = batch["attention_mask"].bool() if "attention_mask" in batch else None
        # Account for padding by updating the position_ids based on attention_mask
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        ).logits
        # We want to make sure this is in full precision
        return logits.float()

    def loss(self, outputs: Tensor, batch: dict):
        targets = batch["labels"].view(-1)
        loss_mask = batch["loss_mask"].view(-1)
        unreduced_loss = F.cross_entropy(
            outputs.float().view(-1, outputs.size(-1)),
            targets,
            reduction="none",
        )
        return (unreduced_loss * loss_mask).sum() / loss_mask.sum()


def convert_to_hf_weights(
    ckpt_path: str,
    base_model_path: str = "/home/ubuntu/Llama-3.1-8B-Instruct",
    add_special_tokens: bool = True,
):
    model = Llama3Model(model_path=base_model_path, add_special_tokens=add_special_tokens)
    device = torch.device("cuda:0")
    model.to(device).to(torch.bfloat16)

    checkpoint = torch.load(ckpt_path)["model"]
    model.load_state_dict(checkpoint, strict=True)

    updated_folder_name = os.path.basename(ckpt_path).replace(".pt", "_hf")
    upd_data_dir = os.path.join(os.path.dirname(ckpt_path), updated_folder_name)
    model.model.save_pretrained(upd_data_dir)

    for fname in ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        shutil.copy(os.path.join(base_model_path, fname), upd_data_dir)
