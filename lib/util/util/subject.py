import gc
import threading
from collections import defaultdict
from queue import Queue
from typing import Any, Callable, Generator, Literal, Sequence, cast

import torch
from env_util import ENV
from nnsight import LanguageModel  # type: ignore
from nnsight.models.LanguageModel import LanguageModelProxy  # type: ignore
from nnsight.util import fetch_attr  # type: ignore
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer  # type: ignore
from transformers.generation.streamers import BaseStreamer  # type: ignore
from util.activations import ModelActivations
from util.chat_input import IdsInput, ModelInput
from util.dataset import construct_dataset
from util.types import GenerateOutput, NDFloatArray, NDIntArray, TopKResult


def _ct(x: Any) -> torch.Tensor:
    return cast(torch.Tensor, x)


class TokenIdStreamer(BaseStreamer):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        verbose: bool = False,
        timeout: float | None = None,
    ):
        self.tokenizer = tokenizer
        self.token_id_queue: Queue[int | None] = Queue()

        self.verbose, self.timeout = verbose, timeout
        self.stop_signal = None

    def put(self, value: torch.Tensor):
        """
        Receives token IDs and puts them in the queue.
        """

        assert isinstance(
            value, torch.Tensor
        ), "TokenIdStreamer only expects streaming torch.Tensors"

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TokenIdStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]  # Discard future batches

        tokens: list[int] = value.tolist()  # type: ignore
        for token_id in tokens:
            self.token_id_queue.put(token_id, timeout=self.timeout)
            if self.verbose:
                print(self.tokenizer.decode(token_id), end="", flush=True)  # type: ignore

    def end(self):
        """Signals the end of token ID generation by putting a stop signal in the queue."""
        self.token_id_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_id_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


def pad_matrix_with_last_row(matrix: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Pad a matrix of shape (S, *) to shape (B, *) by repeating the last row.

    Args:
        matrix (torch.Tensor): Input tensor of shape (S, *)
        target_size (int): Desired number of rows B, where B >= S

    Returns:
        torch.Tensor: Padded tensor of shape (B, *)
    """
    if target_size < matrix.size(0):
        raise ValueError(f"Target size {target_size} must be >= input size {matrix.size(0)}")

    # Calculate number of rows to pad
    num_pad_rows = target_size - matrix.size(0)

    if num_pad_rows == 0:
        return matrix

    # Get the last row and repeat it
    last_row = matrix[-1:, :]  # Keep dim using -1: instead of -1
    padding = last_row.repeat(num_pad_rows, 1)

    # Concatenate original matrix with padding
    return torch.cat([matrix, padding], dim=0)


def batch_interventions(
    neuron_interventions: list[dict[tuple[int, int, int], float]],
    num_pad_tokens: list[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Batch interventions for a list of inputs.

    For each layer, we collect all interventions for that layer, and then pad them to the
    same length across inputs.
    We also update the token indices in the interventions to account for the padding of inputs
    (different from the padding of interventions).

    Args:
        neuron_interventions: List of interventions for each input
        num_pad_tokens: Number of padding tokens for each input
    """
    assert len(neuron_interventions) == len(num_pad_tokens)
    B = len(neuron_interventions)

    interventions_by_layer: dict[int, list[list[tuple[int, int, float]]]] = {}
    for bi, interventions_for_input in enumerate(neuron_interventions):
        for (layer, token, neuron), value in interventions_for_input.items():
            if layer not in interventions_by_layer:
                interventions_by_layer[layer] = [[] for _ in range(B)]
            interventions_by_layer[layer][bi].append((token, neuron, value))

    intervention_tensors_by_layer: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    # Pad interventions when necessary.
    for layer, intervs in interventions_by_layer.items():
        max_len = 0
        bis: list[int] = []
        for bi, intervs_i in enumerate(intervs):
            if not intervs_i:
                continue
            bis.append(bi)
            max_len = max(max_len, len(intervs_i))

        batch_idxs = torch.tensor(bis, dtype=torch.long).unsqueeze(1).expand(-1, max_len)
        tokens: list[torch.Tensor] = []
        neurons: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for bi in bis:
            token_neuron_values = torch.tensor(intervs[bi])  # (num intervs, 3)
            padded_tnv = pad_matrix_with_last_row(token_neuron_values, max_len)  # (max_len, 3)
            # For the token indexing, need to offset by the padding applied to the input tokens.
            tokens.append(padded_tnv[:, 0].long() + num_pad_tokens[bi])
            neurons.append(padded_tnv[:, 1].long())
            values.append(padded_tnv[:, 2].float())

        batched_tokens = torch.stack(tokens, dim=0)  # (len(bis), max_len)
        batched_neurons = torch.stack(neurons, dim=0)  # (len(bis), max_len)
        batched_values = torch.stack(values, dim=0)  # (len(bis), max_len)

        intervention_tensors_by_layer[layer] = (
            batch_idxs,
            batched_tokens,
            batched_neurons,
            batched_values,
        )

    return intervention_tensors_by_layer


class LMConfig(BaseModel):
    """
    Configuration class for Language Models.

    This class defines the structure and properties of a language model,
    including its architecture, module paths, and dimensional information.

    The layernorm_fn should have the following signature:
        (tensor to normalize, tensor to compute statistics with, norm weight, norm_eps) -> normalized tensor
    """

    hf_model_id: str
    is_chat_model: bool
    has_default_sys_prompt: bool
    unembed_module_str: str
    w_in_module_template: str
    w_gate_module_template: str
    w_out_module_template: str
    layer_module_template: str
    mlp_module_template: str
    attn_module_template: str
    v_proj_module_template: str
    o_proj_module_template: str
    input_norm_module_template: str
    unembed_norm_module_str: str

    # Metadata about model dims
    I_name: str  # Intermediate size (# of neurons)
    D_name: str  # Residual stream size
    V_name: str  # Vocab size
    L_name: str  # Num layers
    Q_name: str  # Num query attention heads
    K_name: str  # Num key/value attention heads (might not equal Q in grouped K/V attention)

    # Layernorm impl with signature
    #   (tensor to normalize, tensor to compute statistics with, norm weight, norm_eps)
    #       -> normalized tensor
    layernorm_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]


class Subject:
    """
    This class encapsulates a language model along with its configuration,
    tokenizer, and various components. It provides easy access to model
    metadata, layers, and specific modules like embeddings, attention,
    and MLPs.
    """

    def __init__(
        self,
        config: LMConfig,
        cast_to_hf_config_dtype: bool = True,
        nnsight_lm_kwargs: dict[str, Any] = {},
        preloaded_model: torch.nn.Module | None = None,
        preloaded_tokenizer: AutoTokenizer | None = None,
    ):
        hf_config = AutoConfig.from_pretrained(config.hf_model_id, token=ENV.HF_TOKEN)  # type: ignore

        # Load model + tokenizer
        kwargs = {"dispatch": False, "device_map": "auto", "token": ENV.HF_TOKEN}
        kwargs.update({"torch_dtype": hf_config.torch_dtype} if cast_to_hf_config_dtype else {})  # type: ignore
        kwargs.update(nnsight_lm_kwargs)
        # kwargs.update({"attn_implementation": "eager"})
        if preloaded_model is None:
            self.model = LanguageModel(config.hf_model_id, **kwargs)
        else:
            self.model = LanguageModel(preloaded_model, tokenizer=preloaded_tokenizer, **kwargs)
        self.tokenizer = self.model.tokenizer
        self.hf_config, self.lm_config = hf_config, config  # type: ignore
        self.is_chat_model = config.is_chat_model
        self.has_default_sys_prompt = config.has_default_sys_prompt
        self.model_name = config.hf_model_id.split("/")[-1].lower().replace("-", "_")

        # Padding always on the left
        self.tokenizer.padding_side = "left"

        # Metadata about the model
        self.I: int = int(hf_config.__dict__[config.I_name])  # type: ignore
        self.D: int = int(hf_config.__dict__[config.D_name])  # type: ignore
        self.V: int = int(hf_config.__dict__[config.V_name])  # type: ignore
        self.L: int = int(hf_config.__dict__[config.L_name])  # type: ignore
        self.Q: int = int(hf_config.__dict__[config.Q_name])  # type: ignore
        self.K: int = int(hf_config.__dict__[config.K_name])  # type: ignore
        self.H: int = int(hf_config.__dict__[config.H_name])  # type: ignore

        # Model components
        self.unembed = fetch_attr(self.model, config.unembed_module_str)
        self.unembed_norm = fetch_attr(self.model, config.unembed_norm_module_str)
        self.w_ins = {
            layer: fetch_attr(self.model, config.w_in_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        self.w_gates = {
            layer: fetch_attr(self.model, config.w_gate_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        self.w_outs = {
            layer: fetch_attr(self.model, config.w_out_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        self.layers = {
            layer: fetch_attr(self.model, config.layer_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        self.mlps = {
            layer: fetch_attr(self.model, config.mlp_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        self.attns = {
            layer: fetch_attr(self.model, config.attn_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        self.attn_vs = {
            layer: fetch_attr(self.model, config.v_proj_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        self.attn_os = {
            layer: fetch_attr(self.model, config.o_proj_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        self.input_norms = {
            layer: fetch_attr(self.model, config.input_norm_module_template.format(layer=layer))
            for layer in range(self.L)
        }
        if "gemma" in self.model_name:
            self.pre_mlp_norms = {
                layer: fetch_attr(
                    self.model, config.pre_mlp_norm_module_template.format(layer=layer)
                )
                for layer in range(self.L)
            }
            self.post_mlp_norms = {
                layer: fetch_attr(
                    self.model, config.post_mlp_norm_module_template.format(layer=layer)
                )
                for layer in range(self.L)
            }
            self.post_attention_norms = {
                layer: fetch_attr(
                    self.model, config.post_attention_norm_module_template.format(layer=layer)
                )
                for layer in range(self.L)
            }

        # Layernorm implementation
        self.layernorm_fn = config.layernorm_fn

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype  # type: ignore

    ################
    # Tokenization #
    ################

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text)["input_ids"]  # type: ignore

    def tokenize_single(self, text: str) -> int:
        """
        Tokenize a single token; raises an error if the text is not a single token.
        """

        toks = self.tokenize(text)
        assert (
            len(toks) == 1
        ), f"Expected 1 token, got {len(toks)}: {[self.decode(t) for t in toks]}"
        return toks[0]

    def decode(self, token_ids: int | list[int] | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)  # type: ignore

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id  # type: ignore

    def apply_logit_lens(
        self,
        dirs_X_D: torch.Tensor,
        normalize_logits: bool = True,
        select_tokens_X: list[int] | None = None,
    ):
        """
        Args:
            normalize_logits: If true, we subtract the mean logit across all tokens
            select_tokens_X: If not None, we only select these tokens from the unembedding matrix
        """

        # Collect only the projection matrix rows we need
        unembed_orig_VD = self.unembed.weight
        if select_tokens_X is None:
            unembed_VD = unembed_orig_VD
        else:
            unembed_VD = unembed_orig_VD[select_tokens_X]

        # Move to the same device as the (large) unembed matrix
        dirs_X_D = dirs_X_D.to(unembed_orig_VD.device)

        with torch.no_grad():
            logits_X_V = torch.einsum("vd,...d->...v", unembed_VD, dirs_X_D)
            if normalize_logits:
                logits_X_V -= torch.einsum(
                    "d,...d->...", unembed_orig_VD.mean(dim=0), dirs_X_D
                ).unsqueeze(-1)

        return logits_X_V

    def collect_acts(
        self,
        cis: list[ModelInput],
        layers: list[int],
        include: list[str] | Literal["*"],
        attention_masks: list[list[int]] | None = None,
    ) -> ModelActivations:
        """
        Collect activations for the given layers, only including the given keys.
        """

        # Get tokenized and padded dataset for batch input to the model
        # Apply attention masks to the input ids if provided
        ds = construct_dataset(
            self,
            [(ci, IdsInput(input_ids=[])) for ci in cis],
            shift_labels=False,
            prompt_attn_mask=attention_masks,
        )
        input_ids, attn_mask = (
            torch.tensor([x["input_ids"] for x in ds.to_list()]),  # type: ignore
            torch.tensor([x["attention_mask"] for x in ds.to_list()]),  # type: ignore
        )
        if input_ids.shape[1] == 0:
            raise ValueError("No input tokens provided")

        # If include is "*", we collect all activations
        if include == "*":
            include = [
                "resid_BTD",
                "mlp_in_BTD",
                "mlp_gate_BTD",
                "mlp_out_BTD",
                "attn_heads_BTD",
                "attn_out_BTD",
                "attn_map_BQTT",
                "neurons_BTI",
                "unembed_in_BTD",
                "unembed_out_BTV",
            ]
        else:
            assert isinstance(include, list), "Argument `include` must be a list of strings or '*'"

        acts: dict[str, Any] = {"layers": {}}
        with self.model.trace(  # type: ignore
            {"input_ids": input_ids, "attention_mask": attn_mask},  # type: ignore
            output_attentions=True,  # type: ignore
        ):
            for layer in layers:
                layer_acts: dict[str, LanguageModelProxy] = {}
                if "resid_BTD" in include:
                    layer_acts["resid_BTD"] = self.layers[layer].output[0].detach().save()
                if "mlp_in_BTD" in include:
                    layer_acts["mlp_in_BTD"] = self.mlps[layer].input.detach().save()
                if "mlp_gate_BTD" in include:
                    layer_acts["mlp_gate_BTD"] = self.w_gates[layer].output.detach().save()
                if "mlp_out_BTD" in include:
                    layer_acts["mlp_out_BTD"] = self.mlps[layer].output.detach().save()
                if "attn_heads_BTD" in include:
                    layer_acts["attn_heads_BTD"] = self.attn_os[layer].input.detach().save()
                if "attn_out_BTD" in include:
                    layer_acts["attn_out_BTD"] = self.attns[layer].output[0].detach().save()
                if "attn_map_BQTT" in include:
                    layer_acts["attn_map_BQTT"] = self.attns[layer].output[1].detach().save()
                if "neurons_BTI" in include:
                    layer_acts["neurons_BTI"] = self.w_outs[layer].input.detach().save()
                if "pre_mlp_norms_in_BTD" in include:
                    layer_acts["pre_mlp_norms_in_BTD"] = (
                        self.pre_mlp_norms[layer].input.detach().save()
                    )
                if "post_mlp_norms_in_BTD" in include:
                    layer_acts["post_mlp_norms_in_BTD"] = (
                        self.post_mlp_norms[layer].input.detach().save()
                    )
                if "post_attention_norms_in_BTD" in include:
                    layer_acts["post_attention_norms_in_BTD"] = (
                        self.post_attention_norms[layer].input.detach().save()
                    )
                acts["layers"][layer] = layer_acts

            if "unembed_in_BTD" in include:
                acts["unembed_in_BTD"] = self.unembed.input.detach().save()
            if "unembed_out_BTV" in include:
                acts["unembed_out_BTV"] = self.unembed.output.detach().save()

        # Convert proxies to tensors
        for key in include:
            for layer in layers:
                if key in acts["layers"][layer]:
                    acts["layers"][layer][key] = acts["layers"][layer][key].value
            if key in acts:
                acts[key] = acts[key].value

        return ModelActivations(**acts)

    def generate(
        self,
        ci: ModelInput,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        neuron_interventions: dict[tuple[int, int, int], float] | None = None,
        num_return_sequences: int = 1,
        n_top_logprobs: int = 5,
        stream: bool = False,
        verbose: bool = False,
    ) -> GenerateOutput | Generator[int | None | GenerateOutput, None, None]:
        """
        Generate text using the model with optional neuron and hidden state interventions.
        """

        if stream and num_return_sequences > 1:
            raise ValueError("Cannot stream multiple return sequences; TODO implement this")

        inputs = ci.tokenize(self)
        streamer = TokenIdStreamer(cast(AutoTokenizer, self.tokenizer), verbose=verbose)

        # First group interventions by layer
        intervention_tensors_by_layer: (
            dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None
        if neuron_interventions is not None:
            interventions_by_layer: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
            for (layer, token, neuron), value in neuron_interventions.items():
                interventions_by_layer[layer].append((token, neuron, value))

            intervention_tensors_by_layer = {}
            for layer, intervs in interventions_by_layer.items():
                tokens, neurons, values = zip(*intervs)
                assert all(t is not None for t in tokens), "Cannot intervene on None tokens"
                intervention_tensors_by_layer[layer] = (
                    torch.tensor(tokens),
                    torch.tensor(neurons),
                    torch.tensor(values, dtype=self.dtype),
                )

        output_ids_BT: torch.Tensor | None = None
        log_probs_BV: torch.Tensor | None = None
        tokenwise_log_probs: list[tuple[NDIntArray, NDFloatArray]] | None = None
        all_acts: torch.Tensor | None = None

        def _generate_impl():
            nonlocal output_ids_BT, log_probs_BV, tokenwise_log_probs, all_acts

            acts_by_layer: list[list[torch.Tensor]] = []
            with torch.no_grad():
                with self.model.generate(  # type: ignore
                    inputs,  # type: ignore
                    max_new_tokens=max_new_tokens,  # type: ignore
                    temperature=temperature,  # type: ignore
                    num_return_sequences=num_return_sequences,  # type: ignore
                    do_sample=(False if temperature == 0.0 else True),  # type: ignore
                    streamer=streamer,  # type: ignore
                    scan=False,  # type: ignore
                    validate=False,  # type: ignore
                ):
                    # Save next-token logits and all output IDs
                    log_probs_BV = _ct(torch.log_softmax(self.model.lm_head.output[:, -1].float(), dim=-1).detach().cpu().save())  # type: ignore
                    output_ids_BT = _ct(self.model.generator.output.detach().cpu().save())  # type: ignore

                    # Collect top logprobs at each token
                    local_tokenwise_log_probs: list[tuple[torch.Tensor, torch.Tensor]] = []
                    iter_lm_head = self.model.lm_head
                    for i in range(max_new_tokens):
                        cur_log_probs_BV = torch.log_softmax(
                            _ct(iter_lm_head.output)[:, -1].float(), dim=-1
                        )
                        if i == 0:
                            log_probs_BV = cur_log_probs_BV.detach().cpu().save()  # type: ignore

                        topk_result = torch.topk(cur_log_probs_BV, k=n_top_logprobs, dim=-1)
                        top_logprobs_BX = topk_result.values
                        top_indices_BX = topk_result.indices

                        local_tokenwise_log_probs.append(
                            (
                                top_indices_BX.detach().cpu().save(),  # type: ignore
                                top_logprobs_BX.detach().cpu().save(),  # type: ignore
                            )
                        )

                        if i < max_new_tokens - 1:
                            iter_lm_head = iter_lm_head.next()

                    # Intervene as necessary. Also save activations.
                    for layer in range(self.L):
                        module = self.model.model.layers[layer].mlp.down_proj
                        device = _ct(module.input).device

                        if (
                            intervention_tensors_by_layer is not None
                            and layer in intervention_tensors_by_layer
                        ):
                            tokens, neurons, values = intervention_tensors_by_layer[layer]

                            # Move to the correct device.
                            tokens = tokens.to(device)
                            neurons = neurons.to(device)
                            values = values.to(device)

                            # Intervene on things that are within the prompt.
                            mask = tokens < len(inputs)
                            module.input[:, tokens[mask], neurons[mask]] = values[mask]

                            # Save activations on input tokens.
                            layer_acts = [_ct(module.input.detach().cpu().save())]

                            # Maybe intervene on things that are outside the prompt
                            for i_next in range(max_new_tokens - 1):
                                # Advance the nnsight object to the next token
                                module = module.next()

                                # Intervene on the current token
                                cur_token = len(inputs) + i_next
                                mask = tokens == cur_token
                                module.input[:, -1, neurons[mask]] = values[mask]

                                # Save activations on output tokens.
                                layer_acts.append(_ct(module.input.detach().cpu().save()))
                        else:
                            layer_acts = [_ct(module.input.detach().cpu().save())]

                            # Maybe intervene on things that are outside the prompt
                            for i_next in range(max_new_tokens - 1):
                                # Advance the nnsight object to the next token
                                module = module.next()
                                layer_acts.append(_ct(module.input.detach().cpu().save()))

                        acts_by_layer.append(layer_acts)

            tokenwise_log_probs = [
                (tokens.numpy(), log_probs.float().numpy())  # type: ignore
                for i, (tokens, log_probs) in enumerate(local_tokenwise_log_probs)
                if i < output_ids_BT.shape[1] - len(inputs)
            ]

            # Filter out activations that haven't been "generated".
            # This is because of the loop `for i_next in range(max_new_tokens - 1):`
            # which doesn't take into account the generating ending early.
            filtered_layer_acts = [
                torch.cat(
                    [acts_by_layer[i][j] for j in range(output_ids_BT.shape[1] - len(inputs))],
                    dim=1,
                )  # (num_return_sequences, num tokens, num neurons)
                for i in range(self.L)
            ]
            all_acts = torch.stack(
                filtered_layer_acts, dim=1
            ).float()  # (num_return_sequences, num layers, num tokens, num neurons)

        def _get_output():
            assert output_ids_BT is not None, "output_ids not collected"
            assert log_probs_BV is not None, "log_probs not collected"
            assert tokenwise_log_probs is not None, "tokenwise_log_probs not collected"
            assert all_acts is not None, "all_acts not collected"

            out_strs = [
                self.decode(output_ids_BT[i, len(inputs) :]) for i in range(num_return_sequences)
            ]

            gc.collect()
            torch.cuda.empty_cache()

            return GenerateOutput(
                output_ids_BT.numpy(),  # type: ignore
                log_probs_BV,
                tokenwise_log_probs,
                out_strs,
                all_acts.numpy(),  # type: ignore
            )

        if stream:

            def _generator() -> Generator[int | None | GenerateOutput, None, None]:
                thread = threading.Thread(target=_generate_impl)
                thread.start()
                for token_id in streamer:
                    yield token_id
                thread.join()

                yield _get_output()

            return _generator()
        else:
            _generate_impl()
            return _get_output()

    def do_batched_interventions(
        self,
        cis: Sequence[ModelInput],
        neuron_interventions: list[dict[tuple[int, int, int], float]] | None,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        n_top_logprobs: int = 5,
    ) -> list[GenerateOutput]:
        """
        Generates text for each input in `cis` in a batched manner.
        Also supports applying different neuron interventions for each input in `cis`.

        Note: currently, there is no support for sampling multiple generations per input,
        but this should be pretty straightforward to add.

        Args:
            cis: list of `ModelInput` objects to generate text for.
            neuron_interventions: list of neuron interventions to apply to each input in `cis`.
            max_new_tokens: maximum number of tokens to generate.
            temperature: temperature to use for generation.
            n_top_logprobs: number of top logprobs to collect at each token.

        Returns:
            list[GenerateOutput]: list of generated outputs.
        """
        B = len(cis)

        # Tokenize and pad inputs (left-padded).
        input_ids_list, lengths = [], []
        for ci in cis:
            input_ids = torch.tensor(ci.tokenize(self))
            input_ids_list.append(input_ids)
            lengths.append(len(input_ids))
        lengths = torch.tensor(lengths)
        max_input_len = lengths.max().item()
        num_pad_tokens = max_input_len - lengths

        intervention_tensors_by_layer = (
            batch_interventions(neuron_interventions, num_pad_tokens.tolist())
            if neuron_interventions is not None
            else None
        )

        output_ids_BT: torch.Tensor | None = None
        log_probs_BV: torch.Tensor | None = None
        tokenwise_log_probs: list[tuple[NDIntArray, NDFloatArray]] | None = None
        all_acts: NDFloatArray | None = None

        def _generate_impl():
            nonlocal output_ids_BT, log_probs_BV, tokenwise_log_probs, all_acts

            acts_by_layer: list[list[torch.Tensor]] = []
            with torch.no_grad():
                with self.model.generate(  # type: ignore
                    input_ids_list,  # type: ignore
                    max_new_tokens=max_new_tokens,  # type: ignore
                    temperature=temperature,  # type: ignore
                    do_sample=(False if temperature == 0.0 else True),
                    num_return_sequences=1,  # type: ignore
                    scan=False,  # type: ignore
                    validate=False,  # type: ignore
                ):
                    # Save next-token logits and all output IDs
                    log_probs_BV = _ct(torch.log_softmax(self.model.lm_head.output[:, -1].float(), dim=-1).detach().cpu().save())  # type: ignore
                    output_ids_BT = _ct(self.model.generator.output.detach().cpu().save())  # type: ignore

                    # Collect top logprobs at each token
                    local_tokenwise_log_probs: list[tuple[torch.Tensor, torch.Tensor]] = []
                    iter_lm_head = self.model.lm_head
                    for i in range(max_new_tokens):
                        cur_log_probs_BV = torch.log_softmax(
                            _ct(iter_lm_head.output)[:, -1].float(), dim=-1
                        )
                        if i == 0:
                            log_probs_BV = cur_log_probs_BV.detach().cpu().save()  # type: ignore

                        topk_result = torch.topk(cur_log_probs_BV, k=n_top_logprobs, dim=-1)
                        top_logprobs_BX = topk_result.values
                        top_indices_BX = topk_result.indices

                        local_tokenwise_log_probs.append(
                            (
                                top_indices_BX.detach().cpu().save(),  # type: ignore
                                top_logprobs_BX.detach().cpu().save(),  # type: ignore
                            )
                        )

                        if i < max_new_tokens - 1:
                            iter_lm_head = iter_lm_head.next()

                    # Intervene as necessary. Also save activations.
                    for layer in range(self.L):
                        module = self.model.model.layers[layer].mlp.down_proj
                        device = _ct(module.input).device

                        if (
                            intervention_tensors_by_layer is not None
                            and layer in intervention_tensors_by_layer
                        ):
                            batch_idxs, tokens, neurons, values = intervention_tensors_by_layer[
                                layer
                            ]

                            # Move to the correct device.
                            batch_idxs = batch_idxs.to(device)
                            tokens = tokens.to(device)
                            neurons = neurons.to(device)
                            values = values.to(device).to(dtype=self.dtype)

                            # Intervene on things that are within the prompt.
                            mask = tokens < max_input_len
                            module.input[batch_idxs[mask], tokens[mask], neurons[mask]] = values[
                                mask
                            ]

                            # Save activations on input tokens.
                            layer_acts = [_ct(module.input.detach().cpu().save())]

                            # Maybe intervene on things that are outside the prompt
                            for i_next in range(max_new_tokens - 1):
                                # Advance the nnsight object to the next token
                                module = module.next()

                                # Intervene on the current token
                                cur_token = max_input_len + i_next  #############
                                mask = tokens == cur_token
                                module.input[batch_idxs[mask], -1, neurons[mask]] = values[mask]

                                # Save activations on output tokens.
                                layer_acts.append(_ct(module.input.detach().cpu().save()))
                        else:
                            layer_acts = [_ct(module.input.detach().cpu().save())]

                            # Maybe intervene on things that are outside the prompt
                            for i_next in range(max_new_tokens - 1):
                                # Advance the nnsight object to the next token
                                module = module.next()
                                layer_acts.append(_ct(module.input.detach().cpu().save()))

                        acts_by_layer.append(layer_acts)

            tokenwise_log_probs = [
                (tokens.numpy(), log_probs.float().numpy())  # type: ignore
                for i, (tokens, log_probs) in enumerate(local_tokenwise_log_probs)
                if i < output_ids_BT.shape[1] - max_input_len
            ]

            # Filter out activations that haven't been "generated".
            # This is because of the loop `for i_next in range(max_new_tokens - 1):`
            # which doesn't take into account the generating ending early.
            filtered_layer_acts = [
                torch.cat(
                    [acts_by_layer[i][j] for j in range(output_ids_BT.shape[1] - max_input_len)],
                    dim=1,
                )  # (B, num tokens, num neurons)
                for i in range(self.L)
            ]
            all_acts = (
                torch.stack(filtered_layer_acts, dim=1).float().numpy()
            )  # (B, num layers, num tokens, num neurons)

        def _get_output():
            assert output_ids_BT is not None, "output_ids not collected"
            assert log_probs_BV is not None, "log_probs not collected"
            assert tokenwise_log_probs is not None, "tokenwise_log_probs not collected"
            assert all_acts is not None, "all_acts not collected"

            gc.collect()
            torch.cuda.empty_cache()

            gen_outputs = []
            for bi in range(B):
                offset = num_pad_tokens[bi]
                out_strs = [self.decode(output_ids_BT[bi, offset + len(input_ids_list[bi]) :])]

                gen_outputs.append(
                    GenerateOutput(
                        output_ids_BT[bi].numpy()[offset:],
                        log_probs_BV[bi],
                        [(t[bi], lp[bi]) for t, lp in tokenwise_log_probs],
                        out_strs,
                        all_acts[bi, :, offset:],
                    )
                )
            return gen_outputs

        _generate_impl()
        return _get_output()

    def softmax_top_k(
        self,
        logits_V: torch.Tensor,
        k: int = 5,
        verbose: bool = True,
        bottom_k: bool = False,
    ):
        """
        Get the top k tokens from the softmax of the logits.
        """

        top_indices = torch.argsort(logits_V, descending=not bottom_k)[:k]
        probs_V = logits_V.softmax(dim=0)
        top_logits, top_probs = logits_V[top_indices], probs_V[top_indices]

        if verbose:
            for index, prob, logit in zip(top_indices, top_probs, top_logits):
                token = repr(self.decode(index))
                print(f"Index: {index}, Token: {token}, Prob: {prob.item()}, Logit: {logit.item()}")

        return TopKResult(
            top_indices.detach().cpu().tolist(),  # type: ignore
            top_probs.detach().cpu().float().tolist(),  # type: ignore
        )

    # def get_final_token_logits(
    #     self,
    #     ci: ChatInput,
    #     target_token: int,
    #     distractor_token: int,
    #     final_token_idx: int = -1,
    #     plot: bool = True,
    # ):
    #     """
    #     Get the final token logit difference across layers for a target and distractor token.
    #     """

    #     acts = self.collect_acts(
    #         [ci],
    #         layers=list(range(self.L)),
    #         include=["resid_BTD", "attn_out_BTD", "mlp_out_BTD"],
    #     )

    #     # Collect residuals, attention outputs, and MLP outputs for each layer
    #     resid_LT, attn_out_LT, mlp_out_LT = [], [], []
    #     for layer in range(self.L):
    #         # Careful to apply layernorms
    #         resid_TD = self.layernorm_fn(
    #             acts[layer].resid_BTD[0, :, None],
    #             acts[self.L - 1].resid_BTD[0, final_token_idx][None],
    #             self.unembed_norm.weight,
    #             self.unembed_norm.variance_epsilon,
    #         )[:, 0]
    #         attn_out_TD = self.layernorm_fn(
    #             acts[layer].attn_out_BTD[0, :, None],
    #             acts[self.L - 1].resid_BTD[0, final_token_idx][None],
    #             self.unembed_norm.weight,
    #             self.unembed_norm.variance_epsilon,
    #         )[:, 0]
    #         mlp_out_TD = self.layernorm_fn(
    #             acts[layer].mlp_out_BTD[0, :, None],
    #             acts[self.L - 1].resid_BTD[0, final_token_idx][None],
    #             self.unembed_norm.weight,
    #             self.unembed_norm.variance_epsilon,
    #         )[:, 0]

    #         # Get target and distractor tokens
    #         resid_TV = self.apply_logit_lens(resid_TD)
    #         attn_out_TV = self.apply_logit_lens(attn_out_TD)
    #         mlp_out_TV = self.apply_logit_lens(mlp_out_TD)

    #         # Compute logit diff
    #         resid_T = resid_TV[:, target_token] - resid_TV[:, distractor_token]
    #         attn_out_T = attn_out_TV[:, target_token] - attn_out_TV[:, distractor_token]
    #         mlp_out_T = mlp_out_TV[:, target_token] - mlp_out_TV[:, distractor_token]

    #         # Save
    #         resid_LT.append(resid_T.cpu().float().numpy())
    #         attn_out_LT.append(attn_out_T.cpu().float().numpy())
    #         mlp_out_LT.append(mlp_out_T.cpu().float().numpy())

    #     resid_LT = np.stack(resid_LT, axis=0)
    #     attn_out_LT = np.stack(attn_out_LT, axis=0)
    #     mlp_out_LT = np.stack(mlp_out_LT, axis=0)

    #     # Focus on the last token (perhaps can swap this out)
    #     resid_L = resid_LT[:, final_token_idx]
    #     attn_out_L = attn_out_LT[:, final_token_idx]
    #     mlp_out_L = mlp_out_LT[:, final_token_idx]

    #     if plot:
    #         # Plot
    #         plt.figure(figsize=(12, 6))
    #         plt.plot(range(self.L), resid_L, label="Residual", color="blue")
    #         plt.plot(
    #             range(self.L),
    #             np.cumsum(attn_out_L),
    #             label="Attention (Cumulative)",
    #             color="red",
    #         )
    #         plt.plot(
    #             range(self.L),
    #             np.cumsum(mlp_out_L),
    #             label="MLP (Cumulative)",
    #             color="green",
    #         )
    #         plt.plot(
    #             range(self.L),
    #             np.cumsum(attn_out_L) + np.cumsum(mlp_out_L),
    #             label="Attention + MLP (Cumulative)",
    #             color="purple",
    #         )
    #         plt.plot(
    #             range(self.L),
    #             mlp_out_L,
    #             label="MLP (Individual)",
    #             color="lightgreen",
    #             linestyle=":",
    #             linewidth=2,
    #         )
    #         plt.xlabel("Layer")
    #         plt.ylabel("Logit Difference")
    #         plt.title(
    #             f"Final Token Logit Difference Across Layers: {self.tokenizer.decode(target_token)} - {self.tokenizer.decode(distractor_token)}"
    #         )
    #         plt.legend()
    #         plt.grid(True, linestyle="--", alpha=0.7)
    #         plt.tight_layout()
    #         plt.show()

    #     # Return logit differences between target and distractor tokens
    #     return resid_L, attn_out_L, mlp_out_L


def _llama3_layernorm_fn(
    x_X1X2D: torch.Tensor,
    estimator_X1D: torch.Tensor,
    norm_w_D: torch.Tensor,
    eps: float,
):
    """
    Normalizes x along the X1/X2 dimensions by computing RMS statistics across the D dimension of estimator_X1D,
    then applying the same normalization to constant to X2D for all X1.
    """

    # Put everything on the device that input is on
    device = x_X1X2D.device

    # Compute
    return (
        norm_w_D[None, None, :].to(device)
        * x_X1X2D
        * torch.rsqrt(estimator_X1D.to(device).pow(2).mean(dim=1) + eps)[:, None, None]
    )


def _gemma2_layernorm_fn(
    x_X1X2D: torch.Tensor,
    estimator_X1D: torch.Tensor,
    norm_w_D: torch.Tensor,
    eps: float,
):
    """
    Normalizes x along the X1/X2 dimensions by computing RMS statistics across the D dimension of estimator_X1D,
    then applying the same normalization to constant to X2D for all X1.
    """

    # Put everything on the device that input is on
    device = x_X1X2D.device

    # Compute
    return (
        (1.0 + norm_w_D[None, None, :].to(device).float())
        * x_X1X2D.float()
        * torch.rsqrt(estimator_X1D.float().to(device).pow(2).mean(dim=1) + eps)[:, None, None]
    ).type_as(x_X1X2D)


###########
# Llama 3 #
###########


class Llama3Config(LMConfig):
    unembed_module_str: str = "lm_head"
    unembed_norm_module_str: str = "model.norm"
    w_in_module_template: str = "model.layers.{layer}.mlp.up_proj"
    w_gate_module_template: str = "model.layers.{layer}.mlp.gate_proj"
    w_out_module_template: str = "model.layers.{layer}.mlp.down_proj"
    layer_module_template: str = "model.layers.{layer}"
    mlp_module_template: str = "model.layers.{layer}.mlp"
    attn_module_template: str = "model.layers.{layer}.self_attn"
    v_proj_module_template: str = "model.layers.{layer}.self_attn.v_proj"
    o_proj_module_template: str = "model.layers.{layer}.self_attn.o_proj"
    input_norm_module_template: str = "model.layers.{layer}.input_layernorm"

    I_name: str = "intermediate_size"
    D_name: str = "hidden_size"
    V_name: str = "vocab_size"
    L_name: str = "num_hidden_layers"
    Q_name: str = "num_attention_heads"
    K_name: str = "num_key_value_heads"
    H_name: str = "hidden_size"  # TODO: this is redundant for llama3

    layernorm_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] = (
        _llama3_layernorm_fn
    )


llama3_8B_config = Llama3Config(
    hf_model_id="meta-llama/Meta-Llama-3-8B",
    is_chat_model=False,
    has_default_sys_prompt=False,
)

llama3_8B_instruct_config = Llama3Config(
    hf_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    is_chat_model=True,
    has_default_sys_prompt=False,
)

llama31_8B_config = Llama3Config(
    hf_model_id="meta-llama/Llama-3.1-8B",
    is_chat_model=False,
    has_default_sys_prompt=False,
)

llama31_8B_instruct_config = Llama3Config(
    hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
    is_chat_model=True,
    has_default_sys_prompt=True,
)

llama31_70B_instruct_config = Llama3Config(
    hf_model_id="meta-llama/Llama-3.1-70B-Instruct",
    is_chat_model=True,
    has_default_sys_prompt=True,
)

llama32_1B_instruct_config = Llama3Config(
    hf_model_id="meta-llama/Llama-3.2-1B-Instruct",
    is_chat_model=True,
    has_default_sys_prompt=True,
)

llama31_tiny_random_instruct_config = Llama3Config(
    hf_model_id="yujiepan/llama-3.1-tiny-random",
    is_chat_model=True,
    has_default_sys_prompt=True,
)

###########
# Gemma 2 #
###########


class Gemma2Config(LMConfig):
    unembed_module_str: str = "lm_head"
    unembed_norm_module_str: str = "model.norm"
    w_in_module_template: str = "model.layers.{layer}.mlp.up_proj"
    w_gate_module_template: str = "model.layers.{layer}.mlp.gate_proj"
    w_out_module_template: str = "model.layers.{layer}.mlp.down_proj"
    layer_module_template: str = "model.layers.{layer}"
    mlp_module_template: str = "model.layers.{layer}.mlp"
    attn_module_template: str = "model.layers.{layer}.self_attn"
    v_proj_module_template: str = "model.layers.{layer}.self_attn.v_proj"
    o_proj_module_template: str = "model.layers.{layer}.self_attn.o_proj"
    input_norm_module_template: str = "model.layers.{layer}.input_layernorm"
    # for gemma2 models, we need to collect input activations to get LNs
    pre_mlp_norm_module_template: str = "model.layers.{layer}.pre_feedforward_layernorm"
    post_mlp_norm_module_template: str = "model.layers.{layer}.post_feedforward_layernorm"
    post_attention_norm_module_template: str = "model.layers.{layer}.post_attention_layernorm"

    I_name: str = "intermediate_size"
    D_name: str = "hidden_size"
    V_name: str = "vocab_size"
    L_name: str = "num_hidden_layers"
    Q_name: str = "num_attention_heads"
    K_name: str = "num_key_value_heads"
    H_name: str = "head_dim"

    layernorm_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] = (
        _gemma2_layernorm_fn
    )


gemma2_2b_config = Gemma2Config(
    hf_model_id="google/gemma-2-2b",
    is_chat_model=False,
    has_default_sys_prompt=False,
)

gemma2_tiny_random_instruct_config = Gemma2Config(
    hf_model_id="yujiepan/gemma-2-tiny-random",
    is_chat_model=True,
    has_default_sys_prompt=True,
)


def get_subject_config(hf_model_id: str):
    if hf_model_id == "meta-llama/Meta-Llama-3-8B":
        return llama3_8B_config
    elif hf_model_id == "meta-llama/Llama-3.1-8B":
        return llama31_8B_config
    elif hf_model_id == "meta-llama/Llama-3.1-8B-Instruct":
        return llama31_8B_instruct_config
    elif hf_model_id == "meta-llama/Llama-3.1-70B-Instruct":
        return llama31_70B_instruct_config
    elif hf_model_id == "google/gemma-2-2b":
        return gemma2_2b_config
    # tiny random models for testing
    elif hf_model_id == "yujiepan/gemma-2-tiny-random":
        return gemma2_tiny_random_instruct_config
    elif hf_model_id == "yujiepan/llama-3.1-tiny-random":
        return llama31_tiny_random_instruct_config
    else:
        raise ValueError(f"Unsupported hf_model_id={hf_model_id}")


def make_subject(config: LMConfig, cast_to_hf_config_dtype: bool = True, dispatch: bool = True):
    return Subject(
        config,
        cast_to_hf_config_dtype=cast_to_hf_config_dtype,
        nnsight_lm_kwargs={"dispatch": dispatch},
    )
