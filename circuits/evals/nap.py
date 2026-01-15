import gc
import math
from collections import namedtuple
from functools import partial

import torch as t
from circuits.evals.base import BaseMethod, absolute_logit_metric_fn, logit_difference_metric_fn
from circuits.utils.dictionary_loading_utils import load_saes_and_submodules
from circuits.utils.modeling_utils import SparseAct, Submodule
from dictionary_learning.dictionary import Dictionary, IdentityDict
from tqdm import tqdm

EffectOut = namedtuple("EffectOut", ["effects", "deltas", "grads", "total_effect"])


def patching_effect_ig(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
    suffix_length=None,
):
    hidden_states_clean = {}
    with t.no_grad(), model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(
                act=f.save(), res=residual.save()
            )  # type: ignore
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k: SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res))
            for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with t.no_grad(), model.trace(patch):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(
                    act=f.save(), res=residual.save()
                )  # type: ignore
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k: v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace() as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.requires_grad_().retain_grad()
                f.res.requires_grad_().retain_grad()
                fs.append(f)
                with tracer.invoke(clean):
                    submodule.set_activation(dictionary.decode(f.act) + f.res)
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward()

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)  # type: ignore
        delta = (
            (patch_state - clean_state).detach()
            if patch_state is not None
            else clean_state.detach()
        )
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    # Apply suffix_length filtering if specified
    if suffix_length is not None:
        effects = _apply_suffix_length_filter(effects, clean, model, suffix_length)

    return EffectOut(effects, deltas, grads, total_effect)


def patching_effect_eap_ig_inputs(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
    suffix_length=None,
):
    embed_clean = None
    hidden_states_clean = {}
    with t.no_grad(), model.trace(clean):
        embed_clean = model.model.embed_tokens.output.save()
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(
                act=f.save(), res=residual.save()
            )  # type: ignore
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        embed_patch = t.zeros_like(embed_clean)
        hidden_states_patch = {
            k: SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res))
            for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        embed_patch = None
        hidden_states_patch = {}
        with t.no_grad(), model.trace(patch):
            embed_patch = model.model.embed_tokens.output.save()
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(
                    act=f.save(), res=residual.save()
                )  # type: ignore
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k: v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}

    # for each submodule, compute the effect
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]

        # first collect the hidden act when processed by SAE
        hidden_states = []
        with model.trace() as tracer:
            for step in range(steps):
                alpha = step / steps
                f_embed = (1 - alpha) * embed_clean + alpha * embed_patch
                with t.no_grad(), tracer.invoke(clean):
                    model.model.embed_tokens.output = f_embed
                    x = submodule.get_activation()
                    f = dictionary.encode(x)
                    x_hat = dictionary.decode(f)
                    residual = x - x_hat
                    hidden_states.append(SparseAct(act=f.save(), res=residual.save()))  # type: ignore
        assert len(hidden_states) == steps

        # now do a dummy intervention (no actual change) to collect the grad
        with model.trace() as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f_embed = (1 - alpha) * embed_clean + alpha * embed_patch
                hidden_state = hidden_states[step]
                act = hidden_state.act.clone()
                res = hidden_state.res.clone()
                act.requires_grad_().retain_grad()
                res.requires_grad_().retain_grad()
                fs.append(SparseAct(act=act, res=res))
                with tracer.invoke(clean):
                    model.model.embed_tokens.output = f_embed
                    # set activation in dictionary space
                    submodule.set_activation(dictionary.decode(act) + res)
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward()

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)  # type: ignore
        delta = (
            (patch_state - clean_state).detach()
            if patch_state is not None
            else -clean_state.detach()
        )
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    # Apply suffix_length filtering if specified
    if suffix_length is not None:
        effects = _apply_suffix_length_filter(effects, clean, model, suffix_length)

    return EffectOut(effects, deltas, grads, total_effect)


def patching_effect_conductance_inputs(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
    suffix_length=None,
):
    pass


def patching_effect_random(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
    suffix_length=None,
):
    """
    Returns random effects with the same structure as patching_effect_ig.
    Effects are random, while grads and deltas are set to zero.
    """
    # Get clean states to determine shapes
    hidden_states_clean = {}
    with t.no_grad(), model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(
                act=f.save(), res=residual.save()
            )  # type: ignore
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

    # Compute total effect if patch is provided
    if patch is None:
        total_effect = None
    else:
        with t.no_grad(), model.trace(patch):
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()

    effects = {}
    deltas = {}
    grads = {}

    for submodule in submodules:
        clean_state = hidden_states_clean[submodule]

        # Create random effects as a SparseAct object
        random_act = t.randn_like(clean_state.act)
        random_resc = t.randn_like(clean_state.res).sum(dim=-1, keepdim=True)
        effect = SparseAct(act=random_act, resc=random_resc)

        # Set deltas and grads to zero with correct shapes
        zero_delta = SparseAct(act=t.zeros_like(clean_state.act), res=t.zeros_like(clean_state.res))
        zero_grad = SparseAct(act=t.zeros_like(clean_state.act), res=t.zeros_like(clean_state.res))

        effects[submodule] = effect
        deltas[submodule] = zero_delta
        grads[submodule] = zero_grad

    # Apply suffix_length filtering if specified
    if suffix_length is not None:
        effects = _apply_suffix_length_filter(effects, clean, model, suffix_length)

    return EffectOut(effects, deltas, grads, total_effect)


def patching_effect_delta_selection(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
    suffix_length=None,
):
    """
    Uses the delta (difference) between clean and patched activations.
    Effects are computed based on the magnitude of activation differences for all neurons.
    """
    # Get clean states
    hidden_states_clean = {}
    with t.no_grad(), model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(
                act=f.save(), res=residual.save()
            )  # type: ignore
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

    # Get patch states and compute total effect
    if patch is None:
        hidden_states_patch = {
            k: SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res))
            for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with t.no_grad(), model.trace(patch):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(
                    act=f.save(), res=residual.save()
                )  # type: ignore
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k: v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}

    for submodule in submodules:
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]

        # Compute deltas (differences between patch and clean)
        delta = patch_state - clean_state

        # Create effect based on all neurons as a SparseAct object
        # Use the absolute deltas for all neurons as the effect
        effect_act = t.abs(delta.act)
        effect_resc = t.abs(delta.res).sum(dim=-1, keepdim=True)
        effect = SparseAct(act=effect_act, resc=effect_resc)

        # Set grads to zero (as requested)
        zero_grad = SparseAct(act=t.zeros_like(clean_state.act), res=t.zeros_like(clean_state.res))

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = zero_grad

    # Apply suffix_length filtering if specified
    if suffix_length is not None:
        effects = _apply_suffix_length_filter(effects, clean, model, suffix_length)

    return EffectOut(effects, deltas, grads, total_effect)


def _apply_suffix_length_filter(effects, clean_inputs, model, suffix_length):
    """
    Zero out effects at token positions that are before the suffix.

    Args:
        effects: Dictionary mapping submodules to SparseAct effects
        clean_inputs: List of clean input strings
        model: The language model with tokenizer
        suffix_length: Number of tokens from the end to keep (others get zeroed)

    Returns:
        Dictionary with filtered effects
    """
    # Get the sequence length from the first input
    if isinstance(clean_inputs, list):
        seq_length = len(model.tokenizer(clean_inputs[0]).input_ids)
    else:
        # Handle single string case
        seq_length = len(model.tokenizer(clean_inputs).input_ids)

    # Calculate the position from which to start keeping effects
    # If suffix_length is 5 and seq_length is 20, we zero out positions 0-14 and keep 15-19
    keep_start_pos = max(0, seq_length - suffix_length)

    filtered_effects = {}
    for submodule, effect in effects.items():
        # Create a copy of the effect
        filtered_effect = effect.clone()

        # Zero out positions before the suffix for both act and res/resc
        if filtered_effect.act is not None:
            # Assuming shape is (batch_size, seq_length, features)
            if len(filtered_effect.act.shape) >= 2:
                filtered_effect.act[:, :keep_start_pos] = 0

        if filtered_effect.res is not None:
            if len(filtered_effect.res.shape) >= 2:
                filtered_effect.res[:, :keep_start_pos] = 0

        if filtered_effect.resc is not None:
            if len(filtered_effect.resc.shape) >= 2:
                filtered_effect.resc[:, :keep_start_pos] = 0

        filtered_effects[submodule] = filtered_effect

    return filtered_effects


class NAP(BaseMethod):
    """Implementation of sparse feature circuit.
    ref: https://arxiv.org/pdf/2403.19647

    It seems like this method does not require edge weights for
    their own circuit eval.
    """

    def __str__(self):
        return f"NAP-{self.effect_method}" if self.effect_method != "ig" else "NAP"

    def __init__(self, model, args, mode, **kwargs):
        super().__init__(model, args, mode, **kwargs)

        # Effect computation method selection
        self.effect_method = getattr(args, "effect_method", "ig")  # 'ig', 'random', or 'delta'

        # Suffix length for token filtering
        self.suffix_length = getattr(args, "suffix_length", None)

        self.submodules, self.dictionaries = load_saes_and_submodules(
            model,
            separate_by_type=True,
            include_embed="embed" in self.submodule_types if self.mode == "train" else False,
            include_attn="attn" in self.submodule_types,
            include_mlp="mlp" in self.submodule_types,
            include_resid="resid" in self.submodule_types,
            use_transcoder=self.use_transcoder,
            neurons=self.use_neurons,
            device=self.device,
            dtype=self.dtype,
            module_dims=self.module_dims,
            use_mlp_acts=self.use_mlp_acts,
            width=self.width,
        )

        # main artifacts of the method
        self.nodes = None
        self.edges = None

    def make_dataloader(self, examples, **kwargs):
        """This method does not need a dataloader."""
        n_batches = math.ceil(len(examples) / self.batch_size)
        batches = [
            examples[batch * self.batch_size : (batch + 1) * self.batch_size]
            for batch in range(n_batches)
        ]
        return batches

    def train(self, examples, **kwargs):
        dataloader = self.make_dataloader(examples, **kwargs)
        steps = kwargs.get("steps", 10)

        running_nodes = None

        for batch in tqdm(dataloader, desc="Training"):
            clean_inputs = [e["clean_prefix"] for e in batch]
            clean_answer_idxs = t.tensor(
                [self.tokenizer(e["clean_answer"]).input_ids[-1] for e in batch],
                dtype=t.long,
                device=self.device,
            )
            if self.nopair:
                patch_inputs = None
                metric_fn = partial(absolute_logit_metric_fn, answer_idxs=clean_answer_idxs)
            else:
                patch_inputs = [e["patch_prefix"] for e in batch]
                patch_answer_idxs = t.tensor(
                    [self.tokenizer(e["patch_answer"]).input_ids[-1] for e in batch],
                    dtype=t.long,
                    device=self.device,
                )
                metric_fn = partial(
                    logit_difference_metric_fn,
                    patch_answer_idxs=patch_answer_idxs,
                    clean_answer_idxs=clean_answer_idxs,
                )

            # serialize the submodules
            all_submods = []

            # Add embed if present
            if self.submodules.embed is not None:
                all_submods.append(self.submodules.embed)

            for i in range(self.n_layers):
                if self.submodules.attns:
                    all_submods.append(self.submodules.attns[i])
                if self.submodules.mlps:
                    all_submods.append(self.submodules.mlps[i])
                if self.submodules.resids:
                    all_submods.append(self.submodules.resids[i])

            # get the effects of the submodules (nodes)
            if self.effect_method == "ig":
                effect_fn = patching_effect_ig
            elif self.effect_method == "random":
                effect_fn = patching_effect_random
            elif self.effect_method == "delta":
                effect_fn = patching_effect_delta_selection
            elif self.effect_method == "ig-inputs":
                effect_fn = patching_effect_eap_ig_inputs
            elif self.effect_method == "conductance":
                effect_fn = patching_effect_conductance_inputs
            else:
                raise ValueError(f"Unknown effect method: {self.effect_method}")

            effects, _, _, total_effect = effect_fn(
                clean_inputs,
                patch_inputs,
                self.model,
                all_submods,
                self.dictionaries,
                metric_fn,
                steps=steps,
                metric_kwargs=dict(),
                suffix_length=self.suffix_length,
            )

            # format the effects into nodes
            nodes = {}
            if total_effect is not None:
                nodes = {"y": total_effect}
            if self.submodules.embed is not None:
                nodes["embed"] = effects[self.submodules.embed]
            for i in range(self.n_layers):
                if self.submodules.attns:
                    nodes[f"attn_{i}"] = effects[self.submodules.attns[i]]
                if self.submodules.mlps:
                    nodes[f"mlp_{i}"] = effects[self.submodules.mlps[i]]
                if self.submodules.resids:
                    nodes[f"resid_{i}"] = effects[self.submodules.resids[i]]

            if self.aggregation == "sum":
                for k in nodes:
                    if k != "y":
                        nodes[k] = nodes[k].sum(dim=1)
            nodes = {k: v.mean(dim=0) for k, v in nodes.items()}

            # accumulate the nodes and edges
            if running_nodes is None:
                running_nodes = {
                    k: len(batch) * nodes[k].to("cpu") for k in nodes.keys() if k != "y"
                }
            else:
                for k in nodes.keys():
                    if k != "y":
                        running_nodes[k] += len(batch) * nodes[k].to("cpu")

            # memory cleanup
            del nodes
            gc.collect()
        self.nodes = {k: v.to(self.device) / len(examples) for k, v in running_nodes.items()}

    def load(self, dump_dir, **kwargs):
        start_layer = kwargs["start_layer"]
        # remove modules before start_layer
        serialized_submodules = []
        if isinstance(self.submodules, list):
            serialized_submodules = self.submodules
        else:
            for submods in self.submodules:
                if submods is not None and len(submods) != 0:
                    for submod in submods:
                        if int(submod.name.split("_")[-1]) >= start_layer:
                            serialized_submodules.append(submod)

        self.submodules = serialized_submodules
        self.neuron_dicts = {
            submod: IdentityDict(self.dictionaries[submod].activation_dim)
            for submod in self.submodules
        }

        circuit = t.load(dump_dir, map_location=self.device)
        self.nodes = circuit["nodes"] if "nodes" in circuit else None
