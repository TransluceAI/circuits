import os
from collections import namedtuple
from typing import Literal

import numpy as np
import torch as t
from dictionary_learning import JumpReluAutoEncoder
from dictionary_learning.dictionary import IdentityDict
from huggingface_hub import hf_hub_download, list_repo_files
from sae_lens.toolkit.pretrained_sae_loaders import llama_scope_sae_huggingface_loader
from tqdm import tqdm

from .modeling_utils import Submodule

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids", "transcoders"])


GEMMA_REPO_ID_2B = {
    "embed": "google/gemma-scope-2b-pt-res",
    "attn": "google/gemma-scope-2b-pt-att",
    "mlp": "google/gemma-scope-2b-pt-mlp",
    "resid": "google/gemma-scope-2b-pt-res",
    "transcoder": "google/gemma-scope-2b-pt-transcoders",
}

GEMMA_REPO_ID_9B = {
    "embed": "google/gemma-scope-9b-pt-res",
    "attn": "google/gemma-scope-9b-pt-att",
    "mlp": "google/gemma-scope-9b-pt-mlp",
    "resid": "google/gemma-scope-9b-pt-res",
    "transcoder": "google/gemma-scope-9b-pt-transcoders",
}

LLAMA_REPO_ID = {
    "attn": "llama_scope_lxa_8x",
    "resid": "llama_scope_lxr_8x",
    "mlp": "llama_scope_lxm_8x",
    "transcoder": "llama_scope_lxtc_8x",
}

LLAMA_REPO_ID_32x = {
    "attn": "llama_scope_lxa_32x",
    "resid": "llama_scope_lxr_32x",
    "mlp": "llama_scope_lxm_32x",
    "transcoder": "llama_scope_lxtc_32x",
}


def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid", "transcoder"],
    layer: int,
    width: Literal["16k", "65k", "131k"] = "16k",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
    module_dim: int = None,
    model_size: Literal["2b", "9b"] = "2b",
):
    if neurons:
        return IdentityDict(module_dim)

    repo_dict = GEMMA_REPO_ID_2B if model_size == "2b" else GEMMA_REPO_ID_9B
    repo_id = repo_dict[submod_type]
    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]

    # SAE lens can not be installed, so we need to have the following workaround.
    print("SAE lens can not be installed, so we need to have the following workaround.")
    print(f"Loading {optimal_file} from {repo_id}")
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=f"{optimal_file}/params.npz",
        force_download=False,
    )
    params = np.load(path_to_params)
    state_dict = {k: t.from_numpy(v).cuda() for k, v in params.items()}

    activation_dim, dict_size = state_dict["W_enc"].shape
    autoencoder = JumpReluAutoEncoder(activation_dim, dict_size)
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.to(dtype=dtype, device=device)

    if device is not None:
        device = autoencoder.W_enc.device
    return autoencoder.to(dtype=dtype, device=device)


def _load_gemma_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    include_attn: bool = True,
    include_mlp: bool = True,
    include_resid: bool = True,
    use_transcoder: bool = False,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
    module_dims: dict = None,
    use_mlp_acts: bool = False,
    width: Literal["16k", "65k", "131k"] = "16k",
    model_size: Literal["2b", "9b"] = "2b",
):
    expected_layers = 26 if model_size == "2b" else 42
    assert (
        len(model.model.layers) == expected_layers
    ), f"Not the expected number of layers for Gemma-2-{model_size.upper()}"
    if thru_layer is None:
        thru_layer = len(model.model.layers)

    attns = []
    mlps = []
    resids = []
    transcoders = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.model.embed_tokens,
            use_input=False,
        )
        dictionaries[embed] = load_gemma_sae(
            "embed",
            0,
            neurons=neurons,
            dtype=dtype,
            device=device,
            module_dim=module_dims["embed"],
            model_size=model_size,
        )
    else:
        embed = None
    for i, layer in tqdm(
        enumerate(model.model.layers[: thru_layer + 1]),
        total=thru_layer + 1,
        desc="Loading Gemma SAEs",
    ):
        if include_attn:
            attns.append(
                attn := Submodule(
                    name=f"attn_{i}", submodule=layer.self_attn.o_proj, use_input=True
                )
            )
            dictionaries[attn] = load_gemma_sae(
                "attn",
                i,
                width=width,
                neurons=neurons,
                dtype=dtype,
                device=device,
                module_dim=module_dims["attn"],
                model_size=model_size,
            )
        if include_mlp:
            if use_mlp_acts:
                if use_transcoder:
                    raise ValueError(
                        "Transcoder features are incompatible with --use_mlp_acts for Gemma"
                    )
                mlps.append(
                    mlp := Submodule(
                        name=f"mlp_{i}",
                        submodule=layer.mlp.down_proj,
                        use_input=True,
                    )
                )
            else:
                mlps.append(
                    mlp := Submodule(
                        name=f"mlp_{i}",
                        submodule=layer.mlp if use_transcoder else layer.post_feedforward_layernorm,
                        use_input=use_transcoder,
                        use_transcoder=use_transcoder,
                    )
                )
            dictionaries[mlp] = load_gemma_sae(
                "transcoder" if use_transcoder else "mlp",
                i,
                width=width,
                neurons=neurons,
                dtype=dtype,
                device=device,
                module_dim=module_dims["mlp"],
                model_size=model_size,
            )
        if include_resid:
            resids.append(
                resid := Submodule(
                    name=f"resid_{i}",
                    submodule=layer,
                    is_tuple=True,
                    use_input=False,
                )
            )
            dictionaries[resid] = load_gemma_sae(
                "resid",
                i,
                width=width,
                neurons=neurons,
                dtype=dtype,
                device=device,
                module_dim=module_dims["resid"],
                model_size=model_size,
            )

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids, transcoders), dictionaries
    else:
        submodules = [embed] if include_embed else []
        for i in range(thru_layer):
            if include_attn:
                submodules.append(attns[i])
            if include_mlp:
                submodules.append(mlps[i])
            if include_resid:
                submodules.append(resids[i])
        return submodules, dictionaries


def load_llama_sae(
    submod_type: Literal["attn", "mlp", "resid", "transcoder"],
    layer: int,
    width: Literal["8x", "32x"] = "8x",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
    module_dim: int | None = None,
):
    if neurons:
        return IdentityDict(module_dim)
    if width == "32x":
        dtype = t.bfloat16

    if submod_type == "transcoder":
        repo_id = f"fnlp/Llama3_1-8B-Base-LXTC-{width}"
        folder = f"Llama3_1-8B-Base-L{layer}TC-{width}"
        state_dict = llama_scope_sae_huggingface_loader(repo_id, folder)[1]
        activation_dim, dict_size = state_dict["W_enc"].shape
        autoencoder = JumpReluAutoEncoder(activation_dim, dict_size, device=device)
        autoencoder.load_state_dict(state_dict)
    else:
        repo_id = LLAMA_REPO_ID_32x[submod_type] if width == "32x" else LLAMA_REPO_ID[submod_type]
        autoencoder = JumpReluAutoEncoder.from_pretrained(
            load_from_sae_lens=True,
            release=repo_id,
            sae_id="_".join(repo_id.split("_")[-2:]).replace("lx", f"l{layer}"),
            dtype=dtype,
            device=device,
        )

    if device is not None:
        device = autoencoder.W_enc.device
    return autoencoder.to(dtype=dtype, device=device)


def _load_llama_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    include_attn: bool = True,
    include_mlp: bool = True,
    include_resid: bool = True,
    use_transcoder: bool = False,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
    module_dims: dict = None,
    use_mlp_acts: bool = False,
    width: Literal["8x", "32x"] = "8x",
):
    assert len(model.model.layers) == 32, "Not the expected number of layers for Llama-2-7B"
    if thru_layer is None:
        thru_layer = len(model.model.layers)

    attns = []
    mlps = []
    resids = []
    transcoders = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.model.embed_tokens,
            use_input=False,
        )
        dictionaries[embed] = load_llama_sae(
            "resid",
            0,
            neurons=neurons,
            dtype=dtype,
            device=device,
            module_dim=module_dims["embed"],
            width=width,
        )
    else:
        embed = None
    for i, layer in tqdm(
        enumerate(model.model.layers[: thru_layer + 1]),
        total=thru_layer + 1,
        desc="Loading Llama SAEs",
    ):
        if include_attn:
            attns.append(
                attn := Submodule(
                    name=f"attn_{i}", submodule=layer.self_attn.o_proj, use_input=True
                )
            )
            dictionaries[attn] = load_llama_sae(
                "attn", i, neurons=neurons, dtype=dtype, device=device, width=width
            )
            dictionaries[attn] = load_llama_sae(
                "attn",
                i,
                neurons=neurons,
                dtype=dtype,
                device=device,
                module_dim=module_dims["attn"],
                width=width,
            )
        if include_mlp:
            if use_mlp_acts:
                mlps.append(
                    mlp := Submodule(
                        name=f"mlp_{i}",
                        submodule=layer.mlp.down_proj,
                        use_input=True,
                    )
                )
            else:
                # transcoder will go into this subroutine
                mlps.append(
                    mlp := Submodule(
                        name=f"mlp_{i}",
                        submodule=layer.mlp,
                        use_input=True if use_transcoder else False,
                        use_transcoder=use_transcoder,
                    )
                )
            dictionaries[mlp] = load_llama_sae(
                "transcoder" if use_transcoder else "mlp",
                i,
                neurons=neurons,
                dtype=dtype,
                device=device,
                module_dim=module_dims["mlp"],
                width=width,
            )
        if include_resid:
            resids.append(
                resid := Submodule(
                    name=f"resid_{i}",
                    submodule=layer,
                    is_tuple=True,
                    use_input=False,
                )
            )
            dictionaries[resid] = load_llama_sae(
                "resid",
                i,
                neurons=neurons,
                dtype=dtype,
                device=device,
                module_dim=module_dims["resid"],
                width=width,
            )

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids, transcoders), dictionaries
    else:
        submodules = [embed] if include_embed else []
        for i in range(thru_layer):
            if include_attn:
                submodules.append(attns[i])
            if include_mlp:
                submodules.append(mlps[i])
            if include_resid:
                submodules.append(resids[i])
        return submodules, dictionaries


def load_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    include_attn: bool = True,
    include_mlp: bool = True,
    include_resid: bool = True,
    use_transcoder: bool = False,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
    module_dims: dict = None,
    use_mlp_acts: bool = False,
    width: str = "8x",
):
    model_name = model.config._name_or_path

    if model_name == "EleutherAI/pythia-70m-deduped":
        assert False, "Pythia is not supported"
    elif model_name in ("google/gemma-2-2b", "google/gemma-2-9b"):
        gemma_width = width if width in ("16k", "65k", "131k") else "16k"
        model_size = "2b" if "2b" in model_name else "9b"
        return _load_gemma_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            include_attn=include_attn,
            include_mlp=include_mlp,
            include_resid=include_resid,
            use_transcoder=use_transcoder,
            neurons=neurons,
            dtype=dtype,
            device=device,
            module_dims=module_dims,
            use_mlp_acts=use_mlp_acts,
            width=gemma_width,
            model_size=model_size,
        )
    elif (
        model_name == "meta-llama/Llama-3.1-8B" or model_name == "meta-llama/Llama-3.1-8B-Instruct"
    ):
        return _load_llama_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            include_attn=include_attn,
            include_mlp=include_mlp,
            include_resid=include_resid,
            use_transcoder=use_transcoder,
            neurons=neurons,
            dtype=dtype,
            device=device,
            module_dims=module_dims,
            use_mlp_acts=use_mlp_acts,
            width=width,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
