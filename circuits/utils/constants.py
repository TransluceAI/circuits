import torch as t
from circuits.evals.enap import ENAP
from circuits.evals.nap import NAP
from util.subject import gemma2_2b_config, llama31_8B_config, llama31_8B_instruct_config

UNIT_1M = 1_000_000

N_LAYERS_MAPPING = {
    "google/gemma-2-2b": 26,
    "google/gemma-2-9b": 42,
    "meta-llama/Llama-3.1-8B": 32,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
}

PARALLEL_ATTN_MAPPING = {
    "google/gemma-2-2b": False,
    "google/gemma-2-9b": False,
    "meta-llama/Llama-3.1-8B": False,
    "meta-llama/Llama-3.1-8B-Instruct": False,
}

INCLUDE_EMBED_MAPPING = {
    "google/gemma-2-2b": False,
    "google/gemma-2-9b": False,
    "meta-llama/Llama-3.1-8B": False,
    "meta-llama/Llama-3.1-8B-Instruct": False,
}

DTYPE_MAPPING = {
    "google/gemma-2-2b": t.bfloat16,
    "google/gemma-2-9b": t.bfloat16,
    "meta-llama/Llama-3.1-8B": t.bfloat16,
    "meta-llama/Llama-3.1-8B-Instruct": t.bfloat16,
}

METHOD_MAPPING = {
    "nap": NAP,
    "enap": ENAP,
}

START_LAYER_MAPPING = {
    "google/gemma-2-2b": 0,
    "google/gemma-2-9b": 0,
    "meta-llama/Llama-3.1-8B": 0,
    "meta-llama/Llama-3.1-8B-Instruct": 0,
}

THRESHOLD_RANGE_MAPPING = {
    "google/gemma-2-2b": [0] + t.logspace(-4, 0, 15).tolist(),
    "google/gemma-2-9b": [0] + t.logspace(-4, 0, 15).tolist(),
    "meta-llama/Llama-3.1-8B": [0] + t.logspace(-6, 0, 30).tolist(),
    "meta-llama/Llama-3.1-8B-Instruct": [0] + t.logspace(-6, 0, 30).tolist(),
}

DATASET_TASK_MAPPING = {
    "simple": "sva",
    "nounpp": "sva",
    "rc": "sva",
    "within_rc": "sva",
    "gender": "user_modeling",
    "gender_nopair": "user_modeling_nopair",
    "country": "user_modeling",
    "country_nopair": "user_modeling_nopair",
    "occupation": "user_modeling",
    "occupation_nopair": "user_modeling_nopair",
    "religion": "user_modeling",
    "religion_nopair": "user_modeling_nopair",
    "obituary": "user_modeling",
    "obituary_nopair": "user_modeling_nopair",
}

SUBJECT_CONFIG_MAPPING = {
    "meta-llama/Llama-3.1-8B": llama31_8B_config,
    "meta-llama/Llama-3.1-8B-Instruct": llama31_8B_instruct_config,
    "google/gemma-2-2b": gemma2_2b_config,
}
