"""
Generate circuits for the Wikipedia user modelling case study.
"""

from __future__ import annotations

from pathlib import Path

from circuits.analysis.circuit_ops import Circuit
from user_modeling.datasets.wikipedia import (
    WIKIPEDIA_PROMPT_RESPONSE,
    get_wikipedia_dataset_by_split,
)
from util.subject import Subject, llama31_8B_instruct_config

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

NUM_EXAMPLES = 128
SPLITS = ("country", "gender", "occupation", "religion")
OUTPUT_DIR = Path(ENV.ARTIFACTS_DIR) / "results" / "case_studies" / "user_modelling"


def _prepare_split_examples(
    subject: Subject, split: str, seed: int = 0, num_examples: int = NUM_EXAMPLES
):
    """Collect prompts, seed responses, and labels for a given attribute split."""
    datasets = get_wikipedia_dataset_by_split(
        subject,
        question_types=[split],
        only_good=True,
        seed=seed,
    )
    combined = (
        datasets["train"].datapoints + datasets["valid"].datapoints + datasets["test"].datapoints
    )
    if len(combined) < num_examples:
        raise ValueError(
            f"Not enough datapoints for split '{split}' (needed {num_examples}, got {len(combined)})."
        )

    selected = combined[:num_examples]

    prompts = [dp.conversation[0]["content"] for dp in selected]
    labels = [dp.latent_attributes[split][0] for dp in selected]
    seed_responses = [f"{WIKIPEDIA_PROMPT_RESPONSE} {split} ="] * num_examples
    return prompts, seed_responses, labels


def all_user_modelling_circuits():
    subject = Subject(llama31_8B_instruct_config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        prompts, seed_responses, labels = _prepare_split_examples(subject, split)
        circuit = Circuit.from_dataset(
            subject,
            prompts,
            seed_responses,
            labels,
            return_nodes_only=False,
            neurons=None,
            percentage_threshold=0.005,
            batch_size=1,
            verbose=False,
            k=5,
            apply_blacklist=True,
        )
        out_path = OUTPUT_DIR / f"wikipedia_{split}_circuit.pkl"
        circuit.save_to_pickle(str(out_path))
        print(f"Saved {split} circuit with {NUM_EXAMPLES} examples to {out_path}")


def single_user_modelling_country_circuit():
    subject = Subject(llama31_8B_instruct_config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts, seed_responses, labels = _prepare_split_examples(
        subject, "country", num_examples=1, seed=42
    )
    circuit = Circuit.from_dataset(
        subject,
        prompts,
        seed_responses,
        labels,
        return_nodes_only=False,
        neurons=None,
        percentage_threshold=0.0025,
        batch_size=1,
        verbose=False,
        k=5,
        apply_blacklist=True,
    )
    out_path = OUTPUT_DIR / f"japan_circuit.pkl"
    circuit.save_to_pickle(str(out_path))
    print(f"Saved Japan circuit with 1 example to {out_path}")


def main():
    single_user_modelling_country_circuit()


if __name__ == "__main__":
    main()
