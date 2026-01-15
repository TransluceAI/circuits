"""
Generate circuits for the math case study. Excludes edges.
"""

import os
from pathlib import Path
from circuits.analysis.circuit_ops import Circuit
from util.subject import Subject, llama31_8B_instruct_config

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

ARTIFACTS_DIR = Path(ENV.ARTIFACTS_DIR) / "math_circuit_hypotheses"

subject = Subject(llama31_8B_instruct_config)
tokenizer = subject.tokenizer


def all_math_circuits():
    prompts = [f"What is {x} + {y}?" for x in range(100) for y in range(100)]
    seed_responses = ["Answer: "] * len(prompts)
    labels = [f"{x} + {y} = {x + y}" for x in range(100) for y in range(100)]

    # convert to dataframes
    circuit = Circuit.from_dataset(
        subject,
        prompts,
        seed_responses,
        labels,
        return_nodes_only=True,
        neurons=500,
        batch_size=4,
        verbose=False,
    )

    os.makedirs(ARTIFACTS_DIR / "case_studies", exist_ok=True)
    circuit.save_to_pickle(ARTIFACTS_DIR / "case_studies" / "math_circuit.pkl")


def add_36_59():
    x = 36
    y = 59
    prompt = f"What is {x} + {y}?"
    seed_response = "Answer: "
    label = f"{x} + {y} = {x + y}"

    # convert to dataframes
    circuit = Circuit.from_dataset(
        subject,
        [prompt],
        [seed_response],
        [label],
        return_nodes_only=False,
        neurons=500,
        batch_size=1,
        verbose=True,
        apply_blacklist=True,
    )

    os.makedirs(ARTIFACTS_DIR / "case_studies", exist_ok=True)
    circuit.save_to_pickle(ARTIFACTS_DIR / "case_studies" / "36_plus_59_circuit.pkl")


if __name__ == "__main__":
    add_36_59()
