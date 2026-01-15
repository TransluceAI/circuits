import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from circuits.utils.constants import DATASET_TASK_MAPPING
from user_modeling.datasets.gender import Datapoint, GenderDatapoint, get_gender_dataset_by_split
from user_modeling.datasets.wikipedia import (
    ATTRIBUTE_PROMPTS,
    WIKIPEDIA_PROMPT_RESPONSE,
    WikipediaDatapoint,
    get_wikipedia_dataset_by_split,
)
from util.subject import Subject

from env_util import ENV

OBITUARY_SEED_RESPONSE = """
Here is a possible obituary for your friend:

"[Friend's name] passed away on [date] after a long and fulfilling life.
""".strip()

OBITUARY_FIRST_PERSON_SEED_RESPONSE = """
Here is a possible obituary for you:

"[Your name] passed away on [date] after a long and fulfilling life.
""".strip()

SEED_RESPONSES = {
    "obituary": OBITUARY_SEED_RESPONSE,
    "obituary_first_person": OBITUARY_FIRST_PERSON_SEED_RESPONSE,
    "country": WIKIPEDIA_PROMPT_RESPONSE + ATTRIBUTE_PROMPTS["country"],
    "gender": WIKIPEDIA_PROMPT_RESPONSE + ATTRIBUTE_PROMPTS["gender"],
    "occupation": WIKIPEDIA_PROMPT_RESPONSE + ATTRIBUTE_PROMPTS["occupation"],
    "religion": WIKIPEDIA_PROMPT_RESPONSE + ATTRIBUTE_PROMPTS["religion"],
}


def format_conversation(prompt: str, answer: str):
    return f"User: {prompt}\nAssistant: {answer}"


def format_row(row: Datapoint, split: str) -> dict[str, str]:
    if isinstance(row, GenderDatapoint):
        probs = row.all_probs
        assert probs is not None
        max_key = max(probs.items(), key=lambda x: x[1])[0]
        answer = " He" if max_key == "male" else " She" if max_key == "female" else " They"
        return {
            "prefix": format_conversation(row.conversation[0]["content"], SEED_RESPONSES[split]),
            "answer": answer,
            "case": max_key,
        }
    elif isinstance(row, WikipediaDatapoint):
        answer = " " + row.latent_attributes[split][0].capitalize()
        return {
            "prefix": format_conversation(row.conversation[0]["content"], SEED_RESPONSES[split]),
            "answer": answer,
            "case": row.latent_attributes[split][0],
        }
    else:
        raise ValueError(f"Invalid row type: {type(row)}")


def make_user_model_examples(
    subject: Subject,
    split: str,
    mode: Literal["pair", "nopair"] = "pair",
    train_examples: int = 1000,
    test_examples: int = 100,
    output_path: str | None = None,
):
    if output_path is None:
        if ENV.ARTIFACTS_DIR is None:
            raise RuntimeError("ARTIFACTS_DIR must be set in the .env file or output_path must be provided")
        output_path = str(Path(ENV.ARTIFACTS_DIR) / "circuits" / "data" / "gender_examples")
    # load dataset
    if split in ["obituary"]:
        dataset = get_gender_dataset_by_split(subject, [split])
    elif split in ["country", "gender", "occupation", "religion"]:
        dataset = get_wikipedia_dataset_by_split(subject, [split], only_good=True)
    else:
        raise ValueError(f"Invalid dataset: {split}")

    # make examples
    if mode == "pair":
        for subset in dataset.keys():
            examples = []
            for _ in range(train_examples if split == "train" else test_examples):
                # sample two different classes of examples
                clean, patch = None, None
                while clean is None or patch is None:
                    clean, patch = random.sample(dataset[subset].datapoints, 2)
                    clean, patch = format_row(clean, split), format_row(patch, split)
                    if clean["answer"] == patch["answer"]:
                        clean, patch = None, None

                # store
                examples.append(
                    {
                        "clean_prefix": clean["prefix"],
                        "clean_answer": clean["answer"],
                        "patch_prefix": patch["prefix"],
                        "patch_answer": patch["answer"],
                        "case": clean["case"],
                    }
                )

            # write to file
            with open(f"{output_path}/{split}_{subset}.json", "w") as f:
                for example in examples:
                    f.write(json.dumps(example) + "\n")

    elif mode == "nopair":
        for subset in dataset.keys():
            examples = []
            for _ in range(train_examples if split == "train" else test_examples):
                row = random.choice(dataset[subset].datapoints)
                example = format_row(row, split)
                examples.append(example)

            # write to file
            with open(f"{output_path}/{split}_nopair_{subset}.json", "w") as f:
                for example in examples:
                    f.write(json.dumps(example) + "\n")


def load_user_modeling_nopair_examples(
    dataset: str,
    dataset_path: str,
    num_examples: int,
    *,
    seed: int | None = None,
) -> list[dict]:
    """Load non-paired user modeling examples."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Nopair dataset path '{dataset_path}' does not exist. "
            "Please ensure the data_path in the config points to the generated JSONL file."
        )
    with path.open("r") as f:
        rows = [json.loads(line) for line in f]

    if seed is not None:
        random.Random(seed).shuffle(rows)

    if num_examples is None or num_examples >= len(rows):
        return rows
    return rows[:num_examples]


def load_examples(
    dataset: str,
    n_examples: int,
    model,
    use_min_length_only: bool = False,
    max_length: int | None = None,
    seed: int = 12,
    allow_length_mismatch: bool = False,
    enforce_pad: bool = False,
    apply_chat_template: bool = False,
):
    with open(dataset, "r") as f:
        dataset_items = f.readlines()
    random.Random(seed).shuffle(dataset_items)

    # prepare pad token
    pad_token = model.tokenizer.decode(model.tokenizer.pad_token_id)

    examples = []
    if use_min_length_only:
        min_length = float("inf")
    for line in dataset_items:
        data = json.loads(line)

        # need to preprocess if applying chat template
        if apply_chat_template:
            for key in ["clean_prefix", "patch_prefix"]:
                user, assistant = data[key].split("\nAssistant: ")
                user = user.replace("User: ", "")
                data[key] = model.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )

        clean_prefix = model.tokenizer(data["clean_prefix"]).input_ids
        patch_prefix = model.tokenizer(data["patch_prefix"]).input_ids
        clean_answer = model.tokenizer(data["clean_answer"]).input_ids
        patch_answer = model.tokenizer(data["patch_answer"]).input_ids
        clean_full = model.tokenizer(data["clean_prefix"] + data["clean_answer"]).input_ids
        patch_full = model.tokenizer(data["patch_prefix"] + data["patch_answer"]).input_ids

        # strip BOS token from response if necessary
        if clean_answer[0] == model.tokenizer.bos_token_id:
            clean_answer = clean_answer[1:]
        if patch_answer[0] == model.tokenizer.bos_token_id:
            patch_answer = patch_answer[1:]

        # check that answer is one token
        if len(clean_answer) != 1 or len(patch_answer) != 1:
            continue

        # check that prefixes are the same length
        if not allow_length_mismatch:
            if len(clean_prefix) != len(patch_prefix):
                continue

        # check for tokenization mismatches
        if clean_prefix + clean_answer != clean_full:
            continue
        if patch_prefix + patch_answer != patch_full:
            continue

        if max_length is not None and len(clean_prefix) > max_length:
            continue

        if use_min_length_only:
            # restart collection if we've found a new shortest example
            if (length := max(len(clean_prefix), len(patch_prefix))) < min_length:
                examples = []  # restart collection
                min_length = length
            # skip if too long
            elif length > min_length:
                continue

        examples.append(data)

        if len(examples) >= n_examples:
            break

    # tokenize and pad
    if enforce_pad:
        max_token_length = 0
        for i in range(len(examples)):
            clean_prefix = model.tokenizer(examples[i]["clean_prefix"]).input_ids
            patch_prefix = model.tokenizer(examples[i]["patch_prefix"]).input_ids
            max_token_length = max(max_token_length, len(clean_prefix), len(patch_prefix))

        for i in range(len(examples)):
            clean_prefix = model.tokenizer(examples[i]["clean_prefix"]).input_ids
            patch_prefix = model.tokenizer(examples[i]["patch_prefix"]).input_ids
            examples[i]["clean_prefix"] = (
                pad_token * (max_token_length - len(clean_prefix)) + examples[i]["clean_prefix"]
            )
            examples[i]["patch_prefix"] = (
                pad_token * (max_token_length - len(patch_prefix)) + examples[i]["patch_prefix"]
            )

    # print stats
    print(
        "clean_prefix_lengths",
        Counter([len(model.tokenizer(example["clean_prefix"]).input_ids) for example in examples]),
    )
    print(
        "patch_prefix_lengths",
        Counter([len(model.tokenizer(example["patch_prefix"]).input_ids) for example in examples]),
    )

    return examples


def load_examples_helper(dataset, dataset_path, num_examples, model, nopair=False, **kwargs):
    """
    Generic helper function to load examples from various datasets.

    Args:
        dataset_path (str): Path to the dataset file
        num_examples (int): Number of examples to load
        model: The language model
        nopair (bool): Whether to load examples without pairs
        **kwargs: Additional arguments to pass to the specific loading function

    Returns:
        list: Loaded examples

    Raises:
        ValueError: If no examples could be loaded
    """
    if DATASET_TASK_MAPPING[dataset] == "sva":
        # Set default for use_min_length_only if not provided
        if "use_min_length_only" not in kwargs:
            kwargs["use_min_length_only"] = True
        examples = load_examples(
            dataset_path,
            num_examples,
            model,
            **kwargs,
        )
    elif DATASET_TASK_MAPPING[dataset] == "user_modeling":
        if "use_min_length_only" not in kwargs:
            kwargs["use_min_length_only"] = False
        if "apply_chat_template" not in kwargs:
            kwargs["apply_chat_template"] = True
        if "enforce_pad" not in kwargs:
            kwargs["enforce_pad"] = False
        if "allow_length_mismatch" not in kwargs:
            kwargs["allow_length_mismatch"] = True
        examples = load_examples(
            dataset_path,
            num_examples,
            model,
            **kwargs,
        )
    elif DATASET_TASK_MAPPING[dataset] == "user_modeling_nopair":
        seed = kwargs.get("seed", None)
        examples = load_user_modeling_nopair_examples(
            dataset,
            dataset_path,
            num_examples,
            seed=seed,
        )

    if examples is None:
        raise ValueError(f"Failed to load examples from {dataset_path}")

    return examples


def get_annotation(dataset, model, data):
    # First, understand which dataset we're working with
    structure = None
    if "within_rc" in dataset:
        structure = "within_rc"
        template = "the_subj subj_main that the_dist subj_dist"
    elif "rc.json" in dataset or "rc_" in dataset:
        structure = "rc"
        template = "the_subj subj_main that the_dist subj_dist verb_dist"
    elif "simple.json" in dataset or "simple_" in dataset:
        structure = "simple"
        template = "the_subj subj_main"
    elif "nounpp.json" in dataset or "nounpp_" in dataset:
        structure = "nounpp"
        template = "the_subj subj_main prep the_dist subj_dist"

    if structure is None:
        return {}

    annotations = {}

    # Iterate through words in the template and input. Get token spans
    curr_token = 0
    for template_word, word in zip(template.split(), data["clean_prefix"].split()):
        if word != "The":
            word = " " + word
        word_tok = model.tokenizer(word, return_tensors="pt", padding=False).input_ids
        num_tokens = word_tok.shape[1]
        span = (curr_token, curr_token + num_tokens - 1)
        curr_token += num_tokens
        annotations[template_word] = span

    return annotations
