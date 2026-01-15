"""
Large-scale but highly templatic user modeling dataset where the next token is the attribute of
interest, using Wikipedia infobox requests.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm import tqdm
from util.chat_input import ChatInput
from util.subject import Subject

DATA_DIR = Path("../../../../data")

WIKIPEDIA_OPTION_PROMPT = """
Could you generate {num_samples} unique possible values for the attribue "{attribute}"?

For example, for the "gender" attribute:
"male"
"female"

Format your output in the following way:
```
# Possible Values
1. <value 1>
...
N. <value N>
# End of Possible Values
```

Your values should be unique.
""".strip()

WIKIPEDIA_DESCRIPTION_PROMPT = """
Could you generate {num_samples} stereotypical descriptions, in the first-person, of a person with the attribute "{attribute}" being "{value}"? It should be **easy** to infer the attribute from the description.

Format your output in the following way:
```
# Descriptions
1. <description 1>
...
N. <description N>
# End of Descriptions
```

Remember:
- You should be be creative with your stereotypical descriptions.
- Make sure to not include any explicit mention of "{value}".
- Start each description with the pronoun "I" and ensure it is only one sentence long.
""".strip()

WIKIPEDIA_PROMPT_REQUEST = "Write a hypothetical but realistic Wikipedia biography infobox for me."
WIKIPEDIA_PROMPT_RESPONSE = "{{Infobox person\n|"
ATTRIBUTE_PROMPTS = {
    "gender": " gender =",
    "country": " country =",
    # "age": " birth_date =",
    "occupation": " occupation =",
    "religion": " religion =",
    # "net_worth": " net_worth =",
}
EXPLICIT_ATTRIBUTE_DESCRIPTIONS = {
    "gender": "I am a {value}.",
    "country": "I am from {value}.",
    "age": "I am {value} years old.",
    "occupation": "I work as a {value}.",
    "religion": "I am a {value}.",
    "net_worth": "I have a net worth of {value}.",
}

ATTRIBUTE_COUNTS = {
    "gender": 3,  # to limit to male, female, and non-binary, as in gender.py
}

def parse_completion_for_numbered_list(
    completion_text: str, start_marker: str, end_marker: str
) -> list[str]:
    """
    Parse numbered list from LLM completion text that follows a specific format.

    The text should contain sections marked with `start_marker` and `end_marker`.
    Each item is expected to be on a new line, possibly numbered.

    Args:
        completion_text: The text output from the LLM
        start_marker: The marker that indicates the start of the list
        end_marker: The marker that indicates the end of the list

    Returns:
        A list of extracted descriptions with numbering removed
    """
    # Find the section boundaries
    start_idx = completion_text.find(start_marker)
    end_idx = completion_text.find(end_marker)

    # Return empty list if markers aren't found
    if start_idx == -1 or end_idx == -1:
        return []

    # Extract the text between markers
    text = completion_text[start_idx + len(start_marker) : end_idx].strip()

    # Process each line to extract descriptions
    items: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Remove leading numbers, dots, and whitespace (e.g., "1. ", "2. ")
        line = line.lstrip("0123456789").lstrip(".").lstrip(" ")
        line = line.rstrip().rstrip(".")
        if line:
            items.append(line)

    return items


class WikipediaDatapoint(BaseModel):
    conversation: list[dict[str, str]]
    description: str  # self-described attribute in free-form text
    latent_attributes: dict[
        str, list[str]
    ]  # underlying attributes that are used to generate the description


class WikipediaAttributeDataset(Dataset[tuple[ChatInput, dict[str, list[str]]]]):
    def __init__(self, datapoints: list[WikipediaDatapoint], subject: Subject):
        self.datapoints = datapoints
        self.subject = subject

        self.pad_token_id = subject.pad_token_id

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> tuple[ChatInput, dict[str, list[str]]]:
        datapoint = self.datapoints[idx]
        ci = ChatInput(
            system_prompt=None,
            conversation=datapoint.conversation,  # type: ignore
            use_chat_format=True,
        )
        return ci, datapoint.latent_attributes

    def collate_fn(
        self, batch: list[tuple[ChatInput, dict[str, list[str]]]]
    ) -> tuple[list[ChatInput], list[dict[str, list[str]]]]:
        cis, attributes = zip(*batch)
        return cis, attributes


def get_wikipedia_dataset_by_split(
    subject: Subject,
    question_types: list[str],
    only_good=False,
    custom_path: str = None,
    seed=42,
) -> dict[str, WikipediaAttributeDataset]:
    datapoints = []
    file_path = (
        DATA_DIR / "wikipedia_datasets" / "prompts.jsonl"
        if custom_path is None
        else custom_path
    )

    with open(file_path) as f:
        for line in f:
            raw_datapoint = json.loads(line)
            if (
                len(
                    [
                        x
                        for x in raw_datapoint["latent_attributes"].keys()
                        if x not in question_types
                    ]
                )
                > 0
            ):
                continue
            if only_good and not raw_datapoint.get("top_is_label", True):
                continue
            conversation = [
                {
                    "role": "user",
                    "content": f"{raw_datapoint['description']}. {WIKIPEDIA_PROMPT_REQUEST}",
                },
            ]
            datapoints.append(
                WikipediaDatapoint(
                    conversation=conversation,
                    description=raw_datapoint["description"],
                    latent_attributes=raw_datapoint["latent_attributes"],
                )
            )

    # shuffle and split into train, valid, test
    random.seed(seed)
    random.shuffle(datapoints)
    train_datapoints = datapoints[: int(len(datapoints) * 0.8)]
    valid_datapoints = datapoints[int(len(datapoints) * 0.8) : int(len(datapoints) * 0.9)]
    test_datapoints = datapoints[int(len(datapoints) * 0.9) :]

    all_datapoints = defaultdict(list)
    all_datapoints["train"].extend(train_datapoints)
    all_datapoints["valid"].extend(valid_datapoints)
    all_datapoints["test"].extend(test_datapoints)

    return {
        split: WikipediaAttributeDataset(datapoints, subject)
        for split, datapoints in all_datapoints.items()
    }