import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def write_jsonl(data: list[dict[str, Any]], filename: str | Path) -> None:
    with open(filename, "w") as f:
        for item in tqdm(data):
            json.dump(item, f)
            f.write("\n")


def read_jsonl(filename: str | Path) -> list[dict[str, Any]]:
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]


def read_json(filename: str | Path) -> dict[str, Any]:
    with open(filename, "r") as f:
        return json.load(f)


def write_json(data: dict[str, Any], filename: str | Path) -> None:
    with open(filename, "w") as f:
        json.dump(data, f)


def read_json_or_jsonl(filename: str | Path) -> list[dict[str, Any]] | dict[str, Any]:
    filename_str = str(filename)
    if filename_str.endswith(".jsonl"):
        return read_jsonl(filename)
    elif filename_str.endswith(".json"):
        return read_json(filename)
    else:
        raise ValueError(f"Unsupported file extension: {filename}")
