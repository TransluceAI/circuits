import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import cast

import numpy as np
import pandas as pd
from neurondb.filters import Neuron, NeuronPolarity, SQLANeuron, SQLANeuronDescription
from neurondb.postgres import DBManager, sqla_and_
from tqdm import tqdm

from env_util import ENV

# set up db
try:
    db = DBManager.get_instance("neurons_vincent")
except Exception as e:
    print(f"Error getting DBManager instance: {e}")
    db = None
half_descriptions = {}

# Load half neuron descriptions if ARTIFACTS_DIR is set
_half_neurons_path = None
if ENV.ARTIFACTS_DIR is not None:
    from pathlib import Path
    _half_neurons_path = str(Path(ENV.ARTIFACTS_DIR) / "half_neurons" / "**" / "*.json")

for file in glob.glob(_half_neurons_path) if _half_neurons_path else []:
    layer = file.split("/")[-2]
    neuron = file.split("/")[-1].split(".")[0]
    with open(file, "r") as f:
        data = json.load(f)
    neg = data["explanations"]["negative"]
    pos = data["explanations"]["positive"]
    if neg is not None:
        half_descriptions[(layer, neuron, NeuronPolarity.NEG)] = neg[0].get("explanation", "N.A.")
    if pos is not None:
        half_descriptions[(layer, neuron, NeuronPolarity.POS)] = pos[0].get("explanation", "N.A.")


def get_neuron_description_from_db(neuron: Neuron, db: DBManager) -> str | None:
    assert neuron.polarity is not None

    try:
        res = db.get(
            entities=[SQLANeuronDescription],
            joins=[(SQLANeuron, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
            filter=sqla_and_(
                SQLANeuron.neuron == neuron.neuron,
                SQLANeuron.layer == neuron.layer,
                SQLANeuronDescription.polarity == neuron.polarity,
            ),
        )
        if res and len(res) > 0:
            description = res[0][0].description
            if description and len(description.strip()) > 0:
                return description.strip()
        return None
    except Exception as e:
        print(
            f"Error fetching description for L{neuron.layer}/N{neuron.neuron}/{neuron.polarity}: {e}"
        )
        return None


def get_descriptions_for_neurons(neurons: list[Neuron], db: DBManager) -> list[str | None]:
    assert isinstance(neurons, list)
    with ThreadPoolExecutor() as executor:
        return list(executor.map(partial(get_neuron_description_from_db, db=db), neurons))


def get_descriptions(
    nodes: pd.DataFrame,
    tokenizer,
    last_layer: int,
    get_desc: bool = True,
    verbose: bool = True,
    neuron_label_cache: dict = {},
) -> tuple[pd.DataFrame, dict]:
    """Fetch neuron descriptions from the database and add them to the nodes DataFrame."""
    if not get_desc or db is None:
        nodes["description"] = nodes.apply(lambda x: "", axis=1)
        return nodes, neuron_label_cache

    unique_neurons: set[tuple[int, int, NeuronPolarity]] = set()
    neuron_label_cache.update(half_descriptions)
    total = len(nodes)
    iterator = (
        nodes.iterrows()
        if not verbose
        else tqdm(nodes.iterrows(), total=total, desc="Getting nodes to describe")
    )
    for _, row in iterator:
        layer = cast(int, row.input_variable.layer)
        neuron = cast(int, row.input_variable.neuron)
        polarity = cast(str, row.input_variable.polarity)
        if layer == -1 or layer == last_layer:
            neuron_label_cache[(int(layer), int(neuron), NeuronPolarity.POS)] = tokenizer.decode(
                [neuron]
            )
            continue
        elif isinstance(layer, str):
            continue
        if polarity == "+":
            key = (int(layer), int(neuron), NeuronPolarity.POS)
            if key not in neuron_label_cache:
                unique_neurons.add(key)
        elif polarity == "-":
            key = (int(layer), int(neuron), NeuronPolarity.NEG)
            if key not in neuron_label_cache:
                unique_neurons.add(key)
        else:
            key = (int(layer), int(neuron), NeuronPolarity.POS)
            key2 = (int(layer), int(neuron), NeuronPolarity.NEG)
            if key not in neuron_label_cache:
                unique_neurons.add(key)
            if key2 not in neuron_label_cache:
                unique_neurons.add(key2)

    # make objects
    neuron_objects = [
        Neuron(
            layer=layer,
            neuron=neuron_idx,
            polarity=polarity,
        )
        for (layer, neuron_idx, polarity) in unique_neurons
    ]
    start_time = time.time()

    # batch fetch descriptions
    _neuron_batch_size = 200
    iterator = range(0, len(neuron_objects), _neuron_batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Fetching descriptions")
    for i in iterator:
        neuron_batch = neuron_objects[i : i + _neuron_batch_size]
        descriptions = get_descriptions_for_neurons(neuron_batch, db)
        for neuron, description in zip(neuron_batch, descriptions):
            neuron_label_cache[
                (
                    neuron.layer,
                    neuron.neuron,
                    neuron.polarity,
                )
            ] = (
                description if description else "N.A."
            )

    fetch_time = time.time() - start_time
    print(f"Database fetch completed in {fetch_time:.2f} seconds")

    # add descriptions to nodes
    def get_label(row: pd.Series) -> str:
        layer = cast(int, row.input_variable.layer)
        neuron = cast(int, row.input_variable.neuron)
        polarity = cast(str, row.input_variable.polarity)
        if layer == -1 or layer == last_layer:
            return neuron_label_cache.get((layer, neuron, NeuronPolarity.POS), "?")
        if polarity == "+":
            polarity = NeuronPolarity.POS
            description = neuron_label_cache.get((layer, neuron, polarity), "?")
            return "⁺" + description
        elif polarity == "-":
            polarity = NeuronPolarity.NEG
            description = neuron_label_cache.get((layer, neuron, polarity), "?")
            return "⁻" + description
        pos_desc = neuron_label_cache.get((layer, neuron, NeuronPolarity.POS), "?")
        neg_desc = neuron_label_cache.get((layer, neuron, NeuronPolarity.NEG), "?")
        return f"⁺{pos_desc} | ⁻{neg_desc}"

    nodes["description"] = nodes.apply(get_label, axis=1)
    return nodes, neuron_label_cache


async def get_openai_embeddings_async(
    texts: list[str],
    model: str = "text-embedding-3-small",
) -> list[np.ndarray]:
    """
    Asynchronously fetch OpenAI embeddings for a list of texts.

    Parameters
    ----------
    texts : list[str]
        A list of input strings to embed.
    model : str
        The embedding model to use (default: text-embedding-3-small).

    Returns
    -------
    list[np.ndarray]
        A list of numpy arrays with embeddings, one per input string.
    """
    client = AsyncOpenAI()
    # OpenAI embeddings API can take a batch of texts at once
    response = await client.embeddings.create(model=model, input=texts)

    # Extract embeddings
    embeddings = [np.array(e.embedding, dtype=np.float32) for e in response.data]
    return embeddings
