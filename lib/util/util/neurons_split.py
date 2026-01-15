import numpy as np
from util.subject import Subject, get_subject_config


def get_random_neurons(
    hf_model_id: str, num_neurons: int, seed: int, blacklist: np.ndarray = None
) -> np.ndarray:
    """
    Generate an array of unique random neurons given a seed, subject, and number of neurons,
    excluding blacklisted neurons.

    Args:
    hf_model_id (str): The subject or model identifier.
    num_neurons (int): Number of neurons to generate.
    seed (int): Random seed for reproducibility.
    blacklist (np.ndarray): Array of shape (M, 2) containing blacklisted neurons to exclude.

    Returns:
    np.ndarray: An array of shape (num_neurons, 2) where each row is [layer, neuron_idx].
    """
    np.random.seed(seed)

    subject = Subject(get_subject_config(hf_model_id))
    num_layers = subject.L
    neurons_per_layer = subject.I

    neurons = []
    for layer in range(num_layers):
        neurons.append(np.stack([np.ones(neurons_per_layer) * layer, np.arange(neurons_per_layer)]))
    neurons = np.concatenate(neurons, axis=-1).T

    if blacklist is not None:
        # Convert neurons and blacklist to sets of tuples for efficient comparison
        neuron_set = set(map(tuple, neurons))
        blacklist_set = set(map(tuple, blacklist))

        # Remove blacklisted neurons
        valid_neurons = list(neuron_set - blacklist_set)

        # Ensure we have enough valid neurons
        if len(valid_neurons) < num_neurons:
            raise ValueError(
                f"Not enough valid neurons after removing blacklisted ones. Available: {len(valid_neurons)}, Requested: {num_neurons}"
            )

        neurons = np.array(sorted(valid_neurons))

    neurons = neurons[np.random.permutation(len(neurons))].astype(np.int64)
    return neurons[:num_neurons]
