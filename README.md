# circuits

This repository hosts Transluce's circuit tracing codebase, as described in the [blogpost](https://transluce.org/neuron-circuits).

## Installation

First clone this repo:

```bash
git clone https://github.com/TransluceAI/circuits.git
```

Then install the dependencies using `uv`:

```bash
uv sync
```

You must set environment variables in order to use some parts of the codebase. Copy the template file and fill in the missing values:

```
cp .env.template .env
```

## Codebase structure

The main codebase is in the `circuits` directory. The `lib` directory contains shared libraries used by the codebase, which have been also used in prior releases by Transluce (e.g. [observatory](https://github.com/TransluceAI/observatory)).

The following subdirectories live in `circuits`:

- `clso`: Core implementation of gradient-based node attribution and edge weight tracing in our work. Also includes helpers for applying RelP and Integrated Gradients on arbitrary LMs.
- `evals`: Evaluation code for the SVA benchmark, adapted from [Marks et al. (2024)](https://github.com/saprmarks/dictionary_learning). The evaluations in our blogpost can be replicated with this code.
- `analysis`: Codebase for analysing and visualising circuits, taking in inputs from the algorithms implemented in `clso`. Includes utilities for feature scoring given hypotheses, clustering, and steering. All data is stored and processed as `pandas` dataframes.

Additionally, `scripts` includes utilities for replicating our evaluations and case studies.

## Example

For an example of how to trace and save circuits over a dataset, see the `scripts/case_studies/math/math.py` script.

## Citation

If you use this codebase in your research, please cite [our blogpost](https://transluce.org/neuron-circuits):

```
@misc{arora2025language,
  author       = {Arora, Aryaman and Wu, Zhengxuan and Steinhardt, Jacob and Schwettmann, Sarah},
  title        = {Language Model Circuits are Sparse in the Neuron Basis},
  year         = {2025},
  month        = {November},
  day          = {20},
  howpublished = {\url{https://transluce.org/neuron-circuits}}
}
```