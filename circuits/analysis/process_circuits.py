"""
Utilities for processing a dataset of inputs into serialized CLSO circuits.
"""

import numpy as np
import pandas as pd
import torch
from circuits.core.jvp import ADAGConfig, get_all_pairs_cl_ja_effects_with_attributions
from circuits.core.utils import Edge, Node
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from user_modeling.datasets.wikipedia import get_wikipedia_dataset_by_split
from util.chat_input import ChatInput, IdsInput
from util.subject import Subject, llama31_8B_instruct_config


def prepare_ci(
    subject: Subject,
    tokenizer: PreTrainedTokenizer,
    question: str,
    seed_response: str,
    k: int,
    system_prompt: str | None = None,
    true_answers: list[str] | None = None,
    use_chat_format: bool = True,
    verbose: bool = False,
):
    """
    Prepare a single chat input.
    """
    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": question})
    ci = ChatInput(
        system_prompt=None,
        conversation=conversation,
        seed_response=(
            seed_response if seed_response is not None and len(seed_response) > 0 else None
        ),
        use_chat_format=use_chat_format,
    )

    token_ids = ci.tokenize(subject)
    if seed_response is not None and seed_response.endswith(" "):
        space_token = tokenizer.encode(" ")[1]
        token_ids = token_ids + [space_token]
    ci = IdsInput(input_ids=token_ids)

    if true_answers is not None:
        # then we create the topk using the first token of every true answer
        topk = [tokenizer.encode(answer)[1] for answer in true_answers]
    else:
        logits = subject.generate(ci, max_new_tokens=10, verbose=False).logits_BV
        topk = torch.topk(logits[0], k).indices.tolist()

    if verbose:
        print("Prepared:", question, seed_response, "->", tokenizer.decode(topk[0]))
        # subject.softmax_top_k(logits[0], k)

    return ci, topk


def prepare_cis(
    subject: Subject,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    seed_responses: list[str],
    k: int = 5,
    system_prompt: str | None = None,
    true_answers: list[str] | list[None] | None = None,
    use_chat_format: bool = True,
    verbose: bool = False,
):
    """
    Prepare a list of chat inputs.
    """
    if true_answers is None:
        true_answers = [None] * len(questions)
    res = [
        prepare_ci(subject, tokenizer, q, sr, k, system_prompt, ta, use_chat_format, verbose)
        for q, sr, ta in zip(questions, seed_responses, true_answers)
    ]
    cis = [ci[0] for ci in res]
    topks = [ci[1] for ci in res]
    max_length = max(len(ci.input_ids) for ci in cis)

    attention_masks = []
    focus_tokens = []
    for topk in topks:
        focus_tokens.append(list(topk))

    # pad on left
    starts = []
    for ci in cis:
        starts.append(max_length - len(ci.input_ids))
        attention_mask = [0] * (max_length - len(ci.input_ids)) + [
            1 for _ in range(len(ci.input_ids))
        ]
        ci.input_ids = [tokenizer.pad_token_id] * (max_length - len(ci.input_ids)) + ci.input_ids
        attention_masks.append(attention_mask)

    # keep all tokens
    keep_pos = []
    for i in range(max_length):
        keep_pos.append(i)
    if verbose:
        print(keep_pos)
        print(attention_masks)
        print(focus_tokens)
    return cis, attention_masks, focus_tokens, keep_pos, starts


def prepare_ci_with_rollout(
    subject: Subject,
    tokenizer: PreTrainedTokenizer,
    question: str,
    seed_response: str | None = None,
    max_new_tokens: int = 1,
    verbose: bool = True,
):
    """
    Prepare a single chat input.
    """
    ci = ChatInput(
        system_prompt=None,
        conversation=[{"role": "user", "content": question}],
        seed_response=seed_response,
        use_chat_format=True,
    )
    token_ids = ci.tokenize(subject)
    if seed_response is not None and seed_response.endswith(" "):
        space_token = tokenizer.encode(" ")[1]
        token_ids = token_ids + [space_token]

    # generate additional tokens
    outputs = subject.generate(ci, max_new_tokens=max_new_tokens, verbose=verbose)
    rollout_token_ids = outputs.output_ids_BT[0].tolist()[
        len(token_ids) :
    ]  # the first token generated_ids

    if len(rollout_token_ids) != max_new_tokens:
        raise ValueError(
            f"rollout token ids length {len(rollout_token_ids)} != max_new_tokens {max_new_tokens}"
        )

    if verbose:
        print("Prepared:", question, "->", tokenizer.decode(rollout_token_ids))

    ci = IdsInput(input_ids=token_ids + rollout_token_ids)

    return ci


def prepare_cis_with_rollout(
    subject: Subject,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    seed_responses: list[str] | None = None,
    max_new_tokens: int = 1,
    verbose: bool = True,
):
    """
    Prepare a list of chat inputs.
    """
    if seed_responses is None:
        seed_responses = [None] * len(questions)
    cis = [
        prepare_ci_with_rollout(subject, tokenizer, q, seed_response, max_new_tokens, verbose)
        for q, seed_response in zip(questions, seed_responses)
    ]
    max_length = max(len(ci.input_ids) for ci in cis)
    all_attention_masks = []
    all_focus_tokens = []
    all_tgt_tokens = []

    new_cis = []
    starts = []
    for ci in cis:
        starts.append(max_length - len(ci.input_ids))
        full_ids = ci.input_ids
        end_token = tokenizer.encode("<|end_header_id|>")[-1]
        positions = [i for i, t in enumerate(full_ids) if t == end_token]
        start_assistant = positions[2] + 2
        # offset by 1 because next token prediction
        tgt_tokens = [i - 1 for i in range(start_assistant, len(full_ids))]
        # could be different length
        focus_tokens = [full_ids[i + 1] for i in tgt_tokens]

        ci = IdsInput(input_ids=full_ids)
        attention_masks = [0] * (max_length - len(full_ids)) + [1 for _ in range(len(full_ids))]
        ci.input_ids = [tokenizer.pad_token_id] * (max_length - len(full_ids)) + full_ids
        offset_tgt_tokens = [p + (max_length - len(full_ids)) for p in tgt_tokens]

        all_attention_masks.append(attention_masks)
        all_focus_tokens.append(focus_tokens)
        all_tgt_tokens.append(offset_tgt_tokens)
        new_cis.append(ci)

    # keep all tokens except the last token due to offset
    keep_pos = []
    for i in range(max_length - 1):
        keep_pos.append(i)

    if verbose:
        print(keep_pos)
        print(all_attention_masks)
        print(all_focus_tokens)
        print(all_tgt_tokens)

    # the reason why we return all_tgt_tokens[0] is because tgt token positions are the same for all
    return new_cis, all_attention_masks, all_focus_tokens, all_tgt_tokens[0], keep_pos, starts


def compute_circuits(
    subject: Subject,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    config: ADAGConfig,
    seed_responses: list[str] | None = None,
    k: int = 1,
    bs: int = 4,
    max_new_tokens: int = 1,
    use_rollout: bool = False,
    system_prompt: str | None = None,
    true_answers: list[str] | None = None,
):
    """
    Compute CLSO graphs for all datapoints in a list of prompts, batched.
    """
    # set up data
    prompts = prompts if isinstance(prompts, list) else [prompts]
    if seed_responses is None:
        seed_responses = [None] * len(prompts)
    seed_responses = seed_responses if isinstance(seed_responses, list) else [seed_responses]

    # storage
    all_nodes, all_edges, all_labels, all_focus, all_starts = [], [], [], [], []
    all_cis, all_attention_masks = [], []

    for i in tqdm(range(0, len(prompts), bs), desc="Processing batches"):
        if use_rollout:
            cis, attention_masks, focus_tokens, tgt_tokens, keep_pos, starts = (
                prepare_cis_with_rollout(
                    subject,
                    tokenizer,
                    prompts[i : i + bs],
                    seed_responses[i : i + bs],
                    max_new_tokens=max_new_tokens,
                    verbose=config.verbose,
                )
            )
        else:
            cis, attention_masks, focus_tokens, keep_pos, starts = prepare_cis(
                subject,
                tokenizer,
                prompts[i : i + bs],
                seed_responses[i : i + bs],
                k=k,
                system_prompt=system_prompt,
                true_answers=true_answers,
                verbose=config.verbose,
            )
        nodes, edges = get_all_pairs_cl_ja_effects_with_attributions(
            subject=subject,
            cis=cis,
            config=config,
            attention_masks=attention_masks,
            focus_logits=focus_tokens,
            src_tokens=keep_pos,
            tgt_tokens=[max(keep_pos) for _ in range(k)] if not use_rollout else tgt_tokens,
        )
        all_nodes.append(nodes)
        all_edges.append(edges)
        all_focus.append([_ for _ in range(len(focus_tokens))])
        all_starts.append(starts)
        all_cis.extend(cis)
        all_attention_masks.extend(attention_masks)
        if config.verbose:
            print("focus_tokens:", focus_tokens)
            print("starts:", starts)

    return all_nodes, all_edges, all_labels, all_focus, all_starts, all_cis, all_attention_masks


def compute_cohens_d_loo(vals_x: list[float], all_vals: list[float]) -> float:
    # vals_y is all_vals without vals_x
    vals_y = all_vals[::]
    for val in vals_x:
        vals_y.remove(val)

    std_x = np.std(vals_x, ddof=1) if len(vals_x) > 1 else 0
    std_y = np.std(vals_y, ddof=1) if len(vals_y) > 1 else 0
    s = (
        np.sqrt(((len(vals_x) - 1) * std_x + (len(vals_y) - 1) * std_y) / (len(all_vals) - 2))
        if len(all_vals) > 2
        else 0
    )
    return (np.mean(vals_x) - np.mean(vals_y)) / s if s != 0 else 0


def convert_circuit_to_dataframes(
    nodes: list[list[Node]],
    edges: list[list[Edge]],
    labels: list[str],
    starts: list[list[int]],
    bs: int = 4,
    ignore_bos: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process CLSO graph data into a clean dataframe.
    """
    dfs_node, dfs_edge = [], []
    for batch_idx in range(len(nodes)):
        for idx in range(bs):
            start = starts[batch_idx][idx] + (1 if ignore_bos else 0)
            d = [
                (
                    node.layer,
                    node.token,
                    node.neuron,
                    node.final_attribution[idx].sum().item(),
                    node.activation[idx].item(),
                    node.attr_map[idx, start:].tolist() if node.attr_map is not None else None,
                    node.contrib_map[idx].tolist() if node.contrib_map is not None else None,
                )
                for node in nodes[batch_idx]
                if node.token >= start
            ]
            df_node = pd.DataFrame(
                d,
                columns=[
                    "layer",
                    "token",
                    "neuron",
                    "attribution",
                    "activation",
                    "attr_map",
                    "contrib_map",
                ],
            ).assign(label=labels[batch_idx * bs + idx] + f"___{batch_idx * bs + idx}")
            d = [
                (
                    f"{edge.src.layer}->{edge.tgt.layer}",
                    f"{edge.src.token}->{edge.tgt.token}",
                    f"{edge.src.neuron}->{edge.tgt.neuron}",
                    edge.final_attribution[idx].sum().item(),
                    edge.weight[idx].item(),
                )
                for edge in edges[batch_idx]
                if edge.src.token >= start and edge.tgt.token >= start
            ]
            df_edge = pd.DataFrame(
                d, columns=["layer", "token", "neuron", "attribution", "weight"]
            ).assign(label=labels[batch_idx * bs + idx] + f"___{batch_idx * bs + idx}")

            # normalize attribution by sum of goals
            total_last_layer_attribution = df_node[
                df_node.layer == df_node.layer.max()
            ].attribution.sum()
            df_node.loc[:, "attribution"] = (
                df_node.loc[:, "attribution"] / total_last_layer_attribution
            )
            df_edge.loc[:, "attribution"] = (
                df_edge.loc[:, "attribution"] / total_last_layer_attribution
            )

            # add to df
            dfs_node.append(df_node)
            dfs_edge.append(df_edge)

    # merge dfs
    df_node = pd.concat(dfs_node)
    df_edge = pd.concat(dfs_edge)
    # drop 0 attributions
    df_node = df_node[df_node.attribution != 0]
    df_edge = df_edge[df_edge.attribution != 0].dropna(subset=["attribution"])
    return df_node, df_edge


def convert_inputs_to_circuits(
    subject: Subject,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    config: ADAGConfig,
    seed_responses: list[str] | None = None,
    labels: list[str] | None = None,
    num_datapoints: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 1,
    k: int = 1,
    # TODO: topk_logits: int = 1,
    ignore_bos: bool = False,
    system_prompt: str | None = None,
    use_rollout: bool = False,
    return_cis: bool = False,
    true_answers: list[str] | None = None,
) -> (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, list[IdsInput], list[list[int]]]
):
    """
    Convert a list of prompts and seed responses into a CLSO circuit.
    """
    # num datapoints
    if num_datapoints is None:
        num_datapoints = len(prompts)
        assert len(prompts) == len(labels) == len(seed_responses)
    else:
        assert len(prompts) >= num_datapoints
        assert len(labels) >= num_datapoints
        assert len(seed_responses) >= num_datapoints

    # grab inputs
    prompts = prompts[:num_datapoints]
    labels = labels[:num_datapoints]
    if seed_responses is not None and not use_rollout:
        seed_responses = seed_responses[:num_datapoints]

    print("Prompt:", prompts[0])
    if seed_responses is not None and not use_rollout:
        print("Seed response:", seed_responses[0].replace(" ", "_"))
    print("Number of datapoints:", len(prompts))

    # compute circuits
    nodes, edges, _, focus, starts, cis, attention_masks = compute_circuits(
        subject,
        tokenizer,
        prompts,
        config=config,
        seed_responses=seed_responses,
        k=k,
        bs=batch_size,
        max_new_tokens=max_new_tokens,
        use_rollout=use_rollout,
        system_prompt=system_prompt,
        true_answers=true_answers,
    )

    # convert to dataframes
    df_node, df_edge = convert_circuit_to_dataframes(
        nodes,
        edges,
        labels,
        starts,
        bs=batch_size,
        ignore_bos=ignore_bos,
    )

    if return_cis:
        return df_node, df_edge, cis, attention_masks
    return df_node, df_edge


def example_countries(
    skip_attr_contrib: bool = False,
    return_nodes_only: bool = False,
    ignore_bos: bool = False,
    # TODO: topk_logits: int = 1,
):
    # load model
    subject = Subject(llama31_8B_instruct_config)
    tokenizer = subject.tokenizer

    # prepare inputs
    # countries = ["France"]
    prompts = [f"What is the capital of the state containing Los Angeles?"]
    seed_responses = ["Answer:"] * len(prompts)
    labels = ["Sacramento"]

    # convert to dataframes
    return convert_inputs_to_circuits(
        subject,
        tokenizer,
        prompts,
        seed_responses,
        labels,
        num_datapoints=1,
        batch_size=1,
        neurons=100,
        use_last_residual=False,
        verbose=True,
        skip_attr_contrib=skip_attr_contrib,
        return_nodes_only=return_nodes_only,
        ignore_bos=ignore_bos,
        use_rollout=False,
        # TODO: topk_logits=topk_logits,
    )


def example_wikipedia_user_modeling(
    split: str = "gender",
    num_datapoints: int = 1,
    batch_size: int = 1,
    skip_attr_contrib: bool = False,
    return_nodes_only: bool = False,
    ignore_bos: bool = False,
    neurons: int = 100,
    use_relp_grad: bool = False,
    seed=42,
    # TODO: topk_logits: int = 1,
    return_cis: bool = False,
):
    # load model
    subject = Subject(llama31_8B_instruct_config)
    tokenizer = subject.tokenizer

    # prepare inputs
    datasets = get_wikipedia_dataset_by_split(subject, question_types=[f"{split}"], only_good=True)
    datasets["train"].datapoints = (
        datasets["train"].datapoints + datasets["test"].datapoints + datasets["valid"].datapoints
    )
    prompts = [datapoint.conversation[0]["content"] for datapoint in datasets["train"].datapoints]
    labels = [datapoint.latent_attributes[split][0] for datapoint in datasets["train"].datapoints]
    seed_responses = [f"{{{{Infobox person\n| {split} ="] * len(prompts)

    # convert to dataframes
    return convert_inputs_to_circuits(
        subject,
        tokenizer,
        prompts,
        seed_responses,
        labels,
        num_datapoints=num_datapoints,
        batch_size=batch_size,
        neurons=neurons,
        use_last_residual=False,
        verbose=False,
        skip_attr_contrib=skip_attr_contrib,
        return_nodes_only=return_nodes_only,
        ignore_bos=ignore_bos,
        use_rollout=False,
        use_relp_grad=use_relp_grad,
        return_cis=return_cis,
        # TODO: topk_logits=topk_logits,
    )
