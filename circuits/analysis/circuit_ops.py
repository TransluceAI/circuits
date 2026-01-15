import asyncio
import json
import pickle
import random
from pathlib import Path
from typing import Any, List, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from circuits.analysis.cluster import (
    NeuronId,
    build_circuit_visualization,
    export_neuron_graph,
    plot_neurons_umap,
)
from circuits.analysis.process_circuits import convert_inputs_to_circuits
from circuits.analysis.score_features import cluster_with_hypotheses as cluster_with_hypotheses_impl
from circuits.analysis.score_features import (
    export_hypothesis_score_jsons as export_hypothesis_score_jsons_impl,
)
from circuits.analysis.score_features import get_df_node_summed as score_get_df_node_summed
from circuits.analysis.score_features import score_features as score_features_impl
from circuits.analysis.score_features import (
    score_features_multiclass as score_features_multiclass_impl,
)
from circuits.analysis.steer import export_cluster_data_to_json, get_cluster_steering_effects
from circuits.core.jvp import ADAGConfig
from util.subject import Subject

INDEX_COLS = [
    "input_variable",
    "layer",
    "token",
    "neuron",
    "description",
    "embedding",
    "cluster",
    "Average",
]


class Circuit:
    def __init__(
        self,
        subject: Subject | None,
        df_node: pd.DataFrame,
        df_edge: pd.DataFrame,
        cis: list,
        attention_masks: list[torch.Tensor],
        labels: list[str],
    ):
        self.subject = subject
        if subject is not None:
            self.tokenizer = subject.tokenizer
        self.df_node = df_node
        self.df_edge = df_edge
        self.cis = cis
        self.attention_masks = attention_masks
        self.labels = labels
        self.neuron_label_cache = {}
        self._df_node_summed_cache = None
        self._manual_clusters_map: dict[NeuronId, str] = {}
        self._manual_cluster_labels: list[str] = []

        # steering results
        self.diversity_stats = pd.DataFrame()
        self.cluster_to_cluster = pd.DataFrame()
        self.cluster_to_output = pd.DataFrame()

        # feature scoring results
        self.hypothesis_to_scores: dict[str, pd.DataFrame] = {}

    @staticmethod
    def from_dataset(
        subject: Subject,
        prompts: list[str],
        seed_responses: list[str],
        labels: list[str],
        batch_size: int = 1,
        neurons: int | None = 200,
        k: int = 5,
        use_last_residual: bool = False,
        verbose: bool = False,
        skip_attr_contrib: bool = False,
        return_nodes_only: bool = False,
        ignore_bos: bool = True,
        use_rollout: bool = False,
        use_relp_grad: bool = True,
        use_shapley_grad: bool | None = None,
        disable_half_rule: bool = False,
        disable_stop_grad: bool = False,
        use_stop_grad_on_mlps: bool = True,
        center_logits: bool = False,
        ig_steps: int | None = None,
        ig_mode: Literal["ig-inputs", "conductance"] = "ig-inputs",
        return_only_important_neurons: bool = False,
        percentage_threshold: float | None = None,
        apply_blacklist: bool = False,
        system_prompt: str | None = None,
    ):
        """
        Create a circuit from a dataset.
        """
        tokenizer = subject.tokenizer
        config = ADAGConfig(
            device="cuda",
            verbose=verbose,
            parent_threshold=None,
            edge_threshold=0.01,
            node_attribution_threshold=None,
            topk=None,
            batch_aggregation="any",
            topk_neurons=neurons,
            percentage_threshold=percentage_threshold,
            use_relp_grad=use_relp_grad,
            use_shapley_grad=not use_relp_grad if use_shapley_grad is None else use_shapley_grad,
            disable_half_rule=disable_half_rule,
            disable_stop_grad=disable_stop_grad,
            ablation_mode="zero",
            use_stop_grad_on_mlps=use_stop_grad_on_mlps,
            return_nodes_only=return_nodes_only,
            focus_last_residual=use_last_residual,
            skip_attr_contrib=skip_attr_contrib,
            center_logits=center_logits,
            ig_steps=ig_steps,
            ig_mode=ig_mode,
            return_only_important_neurons=return_only_important_neurons,
            apply_blacklist=apply_blacklist,
        )
        df_node, df_edge, cis, attention_masks = convert_inputs_to_circuits(
            subject,
            tokenizer,
            prompts,
            config=config,
            seed_responses=seed_responses,
            labels=labels,
            num_datapoints=len(prompts),
            batch_size=batch_size,
            k=k,
            return_cis=True,
            system_prompt=system_prompt,
            ignore_bos=ignore_bos,
            use_rollout=use_rollout,
        )
        return Circuit(subject, df_node, df_edge, cis, attention_masks, labels)

    #########################################################
    # CLUSTERING
    #########################################################

    def cluster(
        self,
        sum_over_tokens: bool = False,
        get_desc: bool = True,
        include_attr_contrib: bool = True,
        manual_clusters: Mapping["NeuronId", str] | None = None,
        verbose: bool = False,
    ):
        """
        Cluster the circuit by computing embeddings and clustering the neurons. Persists the clustered
        dataframe and labels in the circuit object. If `manual_clusters` is supplied, those cluster
        ids are used directly instead of recomputing embeddings/clusters.
        """
        (
            self.data_circuit,
            self.df_node_embedded,
            self.df_edge_embedded,
            self.neuron_label_cache,
        ) = build_circuit_visualization(
            self.df_node,
            self.df_edge,
            self.tokenizer,
            sum_over_tokens=sum_over_tokens,
            get_desc=get_desc,
            include_attr_contrib=include_attr_contrib,
            neuron_label_cache=self.neuron_label_cache,
            manual_clusters=manual_clusters,
            verbose=verbose,
            last_layer=self.subject.L,
        )

    #########################################################
    # STEERING
    #########################################################

    def steer(
        self,
        multiplier: float = 0.0,
        verbose: bool = False,
        record: bool = False,
        store_results: bool = False,
        complement: bool = False,
        custom_neuron_ids: list[tuple[int, int, int]] | None = None,
    ):
        """Steer the circuit by computing steering effects for each cluster."""
        diversity_stats, cluster_to_cluster, cluster_to_output = get_cluster_steering_effects(
            subject=self.subject,
            df_node=self.df_node,
            df_edge=self.df_edge,
            df_node_embedded=self.df_node_embedded,
            df_edge_embedded=self.df_edge_embedded,
            cis=self.cis,
            attention_masks=self.attention_masks,
            labels=self.labels,
            multiplier=multiplier,
            verbose=verbose,
            record=record,
            complement=complement,
            custom_neuron_ids=custom_neuron_ids,
        )
        if store_results:
            self.diversity_stats = pd.concat(
                [self.diversity_stats, diversity_stats.assign(multiplier=multiplier)]
            )
            self.cluster_to_cluster = pd.concat(
                [
                    self.cluster_to_cluster,
                    cluster_to_cluster.assign(multiplier=multiplier),
                ]
            )
            self.cluster_to_output = pd.concat(
                [
                    self.cluster_to_output,
                    cluster_to_output.assign(multiplier=multiplier),
                ]
            )
        return diversity_stats, cluster_to_cluster, cluster_to_output

    def clear_steering_results(self):
        """Clear the steering results."""
        self.diversity_stats = pd.DataFrame()
        self.cluster_to_cluster = pd.DataFrame()
        self.cluster_to_output = pd.DataFrame()
        self._df_node_summed_cache = None

    #########################################################
    # SCORING
    #########################################################

    def _get_df_node_summed(self, verbose: bool = False) -> pd.DataFrame:
        """Aggregate node-level attributions by input variable and label (cached)."""
        return score_get_df_node_summed(self, verbose=verbose)

    def score_features(
        self,
        example_labels: Sequence[bool],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Compute ROC-AUC scores for aggregated neuron features."""
        return score_features_impl(self, example_labels, verbose=verbose)

    def score_features_multiclass(
        self,
        example_labels: Sequence[Any],
        hypothesis_name: str | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Per-class ROC-AUC scores for multiclass labels."""
        if not hasattr(self, "hypothesis_to_scores"):
            self.hypothesis_to_scores: dict[str, pd.DataFrame] = {}
        if hypothesis_name is not None and hypothesis_name in self.hypothesis_to_scores:
            return self.hypothesis_to_scores[hypothesis_name]
        scores = score_features_multiclass_impl(self, example_labels, verbose=verbose)
        if hypothesis_name is not None:
            self.hypothesis_to_scores[hypothesis_name] = scores
        return scores

    def cluster_with_hypotheses(
        self,
        hypotheses: Mapping[str, Sequence[Any]],
        above_threshold: float = 0.8,
        below_threshold: float = 0.2,
        in_class_attribution_threshold: float | None = None,
        unique_only: bool = False,
        subset_labels: Sequence[str] | None = None,
        cluster_kwargs: Mapping[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, int]:
        """Assign neurons to hypothesis-defined clusters."""
        return asyncio.run(
            cluster_with_hypotheses_impl(
                self,
                hypotheses=hypotheses,
                above_threshold=above_threshold,
                below_threshold=below_threshold,
                in_class_attribution_threshold=in_class_attribution_threshold,
                unique_only=unique_only,
                subset_labels=subset_labels,
                cluster_kwargs=cluster_kwargs,
                max_layer=self.subject.L,
                verbose=verbose,
            )
        )

    def export_hypothesis_score_jsons(
        self,
        hypotheses: Mapping[str, Sequence[Any]],
        output_dir: Path,
        auc_threshold_high: float = 0.8,
        auc_threshold_low: float = 0.2,
        in_class_attribution_threshold: float | None = None,
    ) -> None:
        export_hypothesis_score_jsons_impl(
            self,
            hypotheses,
            output_dir,
            auc_threshold_high,
            auc_threshold_low,
            in_class_attribution_threshold,
        )

    #########################################################
    # VISUALIZATION
    #########################################################

    def clear_label_cache(self):
        """
        Clear the label cache.
        """
        self.neuron_label_cache = {}

    @staticmethod
    def visualize_circuit(circuits: List["Circuit"]):
        """
        Visualize the circuits.
        """
        raise NotImplementedError("This method needs to be implemented")

    def visualize_circuit_umap(self):
        """
        Visualize the circuit using UMAP.
        """
        plot_neurons_umap(self.df_node_embedded)

    def export_to_html(self, out_html: str):
        """
        Export the circuit to an HTML file.
        """
        export_neuron_graph(self.data_circuit, out_html)

    def export_for_website(self, out_json: str):
        """
        Export the circuit for the website.
        """
        assert self.data_circuit is not None, "No circuit data to export"
        with open(out_json, "w") as f:
            json.dump(self.data_circuit, f)
            print(f"Exported {len(self.data_circuit)} clusters to {out_json}")
            print(f"File size: {len(json.dumps(self.data_circuit)) / 1024:.1f} KB")

    def __getstate__(self):
        """
        Custom pickle state that excludes unpickleable objects.
        """
        state = self.__dict__.copy()
        # Remove unpickleable objects that may contain weak references
        unpickleable_attrs = [
            "subject",  # Contains LanguageModel with weak references
            "tokenizer",  # May contain weak references
            "data_circuit",  # May contain complex objects from clustering
            # 'df_node_embedded',  # May contain complex objects from clustering
            # 'df_edge_embedded',  # May contain complex objects from clustering
        ]

        for attr in unpickleable_attrs:
            state.pop(attr, None)

        return state

    def __setstate__(self, state):
        """
        Custom unpickle state restoration.
        Note: excluded attributes will need to be set manually after loading.
        """
        self.__dict__.update(state)
        # These will be None after loading and need to be set manually
        self.subject = None
        self.tokenizer = None
        self.data_circuit = None
        # self.df_node_embedded = None
        # self.df_edge_embedded = None

    def save_to_pickle(self, out_pickle: str):
        """
        Save the circuit to a pickle file.
        Note: subject, tokenizer, and clustering results are excluded and will need to be restored after loading.
        """
        with open(out_pickle, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved circuit to {out_pickle}")
        print(
            "Note: subject, tokenizer, and clustering results are not saved and will need to be restored after loading."
        )

    @classmethod
    def load_from_pickle(cls, in_pickle: str | Path):
        """
        Load the circuit from a pickle file.
        Note: You'll need to call set_subject() and re-run clustering after loading.
        """
        with open(in_pickle, "rb") as f:
            circuit = pickle.load(f)
        print(f"Loaded circuit from {in_pickle}")
        print("Remember to call circuit.set_subject(subject) and re-run clustering if needed.")
        return circuit

    def set_subject(self, subject: Subject):
        """
        Set the subject and tokenizer after loading from pickle.
        """
        self.subject = subject
        self.tokenizer = subject.tokenizer

    def save_to_json(self, out_json: str | Path):
        """
        Save the circuit to a JSON file.
        """
        with open(out_json, "w") as f:
            json.dump(
                {
                    "nodes": self.df_node.to_dict(orient="records"),
                    "edges": self.df_edge.to_dict(orient="records"),
                    "cis": self.cis,
                    "attention_masks": self.attention_masks,
                    "labels": self.labels,
                },
                f,
            )
        print(f"Saved circuit to {out_json}")

    @classmethod
    def load_from_json(cls, subject: Subject, in_json: str | Path):
        """
        Load the circuit from a JSON file.
        """
        with open(in_json, "r") as f:
            data = json.load(f)
        return cls(
            subject,
            data["nodes"],
            data["edges"],
            data["cis"],
            data["attention_masks"],
            data["labels"],
        )

    def save_cluster_to_output_to_json(self, out_json: str | Path):
        """
        Save the cluster data to a JSON file.
        """
        export_cluster_data_to_json(self.cluster_to_output, out_json)
