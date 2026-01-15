from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Literal, Optional, Type

import numpy as np
from neurondb.postgres import DBManager, sqla_and_
from neurondb.schemas import SQLABase, SQLANeuron, SQLANeuronDescription
from pydantic import BaseModel, model_validator
from sqlalchemy import ColumnElement, Float
from util.errors import EmbeddingException
from util.openai import get_openai_client_sync, get_openai_embeddings_sync
from util.types import NDFloatArray

__all__ = [
    # Neurons
    "Neuron",
    "NeuronPolarity",
    "NeuronDescription",
    # Neuron metadata
    "NeuronGeneralMetadata",
    "AttributionResult",
    "NeuronRunMetadata",
    "NeuronsMetadataDict",
    # Filters
    "NeuronFilter",
    "NeuronDBFilter",
    "ActivationPercentileFilter",
    "AttributionFilter",
    "IdFilter",
    "TokenFilter",
    "ComplexFilter",
    # Quantile keys
    "QTILE_KEYS",
]

DB_VECTOR_SEARCH_LIMIT = 1000
DB_RETURN_LIMIT = 10_000

QTILE_KEYS_TYPE = Literal[
    "0.9999999", "1e-07", "0.999999", "1e-06", "0.99999", "1e-05", "0.9999", "0.0001"
]
FILTER_QTILE_TYPE = Literal["1e-7", "1e-6", "1e-5", "1e-4"]  # ugly; fix this
QTILE_MAP: dict[FILTER_QTILE_TYPE, tuple[QTILE_KEYS_TYPE, QTILE_KEYS_TYPE]] = {
    # (top, bottom) keys for each percentile
    "1e-7": ("0.9999999", "1e-07"),
    "1e-6": ("0.999999", "1e-06"),
    "1e-5": ("0.99999", "1e-05"),
    "1e-4": ("0.9999", "0.0001"),
}
QTILE_KEYS: list[QTILE_KEYS_TYPE] = [
    "0.9999999",
    "1e-07",
    # "0.999999",
    # "1e-06",
    "0.99999",
    "1e-05",
    "0.9999",
    "0.0001",
]


###########
# Neurons #
###########


class NeuronPolarity(str, Enum):
    POS = "1"
    NEG = "-1"


class JSONEncodable(BaseModel):
    class Config:
        json_encoders: dict[Any, Callable[[NeuronPolarity], str]] = {
            NeuronPolarity: lambda p: p.value,
        }


class Neuron(JSONEncodable):
    """
    Unique identifier for a neuron. Token axis is optional, to allow for catch-all neurons.
    """

    layer: int
    neuron: int
    token: int | None = None  # None indicates all tokens
    polarity: NeuronPolarity | None = None  # Optional

    def __hash__(self):
        return hash(
            (self.layer, self.neuron, self.token, self.polarity.value if self.polarity else None)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Neuron):
            return (
                self.layer == other.layer
                and self.neuron == other.neuron
                and self.token == other.token
                and (self.polarity.value if self.polarity is not None else None)
                == (other.polarity.value if other.polarity is not None else None)
            )
        return False


class NeuronWithMetadata(Neuron):
    description: str | None
    activation: float | None
    activation_normalized: float | None

    def __hash__(self):
        return hash(
            (
                super().__hash__(),
                self.description,
                self.activation,
                self.activation_normalized,
            )
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NeuronWithMetadata):
            return (
                super().__eq__(other)
                and self.description == other.description
                and self.activation == other.activation
                and self.activation_normalized == other.activation_normalized
            )
        return False


class NeuronDescription(BaseModel):
    text: str
    text_cleaned: str | None = None
    summary: str | None = None
    score: float | None = None
    is_interesting: bool | None = None


###################
# Neuron metadata #
###################


class NeuronGeneralMetadata(JSONEncodable):
    """
    Metadata for a neuron in general; not specific to any run.
    """

    layer: int
    neuron: int
    descriptions: dict[NeuronPolarity, NeuronDescription | None]
    activation_percentiles: dict[QTILE_KEYS_TYPE, float | None]


class AttributionResult(BaseModel):
    layer: int
    neuron: int
    src_token_idx: int
    tgt_token_idx: int
    attribution: float
    activation: Optional[float] = None
    polarity: Optional[NeuronPolarity] = None


class NeuronRunMetadata(BaseModel):
    """
    Metadata for a neuron during a specific run.
    """

    layer: int
    neuron: int
    token: int
    activation: float
    attributions: dict[int, AttributionResult] | None  # target_token_idx -> AttributionResult


class NeuronsMetadataDict(BaseModel):
    general: dict[tuple[int, int], NeuronGeneralMetadata]  # (layer, neuron) -> metadata
    run: dict[tuple[int, int, int], NeuronRunMetadata] | None = (
        None  # (layer, neuron, token) -> metadata
    )

    class Config(JSONEncodable.Config):
        json_encoders: dict[Any, Callable[[Any], str]] = {
            tuple[int, int]: lambda x: f"{x[0]},{x[1]}",
            tuple[int, int, int]: lambda x: f"{x[0]},{x[1]},{x[2]}",
        }


##################
# Neuron Filters #
##################


class NeuronFilter(BaseModel):
    """
    NeuronFilters allow for filtering neurons based on various criteria.
    """

    def contains_filter_type(self, filter_type: type) -> bool:
        """
        Check if this filter contains a filter of the specified type.
        """

        if isinstance(self, ComplexFilter):
            return any(f.contains_filter_type(filter_type) for f in self.filters)
        else:
            return isinstance(self, filter_type)

    def get_attribution_filters(self) -> list[AttributionFilter]:
        """
        Get a list of all AttributionFilters in this filter and its subfilters.
        """
        if isinstance(self, AttributionFilter):
            return [self]
        elif isinstance(self, ComplexFilter):
            return [af for f in self.filters for af in f.get_attribution_filters()]
        else:
            return []


class NeuronDBFilter(NeuronFilter):
    """
    Almost everything you'd want to filter by in the DB
    """

    concept_or_embedding: str | NDFloatArray | None = None
    keyword: str | None = None
    polarity: NeuronPolarity | None = None
    version: str | None = None
    top_k: int | None = None
    explanation_score_range: tuple[float | None, float | None] | None = None
    is_interesting: bool | None = None

    layer_range: tuple[int | None, int | None] | None = None
    neuron_range: tuple[int | None, int | None] | None = None

    timeout_ms: int | None = 10_000  # FIXME: increase

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    def validate_at_least_one_field(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Check that at least one of the required fields is set.
        """

        # Workaround to fix weird bug I don't understand
        # Pydantic will pass other objects into this validation function, if you put things in ComplexFilter
        # TODO look more into this
        if not isinstance(values, dict):  # type: ignore
            return values

        fields_to_check = [
            "concept_or_embedding",
            "keyword",
            "polarity",
            "version",
            "explanation_score_range",
            "is_interesting",
        ]

        if all(values.get(field) is None for field in fields_to_check):
            raise ValueError(
                "At least one of concept_or_embedding, keyword, polarity, version, explanation_score_range, or is_interesting must be set"
            )
        return values

    def get_matching_ids(
        self, db: DBManager, current_neurons: set[Neuron] | None = None
    ) -> set[Neuron]:
        # Initialize query elements
        entities = [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.polarity]
        joins: list[tuple[Type[SQLABase], ColumnElement[bool]]] = [
            (
                SQLANeuronDescription,
                SQLANeuronDescription.neuron_id == SQLANeuron.id,
            )
        ]
        filters: list[ColumnElement[bool]] = []
        order_by: list[ColumnElement[Any]] = []
        layer_neuron_tuples: list[tuple[int, int]] | None = None
        limit = min(self.top_k or DB_RETURN_LIMIT, DB_RETURN_LIMIT)

        if self.concept_or_embedding is not None:
            if isinstance(self.concept_or_embedding, str):
                vector_D = get_openai_embeddings_sync(
                    get_openai_client_sync(),
                    [self.concept_or_embedding],
                    model_name="text-embedding-ada-002",
                    dimensions=None,
                )[0]
                if vector_D is None:
                    raise EmbeddingException(
                        f"Failed to embed concept or embedding: {self.concept_or_embedding}"
                    )
            else:
                vector_D = self.concept_or_embedding
            entities.append(
                (1 - SQLANeuronDescription.description_embedding.cosine_distance(vector_D)).label(
                    "cosine_sim"
                ),
            )
            if limit <= DB_VECTOR_SEARCH_LIMIT:
                # Sorting by ASC distance uses the HNSW index, but limited by DB_VECTOR_SEARCH_LIMIT
                order_by.append(
                    SQLANeuronDescription.description_embedding.cosine_distance(vector_D)
                )
            else:
                print(
                    f"Limit of {limit} records exceeds HNSW index limit of {DB_VECTOR_SEARCH_LIMIT}; using parallel scan"
                )
                # Sorting by DESC similarity does not use the index, thus slow but unlimited
                order_by.append(
                    (
                        1 - SQLANeuronDescription.description_embedding.cosine_distance(vector_D)
                    ).desc()
                )

            limit = min(limit, DB_RETURN_LIMIT)

        if self.polarity is not None:
            filters.append(SQLANeuronDescription.polarity == self.polarity.value)

        if self.explanation_score_range is not None:
            if self.explanation_score_range[0] is not None:
                filters.append(
                    self.explanation_score_range[0]
                    <= SQLANeuronDescription.description_metadata["score"].astext.cast(Float)
                )
            if self.explanation_score_range[1] is not None:
                filters.append(
                    SQLANeuronDescription.description_metadata["score"].astext.cast(Float)
                    <= self.explanation_score_range[1]
                )

        if self.keyword is not None:
            filters.append(SQLANeuronDescription.description_cleaned.op("~*")(self.keyword))

        if self.is_interesting is not None:
            filters.append(SQLANeuronDescription.is_interesting == self.is_interesting)

        if self.layer_range is not None:
            if self.layer_range[0] is not None:
                filters.append(self.layer_range[0] <= SQLANeuron.layer)
            if self.layer_range[1] is not None:
                filters.append(SQLANeuron.layer <= self.layer_range[1])

        if self.neuron_range is not None:
            if self.neuron_range[0] is not None:
                filters.append(self.neuron_range[0] <= SQLANeuron.neuron)
            if self.neuron_range[1] is not None:
                filters.append(SQLANeuron.neuron <= self.neuron_range[1])

        if current_neurons is not None:
            layer_neuron_tuples = [(n.layer, n.neuron) for n in current_neurons]

        if layer_neuron_tuples is not None:
            print(f"layer_neuron_tuples len: {len(layer_neuron_tuples)}")

        neurons = db.get(
            entities=entities,
            filter=sqla_and_(*filters),
            joins=joins,
            order_by=order_by,
            limit=limit,
            set_ef_search=min(
                limit * 2, DB_VECTOR_SEARCH_LIMIT
            ),  # HNSW docs suggest 2x the number of results
            layer_neuron_tuples=layer_neuron_tuples,
            timeout_ms=self.timeout_ms,
        )

        # TODO once we fix the DB schema, change from int to str
        return set(
            [
                Neuron(
                    layer=n.layer,
                    neuron=n.neuron,
                    polarity=NeuronPolarity.POS if n.polarity == 1 else NeuronPolarity.NEG,
                )
                for n in neurons
            ]
        )


class ActivationPercentileFilter(NeuronFilter):
    """
    Filter chat conversation neurons based on their activation percentile.
    """

    percentile: FILTER_QTILE_TYPE
    direction: Literal["top", "bottom"]

    def get_matching_ids(
        self, percentiles_PLI: dict[str, NDFloatArray], acts_LIT: NDFloatArray
    ) -> set[Neuron]:
        # Description polarity depends on top or bottom percentile
        polarity = NeuronPolarity.POS if self.direction == "top" else NeuronPolarity.NEG

        # Get all percentiles
        pctile_key = QTILE_MAP[self.percentile][int(self.direction == "bottom")]
        percentiles_LI = percentiles_PLI[pctile_key]
        mask_LIT = (
            acts_LIT >= percentiles_LI[..., None]
            if self.direction == "top"
            else acts_LIT <= percentiles_LI[..., None]
        )
        indices_3N = np.nonzero(mask_LIT)
        return set(
            [Neuron(layer=l, neuron=n, token=t, polarity=polarity) for l, n, t in zip(*indices_3N)]
        )


class AttributionFilter(NeuronFilter):
    """
    Filter neurons based on their attribution.
    """

    target_token_idx: int
    top_k: int = 1000

    def get_matching_ids(
        self,
        get_attribution: Callable[[int, int | None, int | None, int], list[AttributionResult]],
        percentiles_PLI: dict[str, NDFloatArray],
        acts_LIT: NDFloatArray,
    ) -> set[Neuron]:
        attribution_results = get_attribution(self.target_token_idx, None, None, self.top_k)
        sorted_results = sorted(attribution_results, key=lambda x: abs(x.attribution), reverse=True)
        top_k_results = sorted_results[: self.top_k]

        # Set polarity to the one that's closer in activation
        def _get_polarity(result: AttributionResult) -> NeuronPolarity:
            act = acts_LIT[result.layer][result.neuron, result.src_token_idx]
            pos_pctile = percentiles_PLI["0.99999"][result.layer][result.neuron]
            neg_pctile = percentiles_PLI["1e-05"][result.layer][result.neuron]
            pos_dist = abs(act - pos_pctile)
            neg_dist = abs(act - neg_pctile)
            if pos_dist < neg_dist:
                return NeuronPolarity.POS
            else:
                return NeuronPolarity.NEG

        return set(
            [
                Neuron(
                    layer=result.layer,
                    neuron=result.neuron,
                    token=result.src_token_idx,
                    polarity=_get_polarity(result),
                )
                for result in top_k_results
            ]
        )


class IdFilter(NeuronFilter):
    ids: set[Neuron]


class TokenFilter(NeuronFilter):
    """
    Filter neurons by token.
    """

    tokens: list[int]


class ComplexFilter(NeuronFilter):
    filters: list[
        ActivationPercentileFilter
        | NeuronDBFilter
        | TokenFilter
        | ComplexFilter
        | IdFilter
        | AttributionFilter
    ]  # Must be specific for FastAPI parsing
    op: Literal["and", "or"]
