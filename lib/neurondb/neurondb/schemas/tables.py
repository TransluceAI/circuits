from typing import Any

from neurondb.schemas.base import SQLABase
from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import mapped_column
from util.subject import Subject


def get_sqla_neuron_id(
    subject_or_model_id: Subject | str,
    layer: int,
    neuron: int,
    expert: int | None = None,
) -> str:
    if isinstance(subject_or_model_id, Subject):
        return f"{subject_or_model_id.lm_config.hf_model_id}_{layer}_{neuron}_{expert}"
    return f"{subject_or_model_id}_{layer}_{neuron}_{expert}"


class SQLALanguageModel(SQLABase):
    __tablename__ = "language_models"

    # HuggingFace ID
    id = mapped_column(String(255), nullable=False, primary_key=True)


class SQLANeuron(SQLABase):
    __tablename__ = "neurons"

    # Derived 'id' column as primary key
    id = mapped_column(String(255), primary_key=True)

    # Which model
    model_id = mapped_column(
        String(255), ForeignKey(f"{SQLALanguageModel.__tablename__}.id"), nullable=False
    )

    # Location in the model
    layer = mapped_column(Integer, nullable=False)
    neuron = mapped_column(Integer, nullable=False)
    expert = mapped_column(Integer)  # Optional; for MoE

    def __init__(self, **kwargs: Any):
        if "id" in kwargs:
            raise ValueError("'id' should not be provided. It will be automatically generated.")

        super().__init__(**kwargs)
        self.id = get_sqla_neuron_id(self.model_id, self.layer, self.neuron, self.expert)


class SQLANeuronDescription(SQLABase):
    __tablename__ = "descriptions_launch_241020"
    # __tablename__ = "descriptions_new"

    # Derived 'id' column as primary key
    id = mapped_column(String(255), primary_key=True)

    neuron_id = mapped_column(
        String(255), ForeignKey(f"{SQLANeuron.__tablename__}.id"), nullable=False
    )
    polarity = mapped_column(Integer, nullable=False)  # Max or min exemplars
    version = mapped_column(Text, nullable=True)

    description = mapped_column(Text, nullable=False)
    description_cleaned = mapped_column(Text, nullable=True)
    description_summary = mapped_column(Text, nullable=True)
    # description_embedding = mapped_column(Vector(1024), nullable=False)
    description_embedding = mapped_column(Vector(1536), nullable=False)
    description_metadata = mapped_column(JSONB)

    is_interesting = mapped_column(Boolean, nullable=True)

    def __init__(self, **kwargs: Any):
        if "id" in kwargs:
            raise ValueError("'id' should not be provided. It will be automatically generated.")

        super().__init__(**kwargs)
        self.id = f"{self.neuron_id}_{self.polarity}_{self.version}"


class SQLANeuronQuantiles(SQLABase):
    __tablename__ = "quantiles"

    # Derived 'id' column as primary key
    id = mapped_column(String(255), primary_key=True)
    neuron_id = mapped_column(
        String(255), ForeignKey(f"{SQLANeuron.__tablename__}.id"), nullable=False
    )

    quantiles = mapped_column(JSONB, nullable=False)

    def __init__(self, **kwargs: Any):
        if "id" in kwargs:
            raise ValueError("'id' should not be provided. It will be automatically generated.")

        super().__init__(**kwargs)
        self.id = self.neuron_id


class SQLANeuronExemplar(SQLABase):
    __tablename__ = "exemplars"

    # Derived 'id' column as primary key
    id = mapped_column(String(255), primary_key=True)

    neuron_id = mapped_column(
        String(255), ForeignKey(f"{SQLANeuron.__tablename__}.id"), nullable=False
    )
    polarity = mapped_column(Integer, nullable=False)  # Max or min exemplars

    text = mapped_column(Text, nullable=False)
    activation_value = mapped_column(Float, nullable=False)
    rank = mapped_column(Integer, nullable=False)  # Rank of the exemplar in a fixed dataset sample

    def __init__(self, **kwargs: Any):
        if "id" in kwargs:
            raise ValueError("'id' should not be provided. It will be automatically generated.")

        super().__init__(**kwargs)
        self.id = f"{self.neuron_id}_{self.polarity}_{self.rank}"
