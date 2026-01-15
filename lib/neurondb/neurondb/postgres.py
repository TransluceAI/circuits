from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
from uuid import uuid4

from env_util import ENV
from neurondb.schemas import DB_INDICES, SQLABase, SQLANeuron
from sqlalchemy import (
    Column,
    Engine,
    Integer,
    Table,
    and_,
    create_engine,
    desc,
    inspect,
    or_,
    select,
    text,
    tuple_,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable
from util.errors import DBTimeoutException

# Convenient alias for sqlalchemy functions
sqla_and_, sqla_or_, sqla_desc = and_, or_, desc


class DBManager:
    creation_lock = Lock()
    instances: Dict[str, "DBManager"] = {}

    def __new__(cls, engine: Engine):
        raise Exception("Use DBManager.get_instance() instead of DBManager()")

    def __deepcopy__(self, _):
        """Prevent deepcopying of singleton and return the original"""
        return self

    @classmethod
    def get_instance(cls, db_name: str | None = None):
        if db_name is None:
            db_name = ENV.PG_DATABASE

        with cls.creation_lock:
            if db_name not in cls.instances:
                instance = super().__new__(cls)
                try:
                    user, pw, host, port = (
                        ENV.PG_USER,
                        ENV.PG_PASSWORD,
                        ENV.PG_HOST,
                        ENV.PG_PORT,
                    )
                    if (
                        user is None
                        or pw is None
                        or host is None
                        or port is None
                        or db_name is None
                    ):
                        raise Exception("Missing PostgreSQL credentials; check your .env")
                    instance.__init__(
                        create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db_name}")
                    )
                    cls.instances[db_name] = instance
                except Exception as e:
                    raise e

            return cls.instances[db_name]

    def __init__(self, engine: Engine):
        self._engine = engine

        # Install pgvector extension if not exists
        with Session(self._engine) as sess:
            sess.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            sess.commit()

        # Create tables
        SQLABase.metadata.create_all(self._engine)

        # Create DB_INDICES if any don't exist
        inspector = inspect(self._engine)
        for index in DB_INDICES:
            existing_index_names = [
                existing_index["name"] for existing_index in inspector.get_indexes(index.table.name)  # type: ignore
            ]
            if index.name not in existing_index_names:
                print(f"Creating index {index.name} for table {index.table.name}")  # type: ignore
                index.create(self._engine)

    #########################
    # Generic query methods #
    #########################

    def get(
        self,
        entities: Sequence[Any],
        joins: Optional[List[Tuple[Any, Any]]] = None,
        filter: Optional[Any] = None,
        order_by: Optional[Sequence[Any]] = None,
        group_by: Optional[Sequence[Any]] = None,
        limit: Optional[int] = None,
        layer_neuron_tuples: Optional[List[Tuple[int, int]]] = None,
        set_ef_search: Optional[int] = None,
        timeout_ms: Optional[int] = 30_000,
    ) -> Sequence[Any]:
        """
        Example usage:

        ```python
        neurons = db.get(
            SQLANeuronDescription,
            filter=and_(SQLANeuron.layer == 5, SQLANeuron.neuron == 100),
            joins=[(SQLANeuron, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
            limit=10,
            order_by=[SQLANeuron.layer.asc(), SQLANeuron.neuron.asc()],
            group_by=[SQLANeuron.layer],
            layer_neuron_tuples=[(5, 100), (6, 200)]
        )
        ```
        """

        with Session(self._engine) as sess:
            # Add timeout setting at the start of the session
            if timeout_ms is not None:
                sess.execute(
                    text(f"SET LOCAL statement_timeout = :timeout_ms"), {"timeout_ms": timeout_ms}
                )

            stmt = select(*entities)

            if joins:
                for related_class, join_condition in joins:
                    stmt = stmt.join(related_class, join_condition)

            # Filter by layer-neuron tuples by JOINing on a temp table
            if layer_neuron_tuples is not None:
                temp_table = Table(
                    f"_temp_neurons_{uuid4()}",
                    SQLABase.metadata,
                    Column("layer", Integer),
                    Column("neuron", Integer),
                )
                # This table will never persist because we never commit
                create_table_sql = str(CreateTable(temp_table).compile(self._engine)).rstrip(";")
                sess.execute(text(create_table_sql))

                # Insert the layer-neuron tuples into the temporary table
                sess.execute(
                    temp_table.insert(),
                    [{"layer": l, "neuron": n} for l, n in layer_neuron_tuples],
                )

                # Ensure that the query is filtered by the temporary table
                condition = tuple_(SQLANeuron.layer, SQLANeuron.neuron).in_(
                    select(temp_table.c.layer, temp_table.c.neuron)
                )
                stmt = stmt.where(condition)

            if filter is not None:
                stmt = stmt.where(filter)
            if group_by is not None:
                stmt = stmt.group_by(*group_by)
            if order_by is not None:
                stmt = stmt.order_by(*order_by)
            if limit is not None:
                stmt = stmt.limit(limit)

            if set_ef_search is not None:
                sess.execute(
                    text("SET LOCAL hnsw.ef_search TO :ef_search"),
                    {"ef_search": set_ef_search},
                )

            try:
                result = sess.execute(stmt)
                return result.all()
            except OperationalError as e:
                if "due to statement timeout" in str(e):
                    raise DBTimeoutException("Query execution timed out") from e
                raise e  # Re-raise if it's a different OperationalError

    def insert(self, objs: List[SQLABase]) -> None:
        with Session(self._engine) as sess:
            sess.add_all(objs)
            sess.commit()

    def upsert_many(self, objs: List[SQLABase]) -> None:
        if not objs:
            return

        with Session(self._engine) as sess:
            table = objs[0].__table__
            stmt = insert(table).values(  # type: ignore
                [{c.key: getattr(obj, c.key) for c in table.columns} for obj in objs]
            )
            stmt = stmt.on_conflict_do_nothing()
            sess.execute(stmt)
            sess.commit()

    def upsert_one(self, obj: SQLABase) -> None:
        with Session(self._engine) as sess:
            sess.merge(obj)
            sess.commit()

    def clear_table(self, sqla_class: Type[SQLABase]) -> None:
        with Session(self._engine) as sess:
            sess.execute(text(f"DELETE FROM {sqla_class.__tablename__}"))
            sess.commit()

    def drop_table(self, sqla_class: Type[SQLABase]) -> None:
        with Session(self._engine) as sess:
            sess.execute(text(f"DROP TABLE IF EXISTS {sqla_class.__tablename__}"))
            sess.commit()

    def bulk_update_mappings(
        self, sqla_class: Type[SQLABase], mappings: List[Dict[str, Any]]
    ) -> None:
        with Session(self._engine) as sess:
            sess.bulk_update_mappings(sqla_class, mappings)  # type: ignore
            sess.commit()
