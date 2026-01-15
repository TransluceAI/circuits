# NeuronDB

NeuronDB is a library for analyzing and filtering neurons in language models. It provides tools to:

- Populate and query databases of neuron descriptions
- Attribute model outputs to neurons
- Filter neurons based on various criteria

## Core Components

### Schemas for storing neuron descriptions

We provide SQLAlchemy models for the following tables in [`neurondb/schemas/tables.py`](neurondb/schemas/tables.py), which comprise our neuron description database:

- `SQLALanguageModel`: LLM model information
- `SQLANeuron`: Neurons and their metadata
- `SQLANeuronDescription`: Descriptions of neuron behavior at top/bottom quantiles, with vector embeddings for search
- `SQLANeuronQuantiles`: Activation statistics
- `SQLANeuronExemplar`: Example inputs that activate neurons

The vector embedding column in `SQLANeuronDescription` has an [HNSW search index](https://arxiv.org/abs/1603.09320) defined in [`neurondb/schemas/indices.py`](neurondb/schemas/indices.py).

### PostgreSQL database management

[`DBManager`](neurondb/postgres.py) provides a thread-safe PostgreSQL interface with methods to insert, update, upsert, and query data. It also supports vector similarity search. When instantiated, it creates the tables and indices defined above if they don't already exist.

See the [Installing PostgreSQL](#installing-postgresql) section below for instructions on how to set up a local Postgres instance.

### Neuron views

[`NeuronView`](neurondb/view.py) provides a stateful interface for analyzing and filtering neurons. As a chat conversation progresses, it continually updates a view on neurons and allows for filtering based on, e.g., token indices, activation quantiles, attribution scores, and database descriptions. See [`neurondb/filters.py`](neurondb/filters.py) for a complete list of supported filters.

### Loading neuron descriptions from the description generation pipeline

> This is coming soon!

## Installing PostgreSQL

### From scratch

NeuronDB requires a Postgres database with the [`pgvector` extension](https://github.com/pgvector/pgvector). We've provided a Dockerfile with the appropriate setup in [`Fresh.Dockerfile`](Fresh.Dockerfile). You can pull the precompiled image and run it with:

```bash
docker pull mengkluce/neurondb-fresh:latest
docker run -d -p 5432:5432 mengkluce/neurondb-fresh
```

You can then set the appropriate environment variables in `.env` for the [`DBManager`](neurondb/postgres.py) to connect to the database:

```
PG_USER=clarity
PG_PASSWORD=sysadmin
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=neurons
```

### Using our preloaded Docker image

We offer a preloaded Docker image that starts a local Postgres instance with our Llama-3.1 8B Instruct neuron descriptions. The Dockerfile is at [`FromBackup.Dockerfile`](FromBackup.Dockerfile). Pull and run the image with:

```
docker pull mengkluce/neurondb-llama-3.1-8b-instruct:latest
docker run -d -p 5432:5432 mengkluce/neurondb-llama-3.1-8b-instruct
```

You must similarly set the appropriate environment variables in `.env` for the [`DBManager`](neurondb/postgres.py) to connect to the database:

```
PG_USER=clarity
PG_PASSWORD=sysadmin
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=neurons
```

You can verify that the container is running correctly with `psql`:

```
PG_PASSWORD=sysadmin psql -h localhost -U clarity -d neurons
```
