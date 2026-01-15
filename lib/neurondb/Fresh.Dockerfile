FROM postgres:16

# Install pgvector extension
RUN apt-get update \
    && apt-get install -y postgresql-16-pgvector \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV POSTGRES_PASSWORD=sysadmin
ENV POSTGRES_USER=clarity
ENV POSTGRES_DB=neurons
ENV PGDATA=/var/lib/postgresql/data

# Expose PostgreSQL port
EXPOSE 5432

# Set the default command to run PostgreSQL
CMD ["postgres"]
