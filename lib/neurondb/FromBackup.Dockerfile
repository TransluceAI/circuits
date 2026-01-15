FROM postgres:16

# Install pgvector extension
RUN apt-get update \
    && apt-get install -y postgresql-16-pgvector \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the backup file into a temporary location
COPY backup.sql /tmp/backup.sql

# Set environment variables
ENV POSTGRES_PASSWORD=sysadmin
ENV POSTGRES_USER=clarity
ENV POSTGRES_DB=neurons
ENV PGDATA=/var/lib/postgresql/data

# Switch to the postgres user
USER postgres

# Initialize the database
RUN initdb --username=postgres

# Start PostgreSQL temporarily to perform setup
RUN pg_ctl -D "$PGDATA" -o "-c listen_addresses=''" -w start \
    && psql --username=postgres --command "CREATE USER clarity WITH SUPERUSER PASSWORD 'sysadmin';" \
    && psql --username=postgres --command "CREATE DATABASE neurons OWNER clarity;" \
    && psql --username=clarity --dbname=neurons --command "CREATE EXTENSION IF NOT EXISTS vector;" \
    && psql --username=clarity --dbname=neurons < /tmp/backup.sql \
    && pg_ctl -D "$PGDATA" -m fast -w stop

# Clear the backup file
USER root
RUN rm /tmp/backup.sql

# Configure pg_hba.conf to allow connections from Docker containers
RUN echo "host all all all scram-sha-256" >> "$PGDATA/pg_hba.conf"

# Expose PostgreSQL port
EXPOSE 5432

# Set the default command to run PostgreSQL
CMD ["postgres"]
