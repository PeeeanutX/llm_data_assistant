"""Shared pytest fixtures.

The DB-backed fixtures rely on ``pytest-postgresql``, which boots a throwaway
PostgreSQL cluster per session. Those tests are marked ``db`` and skipped
automatically if the PostgreSQL server binaries are not available.
"""

from __future__ import annotations

import shutil

import pytest

# Skip the whole DB-fixture machinery gracefully when Postgres isn't installed.
_HAVE_PG_BINARIES = shutil.which("pg_ctl") is not None

if _HAVE_PG_BINARIES:
    from pytest_postgresql import factories

    # An ephemeral, process-scoped PostgreSQL instance.
    postgresql_proc = factories.postgresql_proc()
    postgresql = factories.postgresql("postgresql_proc")


def _dsn_from_connection(connection) -> str:
    info = connection.info
    password = info.password or ""
    auth = f"{info.user}:{password}@" if password else f"{info.user}@"
    return f"postgresql://{auth}{info.host}:{info.port}/{info.dbname}"


@pytest.fixture
def pg_dsn(request):
    """Return a DSN to a migrated, empty interactions database."""
    if not _HAVE_PG_BINARIES:
        pytest.skip("PostgreSQL server binaries not available")

    connection = request.getfixturevalue("postgresql")
    dsn = _dsn_from_connection(connection)

    # Stand up the schema using the same idempotent reconciliation the app
    # uses on startup -- no Alembic needed for a PoC.
    from app.db.pool import PostgresPool
    from app.db.schema import initialize_schema

    pool = PostgresPool(dsn, minconn=1, maxconn=2)
    try:
        initialize_schema(pool, "interactions_experiment")
    finally:
        pool.close()
    return dsn


@pytest.fixture
def repository(pg_dsn):
    """An ``InteractionRepository`` bound to the ephemeral database."""
    from app.db.pool import PostgresPool
    from app.db.repository import InteractionRepository

    pool = PostgresPool(pg_dsn, minconn=1, maxconn=2)
    try:
        yield InteractionRepository(pool, "interactions_experiment")
    finally:
        pool.close()
