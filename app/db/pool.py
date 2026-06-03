"""Thread-safe PostgreSQL connection pool (psycopg2)."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extensions import connection as PGConnection
from psycopg2.extensions import cursor as PGCursor

logger = logging.getLogger(__name__)


class PostgresPool:
    """Wrapper around ``psycopg2.pool.ThreadedConnectionPool``.

    The context managers commit on success and roll back on error, fixing the
    bug in the original code that always committed.

    Pooled connections can be silently dropped by the server (idle timeout) or
    the network/SSL layer, leaving a dead connection in the pool. Each borrowed
    connection is therefore validated with a lightweight ping before use; dead
    connections are discarded and replaced transparently rather than handed back
    to the caller.
    """

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 5) -> None:
        logger.info(
            "Initialising PostgreSQL connection pool (min=%d, max=%d)",
            minconn,
            maxconn,
        )
        self._maxconn = maxconn
        self._pool = pg_pool.ThreadedConnectionPool(
            minconn=minconn, maxconn=maxconn, dsn=dsn
        )

    @contextmanager
    def connection(self) -> Iterator[PGConnection]:
        """Yield a live connection, committing on success and rolling back on error."""
        conn = self._acquire_live_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            self._safe_rollback(conn)
            raise
        finally:
            # Discard the connection instead of pooling it if it died mid-use,
            # so a dead connection is never handed out again.
            self._pool.putconn(conn, close=bool(conn.closed))

    @contextmanager
    def cursor(self) -> Iterator[PGCursor]:
        """Yield a cursor inside a managed connection."""
        with self.connection() as conn, conn.cursor() as cur:
            yield cur

    def close(self) -> None:
        """Close all pooled connections."""
        logger.info("Closing PostgreSQL connection pool")
        self._pool.closeall()

    # ----------------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------------- #
    def _acquire_live_conn(self) -> PGConnection:
        """Borrow a connection, discarding dead ones until a live one is found."""
        last_exc: Exception | None = None
        # Bound the retries by the pool size (+1) so a fully-stale pool can be
        # cycled through once without looping forever.
        for _ in range(self._maxconn + 1):
            conn = self._pool.getconn()
            if self._ping(conn):
                return conn
            logger.warning("Discarding dead pooled connection; reconnecting")
            self._pool.putconn(conn, close=True)
        raise last_exc or psycopg2.OperationalError(
            "Could not obtain a live database connection from the pool"
        )

    @staticmethod
    def _ping(conn: PGConnection) -> bool:
        """Return True if the connection is usable (cheap server round-trip)."""
        if conn.closed:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            # The ping opens a transaction; clear it so the caller starts clean.
            conn.rollback()
            return True
        except psycopg2.Error:
            return False

    @staticmethod
    def _safe_rollback(conn: PGConnection) -> None:
        """Roll back, tolerating an already-dead connection."""
        try:
            conn.rollback()
        except psycopg2.Error:
            # Connection is already gone (e.g. SSL closed); nothing to undo.
            logger.warning("Rollback skipped; connection already closed")
