"""SQLAlchemy engine i sesije."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import BASE_DIR, settings

log = logging.getLogger("app.db")

SEED_DB = BASE_DIR / "football.db"


def _seed_sqlite() -> None:
    """Na prvom pokretanju kopira priloženu bazu na volumen (npr. /data).

    Volumen je na početku prazan; bez ovoga bi produkcija krenula s praznom
    bazom, a sve što scheduler kasnije upiše ostaje sačuvano između deployova.
    """
    url = make_url(settings.sqlalchemy_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        return
    target = Path(url.database)
    if target.exists() or not SEED_DB.exists() or target.resolve() == SEED_DB.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SEED_DB, target)
    log.info("Baza kopirana na %s", target)


_seed_sqlite()

_connect_args = {}
if settings.sqlalchemy_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.sqlalchemy_url,
    pool_pre_ping=True,
    connect_args=_connect_args,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Za skripte i pozadinske poslove."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401  (registrira tablice)

    models.Base.metadata.create_all(bind=engine)
