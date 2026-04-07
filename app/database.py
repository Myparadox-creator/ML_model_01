"""
Database Configuration
=======================
SQLAlchemy engine, session factory, and base model.
Supports both SQLite (dev) and PostgreSQL (prod).
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import get_settings

settings = get_settings()

# ── Engine Setup ─────────────────────────────────────────────────────────────
# SQLite needs special handling for foreign keys and check_same_thread
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # Set to True for SQL query logging during debug
    pool_pre_ping=True,  # Recycle stale connections
)

# Enable foreign keys for SQLite
if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# ── Session Factory ──────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Base Model ───────────────────────────────────────────────────────────────
Base = declarative_base()


def get_db():
    """
    Dependency that provides a database session per request.
    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables():
    """Create all tables defined by ORM models. Call once at startup."""
    Base.metadata.create_all(bind=engine)
