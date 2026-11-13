import os

from sqlalchemy import create_engine, MetaData

# Set DATABASE_URL (e.g. a Render PostgreSQL URL) to keep data across restarts.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///stocks.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
metadata = MetaData()


def table_name_for(symbol: str) -> str:
    return "px_" + "".join(ch if ch.isalnum() else "_" for ch in symbol.upper())
