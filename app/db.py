from sqlalchemy import create_engine, MetaData

DATABASE_URL = "sqlite:///stocks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
metadata = MetaData()


def ensure_table(table_name):
    # Placeholder: pandas.to_sql will create the table if needed.
    return True
