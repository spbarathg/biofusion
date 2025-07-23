"""
ALEMBIC ENVIRONMENT FOR ANTBOT TIMESCALEDB
==========================================

Handles database migrations for TimescaleDB with proper async support
and integration with the existing database configuration system.
"""

from logging.config import fileConfig
import asyncio
import os
from pathlib import Path
import sys

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine
from alembic import context

# Add the project root to sys.path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from worker_ant_v1.core.database import get_database_config

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
# Since we're not using SQLAlchemy models but raw SQL, we'll leave this None
target_metadata = None

# Other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url() -> str:
    """Get database URL from configuration"""
    try:
        db_config = get_database_config()
        return (
            f"postgresql://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )
    except Exception:
        # Fallback to environment variables
        host = os.getenv("TIMESCALEDB_HOST", "localhost")
        port = os.getenv("TIMESCALEDB_PORT", "5432")
        database = os.getenv("TIMESCALEDB_DATABASE", "antbot_trading")
        username = os.getenv("TIMESCALEDB_USERNAME", "antbot")
        password = os.getenv("TIMESCALEDB_PASSWORD", "")
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Run migrations with the given connection"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    """In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Get database URL
    url = get_database_url()
    
    # Override the sqlalchemy.url in the config
    config.set_main_option("sqlalchemy.url", url)
    
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Check if we're in an async context
    try:
        asyncio.run(run_async_migrations())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # We're already in an event loop, create a task
            loop = asyncio.get_event_loop()
            loop.create_task(run_async_migrations())
        else:
            raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online() 