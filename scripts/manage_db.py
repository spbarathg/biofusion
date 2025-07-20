#!/usr/bin/env python3
"""
DATABASE MANAGEMENT SCRIPT
===========================

CLI tool for managing TimescaleDB schema migrations using Alembic.
Provides commands for creating, running, and managing database migrations.

Usage:
    python scripts/manage_db.py init              # Initialize migration environment
    python scripts/manage_db.py revision --autogenerate -m "message"  # Create migration
    python scripts/manage_db.py upgrade head      # Run all pending migrations
    python scripts/manage_db.py downgrade -1      # Rollback one migration
    python scripts/manage_db.py current           # Show current migration
    python scripts/manage_db.py history           # Show migration history
    python scripts/manage_db.py check             # Check database health
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext
from alembic.runtime.migration import MigrationContext

from worker_ant_v1.core.database import get_database_config, TimescaleDBManager
from worker_ant_v1.utils.logger import get_logger


class DatabaseManager:
    """Database management utility with Alembic integration"""
    
    def __init__(self):
        self.logger = get_logger("DatabaseManager")
        self.project_root = Path(__file__).parent.parent
        self.alembic_cfg_path = self.project_root / "alembic.ini"
        self.migrations_dir = self.project_root / "migrations"
        
        # Initialize Alembic config
        self.alembic_cfg = Config(str(self.alembic_cfg_path))
        
        # Set database URL
        try:
            db_config = get_database_config()
            db_url = (
                f"postgresql://{db_config.username}:{db_config.password}"
                f"@{db_config.host}:{db_config.port}/{db_config.database}"
            )
            self.alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        except Exception as e:
            self.logger.warning(f"Could not load database config: {e}")
    
    def init(self) -> bool:
        """Initialize Alembic migration environment"""
        try:
            print("🔧 Initializing Alembic migration environment...")
            
            # Check if already initialized
            if self.migrations_dir.exists() and (self.migrations_dir / "env.py").exists():
                print("✅ Migration environment already initialized")
                return True
            
            # Initialize Alembic
            command.init(self.alembic_cfg, str(self.migrations_dir))
            print("✅ Migration environment initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize migration environment: {e}")
            return False
    
    def create_revision(self, message: str, autogenerate: bool = False) -> bool:
        """Create a new migration revision"""
        try:
            print(f"📝 Creating migration: {message}")
            
            if autogenerate:
                print("⚠️  Auto-generation not fully supported with raw SQL schema")
                print("    You'll need to manually edit the generated migration file")
            
            command.revision(
                self.alembic_cfg,
                message=message,
                autogenerate=autogenerate
            )
            
            print("✅ Migration file created successfully")
            print("💡 Don't forget to edit the migration file to add your SQL commands")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create migration: {e}")
            return False
    
    def upgrade(self, revision: str = "head") -> bool:
        """Upgrade database to specified revision"""
        try:
            print(f"⬆️  Upgrading database to: {revision}")
            
            command.upgrade(self.alembic_cfg, revision)
            
            print("✅ Database upgrade completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upgrade database: {e}")
            return False
    
    def downgrade(self, revision: str) -> bool:
        """Downgrade database to specified revision"""
        try:
            print(f"⬇️  Downgrading database to: {revision}")
            
            command.downgrade(self.alembic_cfg, revision)
            
            print("✅ Database downgrade completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to downgrade database: {e}")
            return False
    
    def current(self) -> bool:
        """Show current migration revision"""
        try:
            print("📍 Current database revision:")
            command.current(self.alembic_cfg, verbose=True)
            return True
            
        except Exception as e:
            print(f"❌ Failed to get current revision: {e}")
            return False
    
    def history(self, verbose: bool = False) -> bool:
        """Show migration history"""
        try:
            print("📜 Migration history:")
            command.history(self.alembic_cfg, verbose=verbose)
            return True
            
        except Exception as e:
            print(f"❌ Failed to get migration history: {e}")
            return False
    
    def show(self, revision: str) -> bool:
        """Show details of a specific revision"""
        try:
            print(f"🔍 Details for revision: {revision}")
            command.show(self.alembic_cfg, revision)
            return True
            
        except Exception as e:
            print(f"❌ Failed to show revision details: {e}")
            return False
    
    async def check_database_health(self) -> bool:
        """Check database connectivity and health"""
        try:
            print("🏥 Checking database health...")
            
            # Test basic connectivity
            db_config = get_database_config()
            db_manager = TimescaleDBManager(db_config)
            
            # Try to initialize (this tests connectivity)
            success = await db_manager.initialize()
            await db_manager.shutdown()
            
            if success:
                print("✅ Database connectivity: OK")
                print("✅ TimescaleDB extension: Available")
                return True
            else:
                print("❌ Database connectivity: Failed")
                return False
                
        except Exception as e:
            print(f"❌ Database health check failed: {e}")
            return False
    
    def check_migration_status(self) -> bool:
        """Check migration status and pending migrations"""
        try:
            print("🔄 Checking migration status...")
            
            # Get script directory
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            
            # Get current revision
            with EnvironmentContext(
                self.alembic_cfg,
                script_dir,
            ) as env_context:
                env_context.configure(
                    url=self.alembic_cfg.get_main_option("sqlalchemy.url"),
                    target_metadata=None,
                )
                
                with env_context.begin_transaction():
                    migration_context = env_context.get_context()
                    current_rev = migration_context.get_current_revision()
            
            # Get head revision
            head_rev = script_dir.get_current_head()
            
            print(f"📍 Current revision: {current_rev or 'None'}")
            print(f"🎯 Head revision: {head_rev or 'None'}")
            
            if current_rev == head_rev:
                print("✅ Database is up to date")
            elif current_rev is None:
                print("⚠️  Database not initialized - run 'upgrade head'")
            else:
                print("⚠️  Pending migrations available - run 'upgrade head'")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to check migration status: {e}")
            return False
    
    def create_initial_migration(self) -> bool:
        """Create initial migration with current schema"""
        try:
            print("🏗️  Creating initial migration with current TimescaleDB schema...")
            
            # Create the initial migration file
            revision_result = command.revision(
                self.alembic_cfg,
                message="Initial TimescaleDB schema setup",
                autogenerate=False
            )
            
            # Get the generated file path
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            revision_file = script_dir.get_revision(revision_result.revision).path
            
            # Read the current schema from database.py and create the migration
            initial_schema = self._get_initial_schema()
            
            # Replace the migration content
            with open(revision_file, 'w') as f:
                f.write(initial_schema)
            
            print(f"✅ Initial migration created: {revision_file}")
            print("💡 Review the migration file and run 'upgrade head' to apply")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create initial migration: {e}")
            return False
    
    def _get_initial_schema(self) -> str:
        """Generate initial schema migration content"""
        return '''"""Initial TimescaleDB schema setup

Revision ID: ${revision}
Revises: 
Create Date: ${create_date}

This migration creates the initial TimescaleDB schema with all required
tables and hypertables for the Antbot trading system.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '${revision}'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial TimescaleDB schema"""
    
    # Enable TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    
    # Create trades hypertable
    op.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            timestamp TIMESTAMPTZ NOT NULL,
            trade_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            wallet_id TEXT NOT NULL,
            token_address TEXT NOT NULL,
            token_symbol TEXT,
            token_name TEXT,
            trade_type TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            amount_sol DOUBLE PRECISION,
            amount_tokens DOUBLE PRECISION,
            price DOUBLE PRECISION,
            slippage_percent DOUBLE PRECISION,
            latency_ms INTEGER,
            gas_cost_sol DOUBLE PRECISION DEFAULT 0,
            rpc_cost_sol DOUBLE PRECISION DEFAULT 0,
            api_cost_sol DOUBLE PRECISION DEFAULT 0,
            profit_loss_sol DOUBLE PRECISION,
            profit_loss_percent DOUBLE PRECISION,
            hold_time_seconds INTEGER,
            tx_signature_hash TEXT,
            retry_count INTEGER DEFAULT 0,
            exit_reason TEXT,
            error_message TEXT,
            market_cap_usd DOUBLE PRECISION,
            volume_24h_usd DOUBLE PRECISION,
            price_change_24h_percent DOUBLE PRECISION,
            metadata JSONB,
            PRIMARY KEY (timestamp, trade_id)
        );
    """)
    
    # Create hypertable for trades
    op.execute("SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);")
    
    # Create system_events hypertable
    op.execute("""
        CREATE TABLE IF NOT EXISTS system_events (
            timestamp TIMESTAMPTZ NOT NULL,
            event_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            component TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            event_data JSONB,
            session_id TEXT,
            wallet_id TEXT,
            resolved BOOLEAN DEFAULT FALSE,
            resolution_time TIMESTAMPTZ,
            PRIMARY KEY (timestamp, event_id)
        );
    """)
    
    op.execute("SELECT create_hypertable('system_events', 'timestamp', if_not_exists => TRUE);")
    
    # Create performance_metrics hypertable
    op.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            timestamp TIMESTAMPTZ NOT NULL,
            metric_name TEXT NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            unit TEXT,
            component TEXT NOT NULL,
            labels JSONB,
            aggregation_period TEXT,
            metadata JSONB,
            PRIMARY KEY (timestamp, metric_name, component)
        );
    """)
    
    op.execute("SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE);")
    
    # Create caller_profiles hypertable
    op.execute("""
        CREATE TABLE IF NOT EXISTS caller_profiles (
            timestamp TIMESTAMPTZ NOT NULL,
            caller_id TEXT NOT NULL,
            username TEXT,
            platform TEXT NOT NULL,
            account_age_days INTEGER,
            follower_count INTEGER,
            verified_account BOOLEAN,
            total_calls INTEGER,
            successful_calls INTEGER,
            success_rate DOUBLE PRECISION,
            avg_profit_percent DOUBLE PRECISION,
            trust_score DOUBLE PRECISION,
            credibility_level TEXT,
            manipulation_risk TEXT,
            risk_indicators TEXT[],
            first_seen TIMESTAMPTZ,
            last_seen TIMESTAMPTZ,
            profile_data JSONB,
            PRIMARY KEY (timestamp, caller_id)
        );
    """)
    
    op.execute("SELECT create_hypertable('caller_profiles', 'timestamp', if_not_exists => TRUE);")
    
    # Create indexes for better query performance
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_trades_token_address ON trades (token_address, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_trades_wallet_id ON trades (wallet_id, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_trades_session_id ON trades (session_id, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_trades_success ON trades (success, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_trades_type ON trades (trade_type, timestamp DESC);",
        
        "CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events (event_type, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_system_events_component ON system_events (component, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events (severity, timestamp DESC);",
        
        "CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics (metric_name, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_metrics_component ON performance_metrics (component, timestamp DESC);",
        
        "CREATE INDEX IF NOT EXISTS idx_caller_profiles_id ON caller_profiles (caller_id, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_caller_profiles_platform ON caller_profiles (platform, timestamp DESC);"
    ]
    
    for index_sql in indexes:
        op.execute(index_sql)
    
    # Set up data retention policies
    op.execute("SELECT add_retention_policy('trades', INTERVAL '1 year', if_not_exists => TRUE);")
    op.execute("SELECT add_retention_policy('system_events', INTERVAL '6 months', if_not_exists => TRUE);")
    op.execute("SELECT add_retention_policy('performance_metrics', INTERVAL '3 months', if_not_exists => TRUE);")
    op.execute("SELECT add_retention_policy('caller_profiles', INTERVAL '1 year', if_not_exists => TRUE);")


def downgrade() -> None:
    """Drop all tables and extensions"""
    
    # Drop retention policies
    op.execute("SELECT remove_retention_policy('trades', if_exists => TRUE);")
    op.execute("SELECT remove_retention_policy('system_events', if_exists => TRUE);")
    op.execute("SELECT remove_retention_policy('performance_metrics', if_exists => TRUE);")
    op.execute("SELECT remove_retention_policy('caller_profiles', if_exists => TRUE);")
    
    # Drop tables (hypertables are dropped automatically)
    op.execute("DROP TABLE IF EXISTS caller_profiles CASCADE;")
    op.execute("DROP TABLE IF EXISTS performance_metrics CASCADE;")
    op.execute("DROP TABLE IF EXISTS system_events CASCADE;")
    op.execute("DROP TABLE IF EXISTS trades CASCADE;")
    
    # Note: We don't drop the TimescaleDB extension as other databases might be using it
'''


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Database management tool for Antbot TimescaleDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_db.py init
  python scripts/manage_db.py revision --autogenerate -m "Add new column"
  python scripts/manage_db.py upgrade head
  python scripts/manage_db.py downgrade -1
  python scripts/manage_db.py current
  python scripts/manage_db.py history --verbose
  python scripts/manage_db.py check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize migration environment')
    
    # Revision command
    rev_parser = subparsers.add_parser('revision', help='Create new migration')
    rev_parser.add_argument('-m', '--message', required=True, help='Migration message')
    rev_parser.add_argument('--autogenerate', action='store_true', help='Auto-generate migration')
    
    # Upgrade command
    up_parser = subparsers.add_parser('upgrade', help='Upgrade database')
    up_parser.add_argument('revision', nargs='?', default='head', help='Target revision (default: head)')
    
    # Downgrade command
    down_parser = subparsers.add_parser('downgrade', help='Downgrade database')
    down_parser.add_argument('revision', help='Target revision (e.g., -1, base, specific revision)')
    
    # Current command
    current_parser = subparsers.add_parser('current', help='Show current revision')
    
    # History command
    hist_parser = subparsers.add_parser('history', help='Show migration history')
    hist_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show revision details')
    show_parser.add_argument('revision', help='Revision to show')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check database health and migration status')
    
    # Initial migration command
    initial_parser = subparsers.add_parser('create-initial', help='Create initial migration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    try:
        if args.command == 'init':
            success = db_manager.init()
            
        elif args.command == 'revision':
            success = db_manager.create_revision(args.message, args.autogenerate)
            
        elif args.command == 'upgrade':
            success = db_manager.upgrade(args.revision)
            
        elif args.command == 'downgrade':
            success = db_manager.downgrade(args.revision)
            
        elif args.command == 'current':
            success = db_manager.current()
            
        elif args.command == 'history':
            success = db_manager.history(args.verbose)
            
        elif args.command == 'show':
            success = db_manager.show(args.revision)
            
        elif args.command == 'check':
            # Run both database health check and migration status
            health_ok = asyncio.run(db_manager.check_database_health())
            status_ok = db_manager.check_migration_status()
            success = health_ok and status_ok
            
        elif args.command == 'create-initial':
            success = db_manager.create_initial_migration()
            
        else:
            print(f"❌ Unknown command: {args.command}")
            success = False
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 