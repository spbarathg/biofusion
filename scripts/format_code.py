#!/usr/bin/env python3
"""
Code formatting script for Antbot project.

This script runs black, isort, and flake8 on the codebase to ensure
consistent formatting and style compliance.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """Run a command and return success status.

    Args:
        command: Command to run
        description: Description of what the command does

    Returns:
        True if command succeeded, False otherwise
    """
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {' '.join(command)}")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Main formatting function."""
    print("🎨 Antbot Code Formatting")
    print("=" * 40)

    # Get project root
    project_root = Path(__file__).parent.parent
    worker_ant_dir = project_root / "worker_ant_v1"
    entry_points_dir = project_root / "entry_points"

    # Check if directories exist
    if not worker_ant_dir.exists():
        print(f"❌ Worker ant directory not found: {worker_ant_dir}")
        return 1

    # Run black formatting
    black_success = run_command(
        ["black", "--line-length=88", str(worker_ant_dir), str(entry_points_dir)],
        "Running black code formatter"
    )

    # Run isort import sorting
    isort_success = run_command(
        ["isort", "--profile=black", str(worker_ant_dir), str(entry_points_dir)],
        "Running isort import sorter"
    )

    # Run flake8 linting
    flake8_success = run_command(
        ["flake8", "--max-line-length=88", "--extend-ignore=E203,W503", str(worker_ant_dir), str(entry_points_dir)],
        "Running flake8 linter"
    )

    # Summary
    print("\n📊 Formatting Summary:")
    print(f"   Black: {'✅' if black_success else '❌'}")
    print(f"   isort: {'✅' if isort_success else '❌'}")
    print(f"   flake8: {'✅' if flake8_success else '❌'}")

    if all([black_success, isort_success, flake8_success]):
        print("\n🎉 All formatting checks passed!")
        return 0
    else:
        print("\n⚠️ Some formatting checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 