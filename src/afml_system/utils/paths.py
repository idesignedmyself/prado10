"""
PRADO9_EVO Path Utilities

Centralized path management for all PRADO state persistence.
All paths are project-local, NOT in user home directory.

Author: PRADO9_EVO Builder
Date: 2025-01-19
"""

from pathlib import Path
import os


def get_prado_root() -> Path:
    """
    Get PRADO state root directory (project-local).

    Uses the project directory where the code is running,
    NOT the user's home directory.

    Returns:
        Path to .prado directory inside project
    """
    # Use current working directory (where user runs prado commands)
    prado_root = Path.cwd() / ".prado"

    # Ensure directory exists
    prado_root.mkdir(parents=True, exist_ok=True)

    return prado_root


def get_config_dir() -> Path:
    """
    Get configs directory for optimized hyperparameters.

    Returns:
        Path to .prado/configs/
    """
    config_dir = get_prado_root() / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_evo_dir() -> Path:
    """
    Get evo state directory for evolutionary modules.

    Stores:
    - meta_learner.pkl
    - meta_learner_metadata.json
    - population.json
    - performance_memory.pkl

    Returns:
        Path to .prado/evo/
    """
    evo_dir = get_prado_root() / "evo"
    evo_dir.mkdir(parents=True, exist_ok=True)
    return evo_dir


def get_models_dir(symbol: str = None) -> Path:
    """
    Get models directory for trained ML models.

    Args:
        symbol: Optional symbol for symbol-specific models

    Returns:
        Path to .prado/models/ or .prado/models/{symbol}/
    """
    models_dir = get_prado_root() / "models"

    if symbol:
        models_dir = models_dir / symbol.lower()

    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_live_dir() -> Path:
    """
    Get live trading state directory.

    Stores:
    - portfolio/{symbol}.json
    - logs/

    Returns:
        Path to .prado/live/
    """
    live_dir = get_prado_root() / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    return live_dir


def get_portfolio_dir() -> Path:
    """
    Get portfolio state directory.

    Returns:
        Path to .prado/live/portfolio/
    """
    portfolio_dir = get_live_dir() / "portfolio"
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    return portfolio_dir


def get_logs_dir() -> Path:
    """
    Get logs directory.

    Returns:
        Path to .prado/logs/
    """
    logs_dir = get_prado_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def migrate_from_home_dir(verbose: bool = True) -> bool:
    """
    Migrate old configs from ~/.prado to project .prado

    This is a one-time migration for backwards compatibility.

    Args:
        verbose: Print migration messages

    Returns:
        True if migration occurred, False if no migration needed
    """
    home_prado = Path.home() / ".prado"
    project_prado = get_prado_root()

    # Skip if home directory doesn't exist or project already has files
    if not home_prado.exists():
        return False

    # Check if project .prado has any significant files
    has_project_files = any(project_prado.rglob("*.pkl")) or any(project_prado.rglob("*.json")) or any(project_prado.rglob("*.yaml"))

    if has_project_files:
        if verbose:
            print(f"ℹ️  Project .prado already has files, skipping migration")
        return False

    # Perform migration
    import shutil

    if verbose:
        print(f"🔄 Migrating {home_prado} → {project_prado}")

    # Copy all subdirectories
    for item in home_prado.iterdir():
        if item.is_dir():
            dest = project_prado / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            if verbose:
                print(f"  ✓ Copied {item.name}/")
        elif item.is_file():
            shutil.copy2(item, project_prado / item.name)
            if verbose:
                print(f"  ✓ Copied {item.name}")

    if verbose:
        print(f"✅ Migration complete")
        print(f"   Old files remain in {home_prado} for safety")
        print(f"   You can delete ~/.prado manually if no longer needed")

    return True


# Auto-migrate on first import (silent)
try:
    migrate_from_home_dir(verbose=False)
except Exception:
    # Silently fail migration, don't break imports
    pass
