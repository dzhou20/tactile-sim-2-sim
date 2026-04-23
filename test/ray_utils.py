"""Lightweight path utilities shared across ray test scripts.

No Isaac Sim imports here — safe to import before SimulationApp is initialized.
"""

import os
import sys

sys.dont_write_bytecode = True


def get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(get_repo_root(), path)


def get_usd_path(usd_path: str | None = None, default_relative: str = "assets/isaac_sim/ray_basic_test.usd") -> str:
    repo_root = get_repo_root()
    if not usd_path:
        return os.path.join(repo_root, default_relative)
    if os.path.isabs(usd_path):
        return usd_path
    cwd_path = os.path.abspath(usd_path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(repo_root, usd_path)
