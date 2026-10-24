"""CPU thread count for torch in the containers that embed text.

Unset, ``OMP_NUM_THREADS`` lets torch start one intra-op thread per visible core,
and a container sees every core on its host (CPU quotas such as Docker ``--cpus``
or Kubernetes CPU limits do not change that count). On a shared host each small
embedding call then waits on its slowest thread: ~3 s instead of ~16 ms per
MiniLM call on a busy 32-core machine. The compose file and the Helm chart set the
variable for the data-manager and chatbot from ``data_manager.torch_threads``.
"""

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dotenv import dotenv_values

DEFAULT_TORCH_THREADS = 8

# Torch sizes its intra-op pool from MKL_NUM_THREADS ahead of OMP_NUM_THREADS, so
# compose sets both to the same value, and an env file that sets either is kept.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")


def configured_torch_threads(config: Optional[Mapping[str, Any]]) -> Optional[int]:
    """Return ``data_manager.torch_threads``, or None when it is not set.

    Raises ValueError for anything but a positive integer, so a bad value fails
    ``archi create`` or ``archi restart`` before either writes a file.
    """
    data_manager = (config or {}).get("data_manager") or {}
    value = data_manager.get("torch_threads")
    if value is None:
        return None
    # bool is an int subclass; True must not pass as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"data_manager.torch_threads must be a positive integer, got {value!r}"
        )
    return value


def visible_cpu_count() -> Optional[int]:
    """CPUs this process may run on: its affinity mask where the OS has one."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def deployment_env_values(base_dir: Path) -> Dict[str, Optional[str]]:
    """The values in the deployment's ``.env``, which every service reads as env_file."""
    env_path = Path(base_dir) / ".env"
    return dict(dotenv_values(env_path)) if env_path.is_file() else {}


def resolve_torch_threads(
    config: Optional[Mapping[str, Any]],
    env_values: Optional[Mapping[str, Optional[str]]] = None,
) -> Optional[int]:
    """Return the thread count compose sets, or None to set nothing.

    An explicit ``data_manager.torch_threads`` wins. Without it, a
    ``THREAD_ENV_VARS`` value the env file already sets is kept: compose's
    ``environment`` entry would take precedence over ``env_file``, so none is
    rendered. Otherwise the default, capped at the visible CPUs so a small host
    is not oversubscribed.
    """
    configured = configured_torch_threads(config)
    if configured is not None:
        return configured
    env_values = env_values or {}
    if any((env_values.get(name) or "").strip() for name in THREAD_ENV_VARS):
        return None
    cpus = visible_cpu_count()
    return min(DEFAULT_TORCH_THREADS, cpus) if cpus else DEFAULT_TORCH_THREADS


def chart_torch_threads(config: Optional[Mapping[str, Any]]) -> int:
    """Return the thread count the Helm chart sets: the explicit value, or 8.

    The pods run on cluster nodes, not on the machine running ``archi create``,
    so neither its CPU count nor its deployment ``.env`` says anything about
    them. The chart's ``env`` entries take precedence over the Secret the pods
    load with ``envFrom``, so a thread value in that Secret is replaced.
    """
    configured = configured_torch_threads(config)
    return DEFAULT_TORCH_THREADS if configured is None else configured
