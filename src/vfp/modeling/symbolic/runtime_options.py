from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

CacheMode = Literal["off", "safe", "all"]
ExecutionMode = Literal["auto", "sequential", "threaded"]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_cache_mode() -> CacheMode:
    raw = os.environ.get("VLP_CACHE_MODE")
    if raw is None:
        return "all" if env_flag("VLP_CACHE_ENABLED", True) else "off"

    value = raw.strip().lower()
    if value in {"off", "safe", "all"}:
        return value  # type: ignore[return-value]

    raise ValueError("VLP_CACHE_MODE must be one of: off, safe, all")


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    cache_mode: CacheMode = "all"
    execution_mode: ExecutionMode = "auto"

    @property
    def result_cache_enabled(self) -> bool:
        return self.cache_mode == "all"

    @property
    def compile_cache_enabled(self) -> bool:
        return self.cache_mode == "all"

    @property
    def fitness_cache_enabled(self) -> bool:
        return self.cache_mode == "all"

    @property
    def static_cache_enabled(self) -> bool:
        return self.cache_mode in {"safe", "all"}

    @property
    def force_sequential(self) -> bool:
        return self.execution_mode == "sequential"


DEFAULT_RUNTIME_OPTIONS = RuntimeOptions(
    cache_mode=env_cache_mode(),
    execution_mode="sequential" if env_flag("VLP_FORCE_SEQUENTIAL", False) else "auto",
)
