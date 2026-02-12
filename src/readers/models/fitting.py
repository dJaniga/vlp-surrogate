from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FitResults:
    well_name: str
    overall: Mapping[str, float] = field(default_factory=dict)
    production: Mapping[str, float] = field(default_factory=dict)
    injection: Mapping[str, float] = field(default_factory=dict)
