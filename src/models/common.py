from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FlowType(Generic[T]):
    production: T | None = None
    injection: T | None = None

    def __iter__(self):
        yield from (self.production, self.injection)
