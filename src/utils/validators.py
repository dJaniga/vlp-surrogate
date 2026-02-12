from typing import TypeVar

T = TypeVar('T')

def ensure_not_none(value: T | None) -> T:
    if value is None:
        raise ValueError("Value cannot be None")
    return value