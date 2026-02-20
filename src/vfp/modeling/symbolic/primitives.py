import math

from deap import gp
import numpy as np


def _add(a: float, b: float) -> float:
    return a + b


def _sub(a: float, b: float) -> float:
    return a - b


def _mul(a: float, b: float) -> float:
    return a * b


def _neg(a: float) -> float:
    return -a


def _square(a: float) -> float:
    return a * a


def _abs(a: float) -> float:
    return abs(a)


def _protected_div(left: float, right: float) -> float:
    """Division that returns the numerator when the denominator is near zero."""
    return left / right if abs(right) > 1e-12 else left


def _protected_sqrt(x: float) -> float:
    """Square root of absolute value to avoid complex numbers."""
    return math.sqrt(abs(x))


def _random_ephemeral_constant() -> float:
    """Ephemeral constant generator for the primitive set."""
    return float(np.random.uniform(-1.0, 1.0))


def build_primitive_set(feature_count: int) -> gp.PrimitiveSet:
    """Build the GP primitive set with operators suitable for VFP regression.

    All primitives use plain Python functions (not numpy ufuncs) to ensure
    consistent return types that DEAP's type system expects.
    """
    pset = gp.PrimitiveSet("MAIN", feature_count)
    pset.addPrimitive(_add, 2)
    pset.addPrimitive(_sub, 2)
    pset.addPrimitive(_mul, 2)
    pset.addPrimitive(_protected_div, 2)
    pset.addPrimitive(_protected_sqrt, 1)
    pset.addPrimitive(_square, 1)
    pset.addPrimitive(_abs, 1)
    pset.addPrimitive(_neg, 1)
    pset.addEphemeralConstant("rand", _random_ephemeral_constant)
    return pset
