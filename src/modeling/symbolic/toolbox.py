from __future__ import annotations

import logging

import numpy as np
from deap import base, creator, gp, tools


logger = logging.getLogger(__name__)


def build_toolbox(
    pset: gp.PrimitiveSet,
    *,
    rng: np.random.Generator,
    max_tree_height: int,
    tournament_size: int,
) -> base.Toolbox:
    if not hasattr(creator, "SymbolicFitness"):
        creator.create("SymbolicFitness", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "SymbolicIndividual"):
        creator.create(
            "SymbolicIndividual",
            gp.PrimitiveTree,
            fitness=creator.SymbolicFitness,  # type: ignore[attr-defined]
        )

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_tree_height)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.SymbolicIndividual,  # type: ignore[attr-defined]
        toolbox.expr,  # type: ignore[attr-defined]
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,  # type: ignore[attr-defined]
    )
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=max_tree_height)
    toolbox.register(
        "mutate",
        gp.mutUniform,
        expr=toolbox.expr_mut,  # type: ignore[attr-defined]
        pset=pset,
    )
    max_nodes = max_tree_height * 4
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_nodes))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_nodes))
    logger.debug(
        "Symbolic toolbox ready",
        extra={"max_height": max_tree_height, "tournament": tournament_size},
    )
    return toolbox
