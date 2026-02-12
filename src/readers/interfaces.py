from typing import Callable

import numpy as np
import numpy.typing as npt

MetricFn = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    float,
]
