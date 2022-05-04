
from typing import Tuple

import numpy as np


def __tridiag(v1: float, v2: float, v3: float, N: int, k1: int = -1, k2: int = 0, k3: int = 1) -> np.ndarray:
    return np.diag(np.full(N-abs(k1), v1), k1) + np.diag(np.full(N-abs(k2), v2), k2) + np.diag(np.full(N-abs(k3), v3), k3)


def explicit_parabollic(h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: function) -> np.ndarray:

    s = k/h**2

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+h, h)

    LEN_X = len(x)
    LEN_T = len(t)

    v1 = v3 = s
    v2 = 1 - 2*s

    m = __tridiag(v1, v2, v3, LEN_X)

    sol = np.zeros((LEN_X, LEN_T))
    sol[:, 0] = u0(x)

    for ti in range(1, LEN_T):
        sol[:, ti] = m@sol[:, ti-1]

    return sol, x, t
