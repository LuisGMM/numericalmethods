

from typing import Callable

import numpy as np


def composite_trapezoid(f_: 'Callable[float]', a: float, b: float, n: float) -> float:
    """Computes the analitical solution of the integral of f from a to b
    following the composite trapezoidal rule.

    Args:
        f_ (Callable[float]): Function to be integrated.
        a (float): Lower bound of hte interval.
        b (float): Upper bound of the interval.
        n (float): The number of parts the interval is divided into.

    Returns:
        float: Numerical solution of the integral.
    """
    x = np.linspace(a, b, n + 1)
    f = f_(x)
    h = (b - a) / (n)

    return h/2 * sum(f[:n] + f[1:n+1])


def composite_simpson(f_: 'Callable[float]', a: float, b: float, n: float) -> float:
    """Computes the analitical solution of the integral of f from a to b
    following the composite Simpson's 1/3 rule.

    Args:
        f_ (Callable[float]): Function to be integrated.
        a (float): Lower bound of hte interval.
        b (float): Upper bound of the interval.
        n (float): The number of parts the interval is divided into.

    Returns:
        float: Numerical solution of the integral.
    """
    x = np.linspace(a, b, n+1)
    f = f_(x)
    h = (b - a) / (n)

    return (h/3) * (f[0] + 2*sum(f[2:n-1:2]) + 4*sum(f[1:n:2]) + f[n])
