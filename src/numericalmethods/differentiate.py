
from typing import Callable

import scipy.special as sci


def forward(order: int, f: Callable[[float], float], x: float, h: float, exact: bool = False) -> float:
    """Use forward finite difference formula of order `order` to compute the derivative of `f` at `x`.

    Args:
        order (int): Order of the derivate. (first, second ...).
        f (Callable[[float], float]): Function for which the derivative will be computed.
            Only one argument will be passed to it, `x`.
        x (float): Point at which the derivative will be computed.
        h (float): Error of the approximation # TODO: Is it the error?
        exact (bool, optional): Set the precision of the method.
            If False, floating comma numbers are used. Set to True to use long numbers. Defaults to False.

    Note:
        See more of `exact` parameter in https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html

    Returns:
        float: Derivative of order `order` of `f` evaluated at `x`.
    """
    return sum([(-1)**(order-k) * sci.comb(order, k, exact) * f(x + k*h) for k in range(order+1)]) / h**order


def backward(order: int, f: Callable[[float], float], x: float, h: float, exact: bool = False) -> float:
    """Use backward finite difference formula of order `order` to compute the derivative of `f` at `x`.

    Args:
        order (int): Order of the derivate. (first, second ...).
        f (Callable[[float], float]): Function for which the derivative will be computed.
            Only one argument will be passed to it, `x`.
        x (float): Point at which the derivative will be computed.
        h (float): Error of the approximation # TODO: Is it the error?
        exact (bool, optional): Set the precision of the method.
            If False, floating comma numbers are used. Set to True to use long numbers. Defaults to False.

    Note:
        See more of `exact` parameter in https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html

    Returns:
        float: Derivative of order `order` of `f` evaluated at `x`.
    """
    return sum([(-1)**(k) * sci.comb(order, k, exact) * f(x - k*h) for k in range(order+1)]) / h**order


def central(order: int, f: 'Callable', x: float, h: float, exact: bool = False) -> float:
    """Use central finite difference formula of order `order` to compute the derivative of `f` at `x`.

    Args:
        order (int): Order of the derivate. (first, second ...).
        f (Callable): Function for which the derivative will be computed.
            Only one argument will be passed to it, `x`.
        x (float): Point at which the derivative will be computed.
        h (float): Error of the approximation # TODO: Is it the error?
        exact (bool, optional): Set the precision of the method.
            If False, floating comma numbers are used. Set to True to use long numbers. Defaults to False.

    Note:
        See more of `exact` parameter in https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html

    Returns:
        float: Derivative of order `order` of `f` evaluated at `x`.
    """
    return sum([(-1)**(k) * sci.comb(order, k, exact) * f(x - (order/2 - k)*h) for k in range(order+1)]) / h**order


if __name__ == '__main__':
    pass
