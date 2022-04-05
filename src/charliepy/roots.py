import sys
import warnings
from typing import Callable

import numpy as np

from charliepy.exceptions import InadequateArgsCombination

# TODO: Shield all methods.
# TODO: Adequately use logging and warnings.
# TODO: Implement euler_explicit_systems with explicit dependance on time.
# TODO: Implement newton for systems.
# TODO: Improve docstrings; implement examples.
# TODO: Implement testcases.


def newton(err: float, f: 'Callable[float]' = None, f_dev: 'Callable[float]' = None,
           integrator: 'Callable[Callable, float, float, float]' = None, differentiator: 'Callable[int, Callable, float, float, bool]' = None, *,
           c: float = 0, x0: float = 0, n: int = 100_000, h_err: float = 1e-4) -> float:
    r"""Newton's method to find roots of a function.

    If no `f` is given but `f_dev` and `integrator` are, it will compute the roots of the integral of `f_dev` with integration constant c.
    If `f_dev` is not given, it will be computed from `f` with the mathematical definition of a derivative.

    Args:
        err (float): Desired error of the method.
        f (Callable[float], optional): Analytical function to find its roots. Its input is the point to be evaluated in. Defaults to None.
        f_dev (Callable[float], optional): Analytical derivative of the function. Its input is the point to be evaluated in. Defaults to None.
        integrator (Callable[Callable, float, float, float], optional): Integration method to compute the integral of `f_dev` and find its roots.
            It should be `composite_trapezoid` or `composite_simpson` methods. Defaults to None.
        differentiator (Callable[int, Callable, float, float, bool]): Differentiation method to compute the derivative of `f` during the method.
            It should be `forward`, `backward` or `central` methods from differentiate module. Defaults to None.
        c (float, optional): Integration constant of the integral of f_dev. Defaults to 0.
        x0 (float, optional): Initial guess of the root.
            Note that an inadequate first guess could lead to undesired outputs such as no roots or undesired roots.
            Defaults to 0.
        n (int, optional): The number of parts the interval of the integrator method is divided into. Defaults to 100_000.
        h_err (float, optional): Finite approximation of 0 to use in the calculation of `f_dev` by its mathematical definition. Defaults to 1e-4.

    Raises:
        InadequateArgsCombination: If the combination of arguments is not valid.

    Returns:
        float|None: Root of the function or None if the algorithm reaches its recursion limit.

    Examples:
        :math: `$$\int_{0}^{x}\frac{1}{\sqrt{2\\pi}}e^{-t^2/2}dt=0.45$$`

        can be solved for x with :math: `$$f(x)=\int_{0}^{x}\frac{1}{\sqrt{2\pi}}e^{-t^2/2}dt-0.45$$`

        and :math: `$$f'(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}.$$`

        To evaluate f(x) at the approximation to the root :math: `$p_k$` we need a quadrature formula to approximate
            :math: `$$\int_{0}^{p_k}\frac{1}{\sqrt{2\pi}}e^{-t^2/2}dt$$`

        Find a solution to :math:`$f(x) = 0$` accurate to within :math:`$10^{-5}$` using Newton\'s method with :math:`$p_0 = 0.5$`
        and the Composite Simpson\'s rule.

        >>> f_dev = lambda x: 1 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
        >>> ERR = 1e-5
        >>> x0 = 0.5
        >>> C = -0.45
        >>> ans = newton(composite_simpson, f_dev, C, ERR, x0)
        >>> ans
        1.6448536269514884
    """
    if (f or integrator) and f_dev:
        if f and integrator:
            warnings.warn('`f`, `f_dev` and `integrator` args detected. Only `f` and `f_dev` will be used for sake of precision.')
            def iteration(iter_idx, iter_dict): return iter_dict[iter_idx] - f(iter_dict[iter_idx]) / f_dev(iter_dict[iter_idx])

        elif integrator:
            def iteration(iter_idx, iter_dict): return iter_dict[iter_idx] - (integrator(f_dev, 0, iter_dict[iter_idx], n) + c) / f_dev(iter_dict[iter_idx])

        else:
            def iteration(iter_idx, iter_dict): return iter_dict[iter_idx] - f(iter_dict[iter_idx]) / f_dev(iter_dict[iter_idx])

    elif (f_dev or differentiator) and f:

        if f_dev and differentiator:
            warnings.warn('`f`, `f_dev` and `differentiator` args detected. Only `f` and `f_dev` will be used for sake of precision.')
            def iteration(iter_idx, iter_dict): return iter_dict[iter_idx] - f(iter_dict[iter_idx]) / f_dev(iter_dict[iter_idx])

        elif differentiator:
            def iteration(iter_idx, iter_dict): return iter_dict[iter_idx] - f(iter_dict[iter_idx]) / \
                differentiator(1, f, iter_dict[iter_idx], h_err, True)

        else:
            def iteration(iter_idx, iter_dict): return iter_dict[iter_idx] - f(iter_dict[iter_idx]) / f_dev(iter_dict[iter_idx])

    else:
        raise InadequateArgsCombination('Cannot compute Newton\'s method with the combination of arguments given. Check the valid combinations.')

    iter, iter_dict = 0, {0: x0}
    limit = sys.getrecursionlimit()

    while True:
        if iter + 10 >= limit:
            warnings.warn(
                f'Iteration limit ({limit}) reached without finding any root. Try with other initial guess or changing the recursion limit. \
                    Maybe there are no roots.')
            return

        iter_dict[iter+1] = iteration(iter, iter_dict)

        if abs(iter_dict[iter+1] - iter_dict[iter]) < err:
            return iter_dict[iter+1]

        iter += 1


def newton_systems(f: 'Callable[float, ...]', J: 'Callable[float, ...]', vec0: np.ndarray, err: float) -> np.ndarray:
    r"""Solves systems of linear and nonlinear equations using the Newton method.

    Args:
        f (Callable[float, ...]): Vector function to find its roots.
        J (Callable[float, ...]): Jacobian of f.
        vec0 (np.ndarray): Initial guess of the solution. Avoid using guesses that make J a singular matrix (:math:`|J(vec0)| = 0`).
        err (float): Stopping criteria for the algorithm. Minimum difference between the to last consecutive solutions.

    Raises:
        ValueError: If the Jacobian of vec0 is a singular matrix, because its inverse cannot be computed.

    Returns:
        np.ndarray|None: Root of the function or None if the algorithm reaches its recursion limit.

    Examples:
        Solve
        ..math::
            `x^2+y^2-25=0  \\ x-y-2=0`
        With an adequate initial guess.

        >>> f = lambda x, y: [x**2 + y**2 -25,
                              x - y - 2]
        >>> J = lambda x, y: [[2*x, 2*y],
                              [1, -1]]
        >>> err = 1e-10
        >>> vec0 = [0, 0] #Invalid initial guess.
        >>> newton_systems(f, J, vec0, err)
        Raises numpy.linalg.LinAlgError: Singular matrix
        >>> vec0 = [1, 0] #Valid initial guess.
        >>> roots = newton_systems(f, J, vec0, err)
        >>> roots
        [4.39116499 2.39116499]
        >>> f(*roots)
        [-3.552713678800501e-15, -4.440892098500626e-16]
    """
    if np.linal.det(J(*vec0)) == 0:
        raise ValueError('Inverse of the Jacobian cannot be computed. It is a singular matrix (Determinant of the matrix is 0). ')

    iter, iter_dict = 0, {0: vec0}
    limit = sys.getrecursionlimit()

    while True:
        if iter + 10 >= limit:
            warnings.warn(
                f'Iteration limit ({limit}) reached without finding any root. Try with other initial guess or changing the recursion limit.\
                     Maybe there are no roots.')
            return

        iter_dict[iter+1] = iter_dict[iter] - np.matmul(np.linalg.inv(J(*iter_dict[iter])), f(*iter_dict[iter]))

        if np.all(abs(iter_dict[iter + 1] - iter_dict[iter]) < err):
            return iter_dict[iter+1]

        iter += 1


def bisection(f: 'Callable[float]', a: float, b: float, err: float, Nmax: int = 100_000) -> float:
    r"""Computes Bisection method to find roots f(x)=0.

    If there are no roots in the interval :math:`[a, b]`, the method will throw an exception.
    This is checked using bolzano's theorem (If :math:`f(a)*f(b) >= 0`).

    Args:
        f (Callable[float]): Function of which we want to find roots :math:`f(x)=0`.
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        err (float): Tolerance of the result. It assures that the root is in :math:`[x-err, x+err]`. #TODO: Is this the interval?
        Nmax (int): Maximum number of iterations. Defaults to 100_000.

    Raises:
        ValueError: If, according to Bolzano's theorem, there cannot be roots in :math:`[a, b]`.
        ValueError: If the method, being at least one root in :math:`[a, b]`, fails to to compute the root.

    Returns:
        float: Root x such as f(x)=0 with a tolerance err.

    Examples:
        >>> f = lambda x: (x**2 - 1)
        >>> bisection(f, -0.5, 2, 1e-10)
        2.9103830456733704e-11
        >>> bisection(f, -0.5, 2, 1e-10, 100)
        ValueError: Could not find a root in the interval [-0.5, 2] with tolerance 1e-10 in 5 iterations.
        >>> bisection(f, 5, 20, 1e-7)
        ValueError: f(a)*f(b)=9576 <0.   No roots in this interval.
    """
    if f(a)*f(b) >= 0:
        raise ValueError(f'f(a)*f(b) = {f(a)*f(b)} <0. \t No roots in this interval.')

    N = int(min(Nmax, np.ceil(np.log((b-a)/err) / np.log(2) - 1)))  # What is this?
    a_n = a
    b_n = b
    m_n = (a_n + b_n)/2
    f_m_n = f(m_n)
    for _ in range(1, N+1):

        if f(a_n)*f_m_n < 0:
            b_n = m_n

        elif f(b_n)*f_m_n < 0:
            a_n = m_n

        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)

        if abs(f_m_n) <= err:
            return m_n

    raise ValueError(f'Could not find a root in the interval [{a}, {b}] with tolerance {err} in {N} iterations.')

# TODO: Not validated


def chord(f: 'Callable[float]', a: float, b: float, err: float, Nmax: int = 100_000, x0: float = None) -> float:
    """Computes Chord method to find roots f(x)=0. 

    If there are no roots in the interval :math:`[a, b]`, the method will throw an exception. 
    This is checked using bolzano's theorem (If :math:`f(a)*f(b) >= 0`).

    Args:
        f (Callable[float]): Function of which we want to find roots :math:`f(x)=0`.
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        err (float): Tolerance of the result. It assures that the root is in :math:`[x-err, x+err]`. #TODO: Is this the interval?
        Nmax (int): Maximum number of iterations. Defaults to 100_000.
        x0 (float): Initial guess for the root. Defaults :math:`(a+b)/2`.

    Raises:
        ValueError: If, according to Bolzano's theorem, there cannot be roots in :math:`[a, b]`. 
        ValueError: If the method, being at least one root in :math:`[a, b]`, fails to to compute the root.

    Returns:
        float: Root x such as f(x)=0 with a tolerance err.
    """
    if f(a)*f(b) >= 0:
        raise ValueError(f'f(a)*f(b) = {f(a)*f(b)} <0. \t No roots in this interval.')

    x0_ = x0 if x0 is not None else (a+b)/2  # TODO: Check if there is a better initial guess
    f_x_n = f(x0_)
    q = (f(b) - f(a)) / (b - a)

    for _ in range(1, Nmax+1):

        x_n = x0_ - f_x_n / q
        f_x_n = f(x_n)

        if abs(f_x_n) <= err:
            return x_n

    raise ValueError(f'Could not find a root in the interval [{a}, {b}] with tolerance {err} in {Nmax} iterations.')


def secant(f: 'Callable[float]', a: float, b: float, err: float, Nmax: int = 100_000, x0: float = None) -> float:
    """Computes Secant method to find roots :math:`f(x)=0`. 

    If there are no roots in the interval :math:`[a, b]`, the method will throw an exception. 
    This is checked using bolzano's theorem (If :math:`f(a)*f(b) >= 0`).

    To computes the first iteration, it computes the previous value as :math: `a-1`

    Args:
        f (Callable[float]): Function of which we want to find roots :math:`f(x)=0`.
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        err (float): Tolerance of the result. It assures that the root is in :math:`[x-err, x+err]`. #TODO: Is this the interval?
        Nmax (int): Maximum number of iterations. Defaults to 100_000.
        x0 (float): Initial guess for the root. Defaults :math:`(a+b)/2`.

    Raises:
        ValueError: If, according to Bolzano's theorem, there cannot be roots in :math:`[a, b]`. 
        ValueError: If the method, being at least one root in :math:`[a, b]`, fails to to compute the root.

    Returns:
        float: Root x such as f(x)=0 with a tolerance err.

    Examples:
        >>> f = lambda x: (x**3 - 5*x - 9)
        >>> secant(f, 2, 5, 1e-4)
        2.8551984513616424
    """
    if f(a)*f(b) >= 0:
        raise ValueError(f'f(a)*f(b) = {f(a)*f(b)} <0. \t No roots in this interval.')

    x_n = x0 if x0 is not None else (a+b)/2  # TODO: Check if there is a better initial guess
    x_previous = a - 1  # TODO: Check if there is a better initial guess
    f_x_n = f(x_n)

    for _ in range(1, Nmax+1):

        f_x_previous = f(x_previous)

        q_n = (f_x_n - f_x_previous) / (x_n - x_previous)
        x_previous = x_n

        x_n = x_n - f_x_n / q_n

        f_x_n = f(x_n)
        if abs(f_x_n) <= err:
            return x_n

    raise ValueError(f'Could not find a root in the interval [{a}, {b}] with tolerance {err} in {Nmax} iterations.')


if __name__ == '__main__':
    pass
