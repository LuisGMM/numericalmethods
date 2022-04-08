
import inspect
from typing import Callable

import numpy as np

from numericalmethods.roots import newton


def euler_explicit(f: 'Callable[float, float]', y0: float, t0: float, t: float, h: float) -> np.ndarray:
    r"""Computes the explicit (forward) Euler method to solve ODEs.

    Args:
        f (Callable[float, float]): Function depending on y and t in that order.
            Equivalent to f(y,t).
        y0 (float): Initial value of the answer.
            Equivalent to y(t0).
        t0 (float): Initial time.
        t (float): Final time.
        h (float): Separation between the points of the interval.

    Returns:
        np.ndarray: Numerical solution of the ODE in the interval [t0, t0+h, ..., t-h, t].

    Examples:

        Lets solve the problem

        :math: `$$\begin{array}{l}
                y'=\lambda y \\
                y(0) = 1
                \end{array}$$`

        for :math:`$\lambda = -1$` over the interval :math: `$[0,1]$` for a stepsize `$h=0.1$`.

        Then:
        >>> f = lambda y, t: -y
        >>> y0 = 1
        >>> h = 0.1
        >>> t0, t = 0, 1
        >>> y = euler_explicit(f, y0, t0, t, h)
        >>> print(y)
    """
    t_ = np.arange(t0, t+h, h)
    N = len(t_)

    u = np.zeros_like(t_)
    u[0] = y0

    for i in range(N-1):
        u[i+1] = u[i] + h * f(u[i], t_[i])

    return u


def euler_explicit_midpoint(f: 'Callable[float, float]', y0: float, t0: float, t: float, h: float) -> np.ndarray:
    r"""Computes the explicit (forward) midpoint Euler method to solve ODEs.

    The **explicit midpoint method** is :math: `u_{n+1}=u_{n-1}+2hf\left(t_n,u_n\right)`

    As two initial values are required, if y0_previous is not provided, it is computed with :math: `$y(-h)=y(0)-hf(0,y(0))$`.

    Args:
        f (Callable[float, float]): Function depending on y and t in that order.
            Equivalent to f(y,t).
        y0 (float): Initial value of the answer.
            Equivalent to y(t0).
        t0 (float): Initial time.
        t (float): Final time.
        h (float): Separation between the points of the interval.

    Returns:
        np.ndarray: Numerical solution of the ODE in the interval [t0, t0+h, ..., t-h, t].

    Examples:

        Lets solve the problem

        :math: `$$\begin{array}{l}
                y'=\lambda y \\
                y(0) = 1
                \end{array}$$`

        for :math:`$\lambda = -1$` over the interval :math: `$[0,1]$` for a stepsize `$h=0.1$`.

        Then:
        >>> f = lambda y, t: -y
        >>> y0 = 1
        >>> h = 0.1
        >>> t0, t = 0, 1
        >>> y = euler_explicit_midpoint(f, y0, t0, t, h)
        >>> print(y)
    """
    t_ = np.arange(t0, t+h, h)
    N = len(t_)

    u = np.zeros_like(t_)
    u_previous = y0 - h * f(y0, t_[0])
    u[0] = y0

    for i in range(N-1):
        if i == 0:
            u[i+1] = u_previous + 2 * h * f(u[i], t_[i])
        else:
            u[i+1] = u[i-1] + 2 * h * f(u[i], t_[i])

    return u

# TODO: Currently the method does not support ODEs that explicitly depend on time. That means `f` and `vec0` must have the same dimensions.


def euler_explicit_systems(f: 'Callable[float, ...]', vec0: np.ndarray, t0: float, t: float, h: float) -> np.ndarray:
    r"""Computes the explicit (forward) Euler method to solve a system of ODEs.

    The order of the arguments (variables) in `f` must the the same of the values in `vec0`.

    Args:
        f (Callable[float, ...]): Function depending on the any number of variables.
            Currently it does not support explicit dependence on time.
            Equivalent to f(y,t).
        vec0 (np.ndarray): Initial values of the answer.
            Equivalent to [x(t0), y(t0), ...].
        t0 (float): Initial time.
        t (float): Final time.
        h (float): Separation between the points of the interval.

    Returns:
        np.ndarray: Numerical solution of the ODE in the interval [t0, t0+h, ..., t-h, t].

    Examples:
        Lets solve the Lorentz equations :math:`$$\begin{array}{l}
                                                \frac{dx}{dt}=\sigma(y-x) \\
                                                \frac{dy}{dt}=x(\rho-z)-y \\
                                                \frac{dz}{dt}=xy-\beta z
                                                \end{array}
                                                $$`

        for :math:`$\sigma=10$`, :math:`$\rho=28$`, :math:`$\beta=8/3$`, :math:`$t_0=0$`, :math:`$t_f=50$` and :math:`$(x[0],y[0],z[0])=(0, 1, 1.05)$`

        Then
        >>> import numpy as np
        >>> t0, t = 0, 50
        >>> vec0 = np.array([0, 1, 1.05])
        >>> s, r, b = 10, 28, 8/3
        >>> f = lambda x, y, z: np.array([s*(y-x), x*(r-z)-y, x*y - b*z])
        >>> h = 1e-4
        >>> u = euler_explicit_systems(f, vec0, t0, tf, h)

        If we want to plot these results

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(figsize = (10,10))
        >>> ax = plt.axes(projection='3d')
        >>> ax.grid()
        >>> ax.plot3D(u[0,:], u[1, :], u[2, :])
        >>> ax.set_xlabel('x', labelpad=20)
        >>> ax.set_ylabel('y', labelpad=20)
        >>> ax.set_zlabel('z', labelpad=20)
    """
    t_ = np.arange(t0, t+h, h)
    N = len(t_)

    u = np.zeros((vec0.shape[0], N))

    u[:, t0] = vec0

    for i in range(N-1):
        u[..., i+1] = u[..., i] + h * f(*u[..., i])

    return u

# TODO: Pending tests


def euler_implicit(f: 'Callable[float, float]', y0: float, t0: float, t: float, h: float, *args, **kwargs) -> np.ndarray:
    """Computes the implicit (backward) Euler method to solve ODEs.

    If `f` argument has an explicit dependence on y, Newton\' method is used to compute the next iteration the algorithm.
    Then, `err` must be passed as extra argument, and it is recommended to pass the analytical derivative of `f` as f_dev.
    In case this last `f_dev` is not passed, Newton\' method will use finite differences to numerically obtain it.
    
    See more about Newton\' method in module roots.

    Args:
        f (Callable[float, float]): Function depending on y and t in that order.
            Equivalent to f(y,t).
        y0 (float): Initial value of the answer.
            Equivalent to y(t0).
        t0 (float): Initial time.
        t (float): Final time.
        h (float): Separation between the points of the interval.

    Returns:
        np.ndarray: Numerical solution of the ODE in the interval [t0, t0+h, ..., t-h, t].
    """
    returns_of_f = inspect.getsource(f).split('return')
    lambdas_of_f = inspect.getsource(f).split('lambda')

    n_returns_of_f = len(returns_of_f)
    n_lambdas_of_f = len(lambdas_of_f)

    if n_returns_of_f > 2 or n_lambdas_of_f > 2:
        raise ValueError('Function `f` is not valid. It can only have one return or one lambda.')

    if n_returns_of_f < 2 and n_lambdas_of_f < 2:
        raise ValueError('Function `f` is not valid. It must have one return or one lambda.')

    elif n_returns_of_f == 2:
        function_of_f = returns_of_f[-1]

    elif n_lambdas_of_f == 2:
        function_of_f = lambdas_of_f[-1].split(':')[-1]

    t_ = np.arange(t0, t+h, h)
    N = len(t_)

    u = np.zeros_like(t_)
    u[0] = y0

    if 'y' in function_of_f:

        for i in range(N-1):
            def g(y): return u[i] - u[i+1] + h*f(y, t_[i+1])
            u[i+1] = newton(*args, **kwargs, f=g, x0=u[i])
    else:
      
        for i in range(N-1):
            u[i+1] = u[i] + h*f(y=None, t=t_[i+1])

    return u

# TODO: To be validated


def heun(f: 'Callable[float, float]', y0: float, t0: float, t: float, h: float) -> np.ndarray:
    """Computes Heun's method to solve ODEs.

    Args:
        f (Callable[float, float]): Function depending on y and t in that order.
            Equivalent to f(y,t).
        y0 (float): Initial value of the answer.
            Equivalent to y(t0).
        t0 (float): Initial time.
        t (float): Final time.
        h (float): Separation between the points of the interval.

    Returns:
        np.ndarray: Numerical solution of the ODE in the interval [t0, t0+h, t-h, t].
    """
    t_ = np.arange(t0, t+h, h)
    N = len(t_)

    u = np.zeros_like(t_)
    u[0] = y0

    for i in range(N-1):
        u[i+1] = u[i] + h/2 * (f(u[i]+h*f(u[i], t_[i]), t_[i+1]) + f(u[i], t_[i]))

    return u


def runge_kutta4(f: 'Callable[float, float]', y0: float, t0: float, t: float, h: float) -> np.ndarray:
    """Solve a first order ODE using Runge-Kutta's method of order 4.

    Equivalent to ode45 in MATLAB.

    Args:
        f (Callable[float, float]): Function of two variables representing the ODE. y' = f(y, t). 
            Args must be in that order. 
        y0 (float): Initial value of the solution.
            Equivalent to y(t0).
        t0 (float): Initial time.
        t (float): Final time.
        h (float): Separation between the points of the interval.

    Returns:
        np.ndarray: Solution of y(t) in [t0, t]
    
    Examples:

        Lets solve the problem in the interval of time [0, 1]

        :math: `$$\begin{array}{l}
                y'=  y + t \\
                y(0) = 1
                \end{array}$$`

        >>> f = lambda y, t : y + t
        >>> y0 = 0
        >>> t0, t = 0, 1
        >>> h = 0.1
        >>> runge_kutta4(f, y0, t0, t, h)
        [1.         1.11034167 1.24280514 1.39971699 1.58364848 1.79744128 
        2.04423592 2.32750325 2.65107913 3.01920283 3.43655949]
    """    
    t_= np.arange(t0, t + h, h)
    N = len(t_)

    u = np.zeros_like(t_)
    u[0] = y0

    for i in range(N-1):

        f1 = f(u[i], t_[i])
        f2 = f(u[i] + (f1 * (h / 2)), t_[i] + h / 2)
        f3 = f(u[i] + (f2 * (h / 2)), t_[i] + h / 2)
        f4 = f(u[i] + (f3 * h), t_[i] + h)

        u[i+1] = u[i] + (h / 6) * (f1 + (2 * f2) + (2 * f3) + f4)
        
    return u