
from typing import Callable

import numpy as np

from roots import newton


def euler_explicit(f: 'Callable[float, float]', y0: float, t0: float, t: float, h: float) -> np.ndarray:
    """Computes the explicit (forward) Euler method to solve ODEs.

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
    t_ = np.arange(t0, t0+t, h)
    N = len(t_)

    u = np.zeros_like(t_)
    u[0] = y0

    for i in range(N-1):
        u[i+1] = u[i] + h * f(u[i], t_[i])

    return u


def euler_explicit_midpoint(f: 'Callable[float, float]', y0: float, t0: float, t: float, h: float) -> np.ndarray:
    """Computes the explicit (forward) midpoint Euler method to solve ODEs.

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
        np.ndarray: Numerical solution of the ODE in the interval [t0, t0+h, t-h, t].

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
    t_ = np.arange(t0, t0+t, h)
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
    """Computes the explicit (forward) Euler method to solve a system of ODEs.

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
        np.ndarray: Numerical solution of the ODE in the interval [t0, t0+h, t-h, t].

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
        >>> 
        >>> 
        >>> 
    """
    t_ = np.arange(t0, t0+t, h)
    N = len(t_)

    u = np.zeros((vec0.shape[0], N))

    u[:, t0] = vec0

    for i in range(N-1):
        u[..., i+1] = u[..., i] + h * f(*u[..., i])

    return u

# TODO: Pending tests


def euler_implicit(f: 'Callable[float, float]', y0: float, t0: float, t: float, h: float, *args, **kwargs) -> np.ndarray:
    """Computes the implicit (backward) Euler method to solve ODEs.

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

        def g(y): return u[i] + u[i+1] + h*f(y, t_[i+1])
        u[i+1] = newton(*args, f=g, x0=u[i], **kwargs)

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
    t_ = np.arange(t0, t0+t, h)
    N = len(t_)

    u = np.zeros_like(t_)
    u[0] = y0

    for i in range(N-1):
        u[i+1] = u[i] + h/2 * (f(u[i]+h*f(u[i], t_[i]), t_[i+1]) + f(u[i], t_[i]))

    return u
