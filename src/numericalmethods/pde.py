
from typing import Tuple, Callable

import numpy as np


def __tridiag(v1: float, v2: float, v3: float, N: int, k1: int = -1, k2: int = 0, k3: int = 1) -> np.ndarray:
    return np.diag(np.full(N-abs(k1), v1), k1) + np.diag(np.full(N-abs(k2), v2), k2) + np.diag(np.full(N-abs(k3), v3), k3)


def explicit_parabollic(h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable) -> Tuple[np.ndarray]:
    r'''Computes, explicitly, a parabollic PDE of the kind:
    :math: `$$\begin{array}{l}
                \frac{du}{dt} = \frac{d^2u}{dt^2} \\
                u(x0,t) = u(xf,t) = 0 \\
                u(x,t0) = u0
                \end{array}$$`

    over the interval :math: `$[t0,tf]$` for a stepsize `h` in x and `k` in t,
    with forward finite differences.

    Note that the coefficient (`$1-2*k/h$`) should be nonnegative, otherwise
    the errors will be magnified.

    Args:
        h (float): Step size in x.
        k (float): Step size in t.
        x0 (float): Initial position.
        xf (float): Final position.
        t0 (float): Initial time.
        tf (float): Final time.
        u0 (function): Function of x in t0. u(x, t0).

    Returns:
        Tuple[np.ndarray]: Solution of the PDE in those intervals, x mesh, t mesh.
    '''
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
    sol[0, 0] = sol[-1, 0] = 0

    for ti in range(1, LEN_T):
        sol[:, ti] = m @ sol[:, ti-1]

    return sol, x, t


def implicit_parabolic(h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable) -> Tuple[np.ndarray]:
    r'''Computes, implicitly, a parabollic PDE of the kind:
    :math: `$$\begin{array}{l}
                \frac{du}{dt} = \frac{d^2u}{dt^2} \\
                u(x0,t) = u(xf,t) = 0 \\
                u(x,t0) = u0
                \end{array}$$`

    over the interval :math: `$[t0,tf]$` for a stepsize `h` in x and `k` in t,
    with backward finite differences.

    Args:
        h (float): Step size in x.
        k (float): Step size in t.
        x0 (float): Initial position.
        xf (float): Final position.
        t0 (float): Initial time.
        tf (float): Final time.
        u0 (function): Function of x in t0. u(x, t0).

    Returns:
        Tuple[np.ndarray]: Solution of the PDE in those intervals, x mesh, t mesh.
    '''
    s = h**2/k

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+h, h)

    LEN_X = len(x)
    LEN_T = len(t)

    v1 = v3 = -1
    v2 = 2 + s

    m = s * __tridiag(v1, v2, v3, LEN_X)**(-1)

    sol = np.zeros((LEN_X, LEN_T))
    sol[:, 0] = u0(x)
    sol[0, 0] = sol[-1, 0] = 0

    for ti in range(1, LEN_T):
        sol[:, ti] = m @ sol[:, ti-1]

    return sol, x, t


def crank_nik_parabolic(h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable) -> Tuple[np.ndarray]:
    r'''Computes, using crank nikolson, a parabollic PDE of the kind:
    :math: `$$\begin{array}{l}
                \frac{du}{dt} = \frac{d^2u}{dt^2} \\
                u(x0,t) = u(xf,t) = 0 \\
                u(x,t0) = u0
                \end{array}$$`

    over the interval :math: `$[t0,tf]$` for a stepsize `h` in x and `k` in t.

    Args:
        h (float): Step size in x.
        k (float): Step size in t.
        x0 (float): Initial position.
        xf (float): Final position.
        t0 (float): Initial time.
        tf (float): Final time.
        u0 (function): Function of x in t0. u(x, t0).

    Returns:
        Tuple[np.ndarray]: Solution of the PDE in those intervals, x mesh, t mesh.
    '''
    s = k/h**2

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+h, h)

    LEN_X = len(x)
    LEN_T = len(t)

    v1_left = v3_left = -s
    v2_left = 2 + 2*s

    v1_right = v3_right = s
    v2_right = 2 - 2*s

    m_left = __tridiag(v1_left, v2_left, v3_left, LEN_X)
    m_right = __tridiag(v1_right, v2_right, v3_right, LEN_X)

    m = (m_left**(-1)) @ m_right

    sol = np.zeros((LEN_X, LEN_T))
    sol[:, 0] = u0(x)
    sol[0, 0] = sol[-1, 0] = 0

    for ti in range(1, LEN_T):
        sol[:, ti] = m @ sol[:, ti-1]

    return sol, x, t


def theta_parabolic(theta: float, h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable) -> Tuple[np.ndarray]:
    r'''Computes, using the Theta method, a parabollic PDE of the kind:
    :math: `$$\begin{array}{l}
                \frac{du}{dt} = \frac{d^2u}{dt^2} \\
                u(x0,t) = u(xf,t) = 0 \\
                u(x,t0) = u0
                \end{array}$$`

    over the interval :math: `$[t0,tf]$` for a stepsize `h` in x and `k` in t,
    using a customizable mixture between explicit and implicit finite differences.

    To solve it explicitly use `theta=0`. `theta=1` solves the PDE implicitly, and
    `theta=1/2` does it with crank nikolson.

    Args:
        h (float): Step size in x.
        k (float): Step size in t.
        x0 (float): Initial position.
        xf (float): Final position.
        t0 (float): Initial time.
        tf (float): Final time.
        u0 (function): Function of x in t0. u(x, t0).

    Returns:
        Tuple[np.ndarray]: Solution of the PDE in those intervals, x mesh, t mesh.
    '''
    s = k/h**2

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+h, h)

    LEN_X = len(x)
    LEN_T = len(t)

    v1_left = v3_left = theta-1
    v2_left = -2*theta + 1

    v1_right = v3_right = theta*s
    v2_right = -2*theta*s - 1

    m_left = __tridiag(v1_left, v2_left, v3_left, LEN_X)
    m_right = __tridiag(v1_right, v2_right, v3_right, LEN_X)

    m = (m_left**(-1)) @ m_right

    sol = np.zeros((LEN_X, LEN_T))
    sol[:, 0] = u0(x)
    sol[0, 0] = sol[-1, 0] = 0

    for ti in range(1, LEN_T):
        sol[:, ti] = m @ sol[:, ti-1]

    return sol, x, t


def expicit_advection_diffusion(v: float, K: float, h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable) -> Tuple[np.ndarray]:
    r'''Computes, explicitly, the Advection diffusion PDE of the kind:
    :math: `$$\begin{array}{l}
                U_t+v U_x=K U_{xx} \\
                U(x0,t) = 0 \; \textrm{ and } \; \frac{\partial U}{\partial x}(xf, t)=0 \\
                U(x,t0)=u0(x) \\
                \end{array}$$`

    where :math:`$v$` is the river's velocity, and :math:`$K$` the diffusion coefficient,
    over the interval :math: `$[t0,tf]$` for a stepsize `h` in x and `k` in t,7
    using `forward finite differences`.

    Args:
        v (float): River's velocity.
        K (float): Diffusion coefficient.
        h (float): Step size in x.
        k (float): Step size in t.
        x0 (float): Initial position.
        xf (float): Final position.
        t0 (float): Initial time.
        tf (float): Final time.
        u0 (function): Function of x in t0. u(x, t0).

    Returns:
        Tuple[np.ndarray]: Solution of the PDE in those intervals, x mesh, t mesh.
    '''
    s = k/h

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+h, h)

    LEN_X = len(x)
    LEN_T = len(t)

    v1 = K*s/h
    v2 = -2*K*s/h - v*s + 1
    v3 = K*s/h -v*s
    
    m = __tridiag(v1, v2, v3, LEN_X)

    sol = np.zeros((LEN_X, LEN_T))
    sol[:, 0] = u0(x)

    for ti in range(1, LEN_T):
        sol[:, ti] = m @ sol[:, ti-1]
    
    return sol, x, t


def implicit_advection_diffusion(v: float, K: float, h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable) -> Tuple[np.ndarray]:
    r'''Computes, implicitly, the Advection diffusion PDE of the kind:
    :math: `$$\begin{array}{l}
                U_t+v U_x=K U_{xx} \\
                U(x0,t) = 0 \; \textrm{ and } \; \frac{\partial U}{\partial x}(xf, t)=0 \\
                U(x,t0)=u0(x) \\
                \end{array}$$`

    where :math:`$v$` is the river's velocity, and :math:`$K$` the diffusion coefficient,
    over the interval :math: `$[t0,tf]$` for a stepsize `h` in x and `k` in t,7
    using `backward finite differences`.

    Args:
        v (float): River's velocity.
        K (float): Diffusion coefficient.
        h (float): Step size in x.
        k (float): Step size in t.
        x0 (float): Initial position.
        xf (float): Final position.
        t0 (float): Initial time.
        tf (float): Final time.
        u0 (function): Function of x in t0. u(x, t0).

    Returns:
        Tuple[np.ndarray]: Solution of the PDE in those intervals, x mesh, t mesh.
    '''
    s = k/h

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+h, h)

    LEN_X = len(x)
    LEN_T = len(t)

    v1 = v*s + K*s/h
    v2 = -2*K*s/h - v*s + 1
    v3 = K*s/h
    
    m = __tridiag(v1, v2, v3, LEN_X)

    sol = np.zeros((LEN_X, LEN_T))
    sol[:, 0] = u0(x)

    # Boundary conditions
    sol[0, 0] = 0
    # ---

    for ti in range(1, LEN_T):
        sol[:, ti] = m @ sol[:, ti-1]
    
    return sol, x, t


def hyperbolic(h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable) -> Tuple[np.ndarray]:
    
    rho = (k/h)**2
    raise NotImplementedError()