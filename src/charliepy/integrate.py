

from typing import Callable

import numpy as np



def composite_trapezoid(f_:'Callable[float]', a:float, b:float, n:float)-> float:
    """Computes the analitical solution of the integral of f from a to b 
    following the composite trapezoidal rule. 

    Args:
        f_ (Callable[float]): Function to be integrated  
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


def composite_simpson(f_:'Callable[float]', a:float, b:float, n:float)-> float:
    """Computes the analitical solution of the integral of f from a to b 
    following the composite Simpson's 1/3 rule. 

    Args:
        f_ (Callable[float]): Function to be integrated  
        a (float): Lower bound of hte interval.
        b (float): Upper bound of the interval.
        n (float): The number of parts the interval is divided into.

    Returns:
        float: Numerical solution of the integral.
    """    
    x = np.linspace(a, b, n+1)
    f = f_(x)
    h = (b - a) / (n)
    
    return (h/3) * ( f[0] + 2*sum(f[2:n-1:2]) + 4*sum(f[1:n:2]) + f[n] )


def euler_explicit(f:'Callable[float, float]', y0:float, t0:float, t:float, h:float)-> np.ndarray:
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

def euler_explicit_midpoint(f:'Callable[float, float]', y0:float, t0:float, t:float, h:float)-> np.ndarray:
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
            u[i+1] = u_previous +2 * h * f(u[i], t_[i])    
        else:
            u[i+1] = u[i-1] + 2 *h * f(u[i], t_[i])
    
    return u