import sys
import warnings
from typing import Callable

import numpy as np

from exceptions import InadequateArgsCombination

# TODO: Shield all methods. 
# TODO: Adequately use logging and warnings. 
# TODO: Implement euler_explicit_systems with explicit dependance on time. 
# TODO: Implement newton for systems.
# TODO: Improve docstrings; implement examples.
# TODO: Implement testcases. 

def newton(err:float, f:'Callable[float]' = None, f_dev:'Callable[float]' = None,
    composite:'Callable[Callable, float, float, float]' = None,  c:float = 0, x0:float = 0, h_err:float = 1e-4) -> float:
    """Newton's method to find roots of a function.
    
    If no `f` is given but `f_dev` and `composite` are, it will compute the roots of the integral of `f_dev` with integration constant c.
    If `f_dev` is not given, it will be computed from `f` with the mathematical definition of a derivative.

    Args:
        err (float): Desired error of the method.
        f_dev (Callable[float], optional): Analytical function to find its roots. Its input is the point to be evaluated in. Defaults to None.
        f_dev (Callable[float], optional): Analytical derivative of the function. Its input is the point to be evaluated in. Defaults to None.
        composite (Callable[Callable, float, float, float], optional): Integration method to compute the integral of `f_dev` and find its roots. 
            It should be `composite_trapezoid` or `composite_simpson` methods. Defaults to None.
        c (float, optional): Integration constant of the integral of f_dev. Defaults to 0.
        x0 (float, optional): Initial guess of the root. 
            Note that an inadequate first guess could lead to undesired outputs such as no roots or undesired roots.
            Defaults to 0.
        h_err (float, optional): Finite approximation of 0 to use in the calculation of `f_dev` by its mathematical definition. Defaults to 1e-4.

    Returns:
        float|None: Root of the function or None if the algorithm reaches its recursion limit.
    """    
    def dev(x:float, f:'Callable[float]' = f, h_err:float = h_err)-> float:
        return ( f(x+h_err) - f(x) ) / h_err 

    if (f or composite) and f_dev:
        if f and composite:
            warnings.warn('`f`, `f_dev` and `composite` args detected. Only `f` and `f_dev` will be used for sake of precision.') 
            iteration = lambda iter_idx, iter_dict: iter_dict[iter_idx] - f(iter_dict[iter_idx]) / f_dev(iter_dict[iter_idx])

        elif composite:
            iteration = lambda iter_idx, iter_dict: iter_dict[iter_idx] - (composite(f_dev, x0, iter_dict[iter_idx], 100_000) + c) / f_dev(iter_dict[iter_idx])

        else:
            iteration = lambda iter_idx, iter_dict: iter_dict[iter_idx] - f(iter_dict[iter_idx]) / f_dev(iter_dict[iter_idx])

    elif f and f_dev == None:
        warnings.warn(f'`f_dev` was not given. It will be computed using the derivative definition with `h`={h_err} .') 
        iteration = lambda iter_idx, iter_dict: iter_dict[iter_idx] - f(iter_dict[iter_idx]) / dev(x=iter_dict[iter_idx], f=f)
    
    else:
        raise InadequateArgsCombination('Cannot compute Newton s method with the combination of arguments given. Check the valid combinations.')

    iter, iter_dict = 0, {0:x0}
    limit = sys.getrecursionlimit()

    while True:
        if iter + 10 >= limit:
            warnings.warn(f'Iteration limit ({limit}) reached without finding any root. Try with other initial guess or changing the recursion limit. Maybe there are no roots.')
            return 
        
        iter_dict[iter+1] = iteration(iter, iter_dict)
        
        if abs(iter_dict[iter+1] - iter_dict[iter]) < err:
            return iter_dict[iter+1]
        
        iter += 1


def bisection(f:'Callable[float, float]', y0:float, t0:float, t:float, h:float) -> np.ndarray:
    raise NotImplementedError()