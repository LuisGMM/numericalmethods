
import numpy as np

from charliepy.integrate import composite_simpson, composite_trapezoid



def test_composite_trapezoid(): 
    ans = 1.4887
    f = lambda x: np.exp(-x**2)
    
    assert round( composite_trapezoid(f_=f, a=-1, b=1, n=10), 4) == ans

def composite_simpson():
    ans = 1.49367
    f = lambda x: np.exp(-x**2)
    
    assert round( composite_simpson(f_=f, a=-1, b=1, n=10), 5) == ans