
import math 

import numpy as np
import scipy.special as sci

# Tested
def forward(order:int, f:'function', x:float, h:float) -> float: 
    
    return sum( [ (-1)**(order-k) * sci.comb(order, k) * f(x+k*h) for k in  range(order+1)] ) / h**2


def backward(order:int, f:'function', x:float, h:float) -> float: 
    raise NotImplementedError()

def central(order:int, f:'function', x:float, h:float) -> float: 
    raise NotImplementedError()



if __name__ == '__main__':
    
    def f(x): return np.e**(- x**2)

    def f_dev2(x): return (-2 + 4*x**2)*f(x)

    def forward_dev2(f, x, h) -> float: return ( f(x) -2*f(x+h) + f(x+2*h) )/ h**2


    print(forward_dev2(f, 1, 1e-4))
    print(forward(2,f,1,1e-4))