
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
    pass