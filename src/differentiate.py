

import scipy.special as sci

# Tested
def forward(order:int, f:'function', x:float, h:float, exact:bool = False) -> float:     
    return sum( [ (-1)**(order-k) * sci.comb(order, k, exact) * f(x + k*h) for k in  range(order+1)] ) / h**2


def backward(order:int, f:'function', x:float, h:float, exact:bool = False) -> float: 
    return sum( [ (-1)**(k) * sci.comb(order, k, exact) * f(x - k*h) for k in  range(order+1)] ) / h**2


def central(order:int, f:'function', x:float, h:float) -> float: 
    return sum( [ (-1)**(k) * sci.comb(order, k) * f(x - (order/2 - k)*h) for k in  range(order+1)] ) / h**2



if __name__ == '__main__':
    pass