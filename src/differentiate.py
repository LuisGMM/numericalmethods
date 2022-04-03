
import numpy as np
import matplotlib.pyplot as plt



def dev2(f, x, h) -> float: return ( f(x-h) -2*f(x) + f(x+h) )/ h**2

def forward_dev2(f, x, h) -> float: return ( f(x) -2*f(x+h) + f(x+2*h) )/ h**2

def backward_dev2(f, x, h) -> float: return ( f(x-2*h) -2*f(x-h) + f(x) )/ h**2


def error(f_dev2, approx_dev2, f, x, h): return abs( f_dev2(x) - approx_dev2(f, x, h) )


def array_forward_dev2(x, fx): 

    ans = []

    for x, fx in zip(x, fx): ans.append(forward_dev2)


if __name__ == '__main__':
    
    def f(x): return np.e**(- x**2)

    def f_dev2(x): return (-2 + 4*x**2)*f(x)

    h = np.logspace(0,-6,7)


    for method in (dev2, forward_dev2, backward_dev2):

        for x in (0,1): 
            [print(f'Ans: {method(f, x, h_i)},   Error: {error(f_dev2, method, f, x, h_i)} ') for h_i in h]

        print('\n')

        plt.plot(h, error(f_dev2, method, f, 1, h))
        plt.xscale('log')
        plt.yscale('log')
        plt.show()



