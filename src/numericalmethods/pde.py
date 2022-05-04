
import numpy as np


def parabollic(theta, h: float, k: float, x0: float, xf: float, tf: float, u0: np.ndarray) -> np.ndarray:
    
    sigma = k/h**2

    x = np.arange(x0, xf+h, h)
    t = np.arange(0, tf+h, h)

    LEN_X = len(x)
    LEN_T = len(t)
