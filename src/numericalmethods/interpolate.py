
import numpy as np


def newton_horner(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    if len(x)!=len(y): 
        raise ValueError('x and y must be the same length.')

    LEN = len(y)
    matrix = np.zeros((LEN, LEN))

    # matrix column -> j
    for j in range(LEN):
        if j == 0:
            matrix[:, 1] = y

        # matrix row -> i
        for i in range(LEN-j):

            matrix[i, j] = (matrix[i+1, j] - matrix[i, j]) / (x[i+1] - x[i])

    return matrix[0]
