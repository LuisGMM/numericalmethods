
import numpy as np


def newton_horner(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Computes Newton interpolation polynomial by Horner's algorithm for some given coordinates.

    `x` and `y` must have the same length.

    Args:
        x(np.ndarray): x coordinates of the points.
        y(np.ndarray): y coordinates of the points.

    Raises:
        ValueError: If `x` and `y` are not of the same length. 

    Returns:
        list: with the polynomial coefficients of the Newton interpolation.

    """
    if len(x) != len(y):
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
