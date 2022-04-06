
import numpy as np


def newton_horner(x, x_points:list = None, y_points:list = None, coeffs:list = None) -> np.ndarray:
    """ Evaluates the polynomial returned by Horner's algorithm.

    If no coefficients are given this method will compute them

    Args:
        x(float): The point where to evaluate the polynomial.
        x(list(float)): x coordinates of the points.
        y(list(float)): y coordinates of the points.
        coeffs(list(float)): coefficients of the polynomial.

    Returns:
        float: the polynomial evaluated at x.
    
    """
    coeffs_ = coeffs if coeffs is not None else horner_algorithm(x_points, y_points)

    N = len(x_points) - 1
    polynom = coeffs_[N]
    
    for k in range(1,N+1):
        polynom = coeffs_[N-k] + (x -x_points[N-k])*polynom
    
    return polynom
    

def horner_algorithm(x: np.ndarray, y: np.ndarray) -> float:
    """ Computes Newton interpolation polynomial by Horner's algorithm for some given coordinates.

    `x` and `y` must have the same length.

    Args:
        x(np.ndarray): x coordinates of the points.
        y(np.ndarray): y coordinates of the points.

    Raises:
        ValueError: If `x` and `y` are not of the same length. 

    Returns:
        np.ndarray: Polynomial coefficients of the Newton interpolation.

    """
    if len(x) != len(y):
        raise ValueError('x and y must be the same length.')

    LEN = len(y)
    matrix = np.zeros((LEN, LEN))

    for j in range(LEN):
        if j == 0:
            matrix[:, 1] = y

        for i in range(LEN-j):
            matrix[i, j] = (matrix[i+1, j] - matrix[i, j]) / (x[i+1] - x[i])

    return matrix[0]
