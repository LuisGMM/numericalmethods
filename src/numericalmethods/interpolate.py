
import numpy as np

from numericalmethods.exceptions import InadequateArgsCombination


def newton_horner(x, x_points: np.ndarray, y_points: np.ndarray = None, coeffs: np.ndarray = None) -> np.ndarray:
    """ Evaluates the polynomial returned by Horner's algorithm.

    The user must give `y_points` or `coeffs`. If `coeffs` is not given this method will compute them with `x_points` and `y_points`.

    Args:
        x(float): The point where to evaluate the polynomial.
        x(np.ndarray): `x` coordinates of the points.
        y(np.ndarray): `y` coordinates of the points. Defaults to None.
        coeffs(np.ndarray): coefficients of the polynomial. Defaults to None.

    Raises:
        InadequateArgsCombination: If the combination  of arguments is not valid.
        ValueError: If `x` and `y` are not of the same length.

    Returns:
        float: the polynomial evaluated at x.

    """
    if y_points is None and coeffs is None:
        raise InadequateArgsCombination('Cannot evaluate Newton\'s polynomial with the combination of arguments given. Check the valid combinations.')

    if y_points is not None:
        if len(x_points) != len(y_points):
            raise ValueError('`x_points` and `y_points` must be the same length.')

    coeffs_ = coeffs if coeffs is not None else horner_algorithm(x_points, y_points)

    N = len(x_points) - 1
    polynom = coeffs_[N]

    for k in range(1, N+1):
        polynom = coeffs_[N-k] + (x - x_points[N-k])*polynom

    return polynom


def horner_algorithm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
