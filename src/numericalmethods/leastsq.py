
import numpy as np


def leastsq(x: np.ndarray, y: np.ndarray, sigma: float) -> 'tuple[[float, float], [float, float]]':
    """ Computes the least squares of the 1D vectors x and y.
    Raises:
        ValueError: If the lengths of the arrays are not equal.
        ValueError: If the array x is empty. It is checked after the lengths so y would also be empty.
    Returns:
        tuple((float, float), (float, float)): Returns a tuple containing two tuples.
            The first one contains at position 0 the slope (m in literature) and at position 1 its error.
            The second one contains at position 0 the y-intercept (b in literature) and at position 1 its error.
    """
    n = len(x)

    if n != len(y):
        raise ValueError(f'Length of the data array must be equal, length of x is {n} and y is {len(y)}. Please check. ')

    if n == 0:
        raise ValueError('Arrays cannot be empty. Please check.')

    sum_x, sum_y = np.sum(x), np.sum(y)
    sum_x2 = np.sum(x*x)
    sum_xy = np.sum(x*y)

    div = 1 / (sum_x**2 - n*sum_x2)
    m = (sum_x*sum_y - n*sum_xy) * div
    b = (sum_x*sum_xy - sum_y*sum_x2) * div

    m_e = np.sqrt(n*sigma**2 * (-1) * div)
    b_e = np.sqrt(sum_x2*sigma**2 * (-1) * div)

    return (m, m_e), (b, b_e)


if __name__ == '__main__':

    x = eval('[' + input("Introduce x data separated by ,:") + ']')
    y = eval('[' + input("Introduce y data separated by ,:") + ']')

    sigma = float(input("Introduce Sigma: "))

    leastsq = leastsq(np.array(x), np.array(y), sigma)

    print(f'm={leastsq[0][0]}  m_error={leastsq[0][1]} \n b={leastsq[1][0]}  b_error={leastsq[1][1]}')
