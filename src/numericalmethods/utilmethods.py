

def per_100_diff(v1: float, v2: float, n_digits: int = 2) -> float:
    """Computes the percentage difference between two values and rounds it to a number of digits.

    Args:
        v1 (float): One value.
        v2 (float): Other value.
        n_digits (int, optional): Number of digits to round the result. Defaults to 2.

    Returns:
        float: rounded percentage difference.
    """
    return round(2 * abs(v1 - v2) / (v1+v2) * 100, n_digits)
