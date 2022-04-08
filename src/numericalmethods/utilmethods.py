

def per_100_diff(v1: float, v2: float, n_digits: int = 2) -> float:

    return round(2 * abs(v1- v2)/ (v1+v2) *100, n_digits)