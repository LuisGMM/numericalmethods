

def to_base(base:int, number:int) -> str:
    """ Changes an integer from decimal base to other base. 

    Args:
        base (int): New base of the number to be converted.
        number (int): Number to be converted to the new base.

    Returns:
        str: A string representation of the number in the new base.
    """
    quotient = number // base
    remainder = number % base
    ans = str(remainder)

    while (quotient >= base):

        remainder = quotient % base
        quotient //= base
        ans = str(remainder) + ans

    else: 
        return str(quotient%base) + ans