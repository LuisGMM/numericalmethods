

def to_base(base:int, number:int) -> str:


    cociente = number // base
    remainder = number % base
    ans = str(remainder)

    while (cociente >= base):

        remainder = cociente % base
        cociente //= base
        ans = str(remainder) + ans

    else: 
        return str(cociente%base) + ans