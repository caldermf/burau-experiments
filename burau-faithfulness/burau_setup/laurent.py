# Laurent polynomial API and helper functions.
import numpy as np

def dequantize(d, q=1):
    """
    Given a Laurent polynomial in the form of a dictionary, dequantize it.
    Namely, find the value after substituting q to be a number (usually 1).

    INPUT:
    - ``d``: a Laurent polynomial dictionary in the form {deg : coeff}
    - ``q``: a positive real number (default 1)

    OUTPUT:
    - A number, which is the sum of all the coefficients.
    """
    if q == 1:
        return sum(d.values())
    else:
        return sum([(q**k) * v for (k,v) in d.items()])

def degree(d):
    """
    Given a Laurent polynomial in the form of a dictionary, output its (top) degree.

    INPUT:
    - ``d``: a Laurent polynomial dictionary in the form {deg : coeff}

    OUTPUT:
    - An integer or -infinity.
    """
    if len(d) == 0:
        return -np.inf
    else:
        return max(d.keys())

def valuation(d):
    """
    Given a Laurent polynomial in the form of a dictionary, output its valuation or bottom degree.

    INPUT:
    - ``d``: a Laurent polynomial dictionary in the form {deg : coeff}

    OUTPUT:
    - An integer or infinity.
    """
    if len(d) == 0:
        return np.inf
    else:
        return min(d.keys())


def to_hashable(d):
    """
    A hashable representation for Laurent dictionaries, returning an immutable tuple representation of the Laurent polynomial.

    INPUT:
    - ``d``: a Laurent polynomial dictionary of the form {deg : coeff}

    OUTPUT:
    - A tuple (n, c1, c2, ..., ck) where n is the valuation and c1, c2, etc are the coefficients from lowest degree to highest. If the input is the zero polynomial, return the empty tuple.
    """
    if d == {}:
        return ()
    
    d = trim(d)
    d_min, d_max = valuation(d), degree(d)
    output = [d_min]
    for i in range(d_min, d_max+1):
        output.append(d.get(i, 0))
    return tuple(output)
    
def addition(d1, d2):
    """
    Given two Laurent polynomials in the form of dictionaries, sum them.

    INPUT:
    - ``d1``: a Laurent polynomial dictionary of the form {deg : coeff}
    - ``d2``: a Laurent polynomial dictionary of the form {deg : coeff}

    OUTPUT:
    Their sum as a Laurent polynomial dictionary
    """
    return {k : d1.get(k,0) + d2.get(k,0) for k in set(d1) | set(d2)}

def product(d1, d2):
    """
    Given two Laurent polynomials in the form of dictionaries, multiply them.

    INPUT:
    - ``d1``: a Laurent polynomial dictionary of the form {deg : coeff}
    - ``d2``: a Laurent polynomial dictionary of the form {deg : coeff}

    OUTPUT:
    Their product as a Laurent polynomial dictionary
    """
    output = {}
    for k in d1:
        kth_output = {k + j : d1[k] * d2[j] for j in d2}
        output = addition(output, kth_output)
        
    return output

def trim(d):
    """
    Remove all zero entries from a dictionary of a Laurent polynomial.

    INPUT:
    - ``d``: a Laurent polynomial dictionary of the form {deg : coeff}

    OUTPUT:
    The dictionary ``d`` with all zero coefficients removed.
    """
    return {x:y for x,y in d.items() if y != 0}

def qtoqinv(laurentpoldic):
    """
    Given a Laurent polynomial dictionary, applies q-->q^{-1}
    """
    newdic={}
    for key in laurentpoldic :
        newdic[-key]=laurentpoldic[key]
    return newdic

