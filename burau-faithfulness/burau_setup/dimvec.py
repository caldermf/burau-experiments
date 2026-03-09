from . import laurent
import numpy as np

# Utilities to manipulate dimension vectors.
def normalize(dim_vec):
    """
    Function to normalize a dimension vector of Laurent polynomial dictionaries.
    The normalized dimension vector has its lowest degree q-polynomial set at degree zero, with leading coefficient positive.

    INPUT:
    - ``dim_vec``, a dimension vector whose entries are Laurent polynomial dictionaries. So each entry is a dictionary of the form {deg : coeff}.

    OUTPUT:
    - A normalized dimension vector as explained above. Returned as a list.
    """
    bottom_deg, min_index = np.inf, None
    new_deg = np.inf

    for k in range(len(dim_vec)):
        if dim_vec[k] == {}:
            new_valuation = np.inf
        else:
            new_valuation = min(dim_vec[k].keys())

        if new_valuation < bottom_deg:
            bottom_deg, min_index = new_valuation, k

    if min_index is not None:
        leading_coeff_index = max(dim_vec[min_index].keys())
        leading_coeff = dim_vec[min_index][leading_coeff_index]
        sgn = 1 if leading_coeff >= 0 else -1
        q_factor = {-bottom_deg : sgn}

        return [laurent.product(q_factor, d) for d in dim_vec]

    else:
        return dim_vec


def top_degree(dim_vec):
    """
    Given a dimension vector with entries Laurent polynomials dictionaries, returns the top degree.
    
    INPUT:
    - ``dim_vec`` a dimension vector
    
    OUTPUT:
    - Either an integer or -infinity.
    """
    return max([laurent.degree(x) for x in dim_vec] + [-np.inf])

def num_terms(dim_vec):
    """
    Given a dimension vector with entries Laurent polynomials dictionaries, returns the number of "terms" in it.
    
    INPUT:
    - ``dim_vec`` a dimension vector
    
    OUTPUT:
    - an array that represents the associated root in terms of coordinates in the ai's.
    """

    # WARNING: currently buggy because unsure if it should only return integers.
    return sum([sum([int(abs(v)) for v in d.values()]) for d in dim_vec])

# Cast to native python types to help convert to/from json.
def jsonify(dim_vec):
    new_dim_vec = []
    for d in dim_vec:
        new_d = {int(k):v for (k,v) in d.items()}
        new_dim_vec.append(new_d)
    return new_dim_vec
    
