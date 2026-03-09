import sys
sys.path.append("..")

from burau_setup import dimvec
import numpy as np

def dimvec_tests():
    """
    Test suite for dimension vector utilities.
    """
    p1 = {0:1, 1:1, -1:-2, 4:4}
    p2 = {3 : 5}
    p3 = {-3: 1/5}
    zero = {}
    one = {0:1}

    d1 = [p1, p2, p3]
    d2 = [p1, zero]
    d3 = [p3, zero]
    d4 = [zero]
    d5 = [zero, one, p3]

    def normalize_test():
        assert dimvec.normalize(d1) == ({2: -2, 3: 1, 4: 1, 7: 4}, {6: 5}, {0: 0.2})
        assert dimvec.normalize(d2) == ({0: -2, 1: 1, 2: 1, 5: 4}, {})
        return True

    def top_degree_test():
        assert dimvec.top_degree(d1) == 4
        assert dimvec.top_degree(d2) == 4
        assert dimvec.top_degree(d3) == -3
        assert dimvec.top_degree(d4) == -np.inf
        assert dimvec.top_degree(d5) == 0
        return True

    all_tests = [normalize_test, top_degree_test]
    for t in all_tests:
        print("Running test {}".format(t.__name__))
        if t():
            print("... passed")

    return 
        

    
