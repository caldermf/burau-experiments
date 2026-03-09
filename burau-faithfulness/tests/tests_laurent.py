import sys
sys.path.append("..")

from burau_setup import laurent

def laurent_tests():
    """
    Test suite for Laurent polynomial utilities
    """
    zero = {}
    one = {0:1}
    p1 = {0:1, 1:1, -1:-2, 4:4}
    p2 = {3 : 5}
    p3 = {-3: 1/5}

    def addition_test():
        assert laurent.addition(zero, p1) == p1
        assert laurent.addition(zero, p2) == p2
        return True

    def product_test():
        assert laurent.product(zero, p3) == zero
        assert laurent.product(one, one) == one
        assert laurent.product(one, zero) == zero
        assert laurent.product(one, p2) == p2
        assert laurent.product(p1, p3) == {-3: 1/5, -2: 1/5, -4:-2/5, 1:4/5}
        assert laurent.product(p2, p3) == one
        return True

    def dequantize_test():
        assert laurent.dequantize(zero) == 0
        assert laurent.dequantize(one) == 1
        assert laurent.dequantize(p1) == 4
        assert laurent.dequantize(p2,q=-1) == -5
        return True

    def degree_test():
        assert laurent.degree(zero) == -np.inf
        assert laurent.degree(one) == 0
        return True

    def valuation_test():
        assert laurent.valuation(zero) == np.inf
        assert laurent.valuation(one) == 0
        return True
        
    all_tests = [addition_test, product_test, dequantize_test, degree_test, valuation_test]
    for t in all_tests:
        print("Running test {}".format(t.__name__))
        if t():
            print("... passed")
    
