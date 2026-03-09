import numpy as np
from . import laurent
import math
from functools import partial

def make_coxeter(digraph) :
    """
    Converts a directed graph into a Coxeter matrix.
    """
    loclist=[]
    for x in digraph.keys() :
        locvect=[]
        for y in digraph.keys() :
            if y==x :
                locvect.append(1)
            elif y in digraph[x] :
                locvect.append(3)
            else :
                locvect.append(2)
        loclist.append(locvect)
    return np.array(loclist)

def pair(x,y, digraph) :
    """Returns 2 if x==y, -1 if y--y, 0 if x and y are not adjacent"""
    if x==y :
        return 2
    if y in digraph[x] :
        return -1
    else :
        return 0

#Simple roots
def make_simple_roots(digraph):
    """
    Return a list of simple roots of the given directed graph.

    Each simple root is formatted as a row vector of the form [0, ...,
    0, 1, 0, ..., 0], where the length of the vector is the number of
    vertices in the graph.
    """
    alpha=[]
    for i in range(len(digraph.keys())) :
        locvect=[]
        for j in range(len(digraph.keys())) :
            if i != j :
                locvect.append(0)
            else :
                locvect.append(1)
        alpha.append(np.array(locvect))
    return alpha

def is_root_in(root,listofroots) :
    for root2 in listofroots :
        if np.all(root==root2) :
            return True
    return False

class DigraphData:
    """
    A class that stores various data about a given directed graph.
    """
    def __init__(self, digraph, frozen=[]):
        self.digraph = digraph
        self.positive_letters = list(digraph.keys())
        self.negative_letters = [-x for x in self.positive_letters]
        self.all_letters = self.positive_letters + self.negative_letters
        self.restricted_letters = [x for x in self.all_letters if (x not in frozen and x not in [-y for y in frozen])]

        self.coxeter_matrix = make_coxeter(digraph)
        self.coxeter_pairing = np.array([[pair(x,y, digraph) for y in digraph.keys()] for x in digraph.keys()])
        self.simple_roots = make_simple_roots(digraph)
        self.dim_vectors = {i : self.make_pi_dim_vector(i) for i in self.positive_letters}
        self.burau_fns = {i : partial(self._burau_fn_i, i=i) for i in self.all_letters}
        self.roots_mod_2 = self._all_roots_mod_2()

    def __repr__(self):
        return "Braid, Coxeter, and Burau data for the digraph {}".format(self.digraph)

    # Careful, the function below is coded with nodes labeled from 1. Later in some of the dictionaries the entries start from 0
    def exists_digraph_edge(self,i,j):
        return j in self.digraph.get(i, set())

    def pairing(self, dim_vec1, dim_vec2):
        """
        Given two dimension vectors with entries Laurent polynomials dictionaries, returns the pairing.
        
        INPUT:
        - dim_vec1, dim_vec2: 2 dimension vectors
        
        OUTPUT:
        - a Laurent polynomial dictionary that gives the q-pairing
        """
        total_sum = {}
        for i in range(len(dim_vec1)):
            fi = laurent.qtoqinv(dim_vec1[i])
            tmp_sum = {}
            for j in range(len(dim_vec2)):
                gj = dim_vec2[j]
                if j == i:
                    # Add (1+q^2) * gj to the intermediate sum if j = i
                    tmp_sum = laurent.addition(tmp_sum, laurent.product({0:1,2:1}, gj))
                elif self.exists_digraph_edge(i+1,j+1):
                    # Add q * gj to the intermediate sum if j - i is an edge
                    tmp_sum = laurent.addition(tmp_sum, laurent.product({1:1}, gj))
            total_sum = laurent.addition(total_sum, laurent.product(fi, tmp_sum))
        return laurent.trim(total_sum)

    def make_pi_dim_vector(self, i):
        """
        Helper function to create the dimension vector (with Laurent polynomial dictionary entries) of the ith indecomposable projective Pi.
        
        INPUT:
        - ``i`` a positive letter of the chosen Cartan type.
        
        OUTPUT:
        - A list whose ith entry is the dictionary {0:1} and all other entries are {}.
        This corresponds to the fact that the dimension vector of Pi is 1 in the ith coordinate and zero in all other coordinates.
        """
        output_vec = [{} for _ in range(len(self.digraph))]
        output_vec[i - 1] = {0 : 1}
        return output_vec

    def _burau_fn_i(self, input_vec, i):
        """
        Given an input vector of Laurent polynomial dictionaries, returns an output vector consisting of the ith Burau matrix applied to it.

        Note that only the entry at coordinate ``index`` of the input vector changes in this situation, so that is the only one we update.
        """
        # Record the sign of i; the Burau matrix depends on whether we have a letter or an inverse letter.
        i_sgn = -1 if i < 0 else 1
        
        # Matrices are zero-indexed, hence the following.
        index = abs(i) - 1

        updated_coordinate = {}
        for k in range(0, len(input_vec)):
            kth_summand = {}
            
            if k == index:
                # In this case, the [index,k]th entry of the Burau matrix is precisely
                # -q^(-2) if i > 0 and q^2 if i < 0.
                kth_summand = laurent.product({-2*i_sgn : -1}, input_vec[k])

            elif self.coxeter_matrix[index, k] > 2:
                # In this case, the [index, k]th entry of the Burau matrix is precisely
                # -q^(-1) if i > 0 and -q if i < 0.
                kth_summand = laurent.product({-1*i_sgn : -1}, input_vec[k])
            else:
                # In all other cases, the [index, k]th entry of the Burau matrix is zero.
                # We represent this by an empty dictionary.
                kth_summand = {}

            updated_coordinate = laurent.trim(laurent.addition(updated_coordinate, kth_summand))

        output_vec = input_vec.copy()
        output_vec[index] = updated_coordinate
        return output_vec

    # For keys I'll take coefficients in Z/2
    def makekey(self,root,nb):
        """
        Helper function to convert a root into a key for tabulation.
        """
        #rootmod2=[x-2*math.floor(x/2) for x in root]
        rootmod2=[x % 2 for x in root]
        key_tuple = tuple(rootmod2) + (nb,)
        return '-'.join([str(x) for x in key_tuple])

    # Producing the list of roots mod 2
    def _all_roots_mod_2(self):
        roots_mod_2=self.simple_roots
        prev=0
        cur=len(roots_mod_2)
        while cur>prev :
            toadd=[]
            for roo in roots_mod_2 :
                for ind in self.positive_letters :
                    newroo=roo-np.dot(np.dot(np.transpose(roo),self.coxeter_pairing),self.simple_roots[ind-1])*self.simple_roots[ind-1]
                    newroomod2=[x-2*math.floor(x/2) for x in newroo]
                    if is_root_in(newroomod2, roots_mod_2)==False :
                        toadd.append(newroomod2)
            roots_mod_2 = roots_mod_2+toadd
            prev=cur
            cur=len(roots_mod_2)
            return roots_mod_2

    def find_ends_vector(self, dim_vec):
        """
        Given a dimension vector with entries Laurent polynomials dictionaries, returns the associated root.
        
        INPUT:
        - ``dim_vec`` a dimension vector
        
        OUTPUT:
        - an array that represents the associated root in terms of coordinates in the ai's.
        
        """
        
        # WARNING: currently buggy because "alpha" is undefined. Move elsewhere.
        dequantized_dim_vec=[laurent.dequantize(d) for d in dim_vec]
        return sum([abs(dequantized_dim_vec[i])*self.simple_roots[i] for i in range(len(self.simple_roots))])

    def find_ends_vector_mod2(self, dim_vec):
        """
        Given a dimension vector with entries Laurent polynomials dictionaries, returns the associated root with mod 2 coefficients.
        
        INPUT:
        - ``dim_vec`` a dimension vector
        OUTPUT:
        - an array that represents the associated root in terms of coordinates in the ai's with mod 2 coefficients.
        """
        
        # WARNING: currently buggy because "alpha" is undefined. Move elsewhere.
        dequantized_dim_vec=[laurent.dequantize(d) for d in dim_vec]
        #return sum([mod2(abs(dequantized_dim_vec[i]))*self.simple_roots[i] for i in range(len(self.simple_roots))])
        return sum([(abs(dequantized_dim_vec[i]) % 2)*self.simple_roots[i] for i in range(len(self.simple_roots))])



