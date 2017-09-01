import numpy as np
import math as math
from scipy import special
from ClenshawCurtis_set import *

def d_tuples(d, q):
        
    k = np.ones((1,d))
    khat = (q-d+1)*k
    ind = np.zeros((1,d))
    
    p = 0
    while k[0,d-1] <= q:
        k[0,p] = k[0,p]+1
        if k[0,p] > khat[0,p]:
            if p != d-1:
                k[0,p] = 1
                p = p+1
        else:
            for j in range(p):
                khat[0,j] = khat[0,p]-k[0,p]+1
            k[0,0] = khat[0,0]
            p = 0
            ind = np.concatenate((ind,k))
    
    n = ind.shape[0]
    ind = ind[1:n,:]
    n = ind.shape[0]
        
    return ind, n

def combvec(m,v): # m matrix, v row vector
    
    n1 = m.shape[0]
    a = np.zeros((n1+1,m.shape[1]*v.shape[1]))
    for j in range(n1):
        for i in range(m.shape[1]):
            for k in range(v.shape[1]):
                a[j,k*m.shape[1]+i] = m[j,i]
    for i in range(m.shape[1]):
        for j in range(v.shape[1]):
            a[n1,j*m.shape[1]+i] = v[0,j]        
    
    return a    
    
def univariate_rule(n,rule,alpha=0.,beta=0.): # univariate rule for interval [0,1] 
    
    if rule == 'Linspace':
        weights = np.ones((1,n))
        if n == 1:
            nodes = np.array([[0.5]])
        else:
            nodes = np.array([np.linspace(0,1,n)])
            weights = (1./n)*weights # legendre weights instead?
    elif rule == 'ClenshawCurtis':
        nodes, weights = ClenshawCurtis_set(n) 
        nodes = nodes*0.5 + 0.5
        weights = weights*0.5
    elif rule == 'ClenshawCurtis_nested':
        if n == 1:
            nodes, weights = ClenshawCurtis_set(1)
        else:
            nodes, weights = ClenshawCurtis_set(int(2**(n-1)+1)) 
        nodes = nodes*0.5 + 0.5
        weights = weights*0.5
    elif rule == 'GaussLegendre':
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = np.array([nodes])*0.5 + 0.5
        weights = np.array([weights])*0.5
    elif rule == 'GaussJacobi':            
        nodes, weights = special.j_roots(n,alpha,beta)
        nodes = np.array([nodes])*0.5 + 0.5
        weights = np.array([weights])*0.5
    
    return nodes, weights

def binom_coeff(n,k):
    
    return math.factorial(n)/(math.factorial(n-k)*math.factorial(k))

def no_repetitions(A,axis): # axis can be 0 or 1
    
    if axis == 1:
        n = A.shape[1]
        m = A.shape[0]
        A_reduced = np.zeros((m,1))
        A_reduced[:,0] = A[:,0]
        n_reduced = 1
        count = [[0]]
        if n>1:
            for k in range(1,n):
                control = 1
                for i in range(n_reduced):
                    if np.array_equal(A_reduced[:,i],A[:,k]):
                        count[i].append(k)
                        control = 0
                if control:
                    A_reduced = np.concatenate((A_reduced,np.reshape(A[:,k],(m,1))),1)
                    n_reduced = n_reduced + 1
                    count.append([k])
    else:
        A_reduced, count, n_reduced = no_repetitions(A.T,1)
        A_reduced = A_reduced.T
    
    return A_reduced, count, n_reduced
    
def weight_compress(v,indices,count):

    w = np.zeros((1,count))
    for i in range(count):
        for j in range(len(indices[i])):
            w[0,i] = w[0,i]+v[0,int(indices[i][j])]
    w = np.ndarray.tolist(w)[0]   
                
    return w                

def array2tuple(m,axis): # axis can be 0 or 1
    
    if axis == 1:
        v = []
        for j in range(m.shape[1]):
            v.append(tuple(np.ndarray.tolist(m[:,j])))
    else:
        v = array2tuple(m.T,1) 
    
    return v

