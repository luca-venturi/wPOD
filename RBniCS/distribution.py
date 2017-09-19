import numpy as np
import itertools # for linspace sampling
from quadrature.QuadratureRule import *

class Distribution(object):
    def sample(self, mu_range, n):
        pass

class QuadratureDistribution(Distribution):
    def __init__(self, rule, sparse_flag, alpha=0):
        self.sparse_flag = sparse_flag # if 1 it returns a sparse grid, if 0 a tensor product grid
        self.rule = rule # rule can be 'Uniform', 'ClenshawCurtis', ClenshawCurtis_nested', 'GaussLegendre'; 
                         # other rules to be implemented in /quadrature/ausiliar_functions
        self.alpha = alpha
    def sample(self, mu_range, n):
        xi, w, m = QuadratureRule(n,self.rule,mu_range,self.sparse_flag,0,param=self.alpha)
        return xi

class UniformDistribution(Distribution):
    def sample(self, mu_range, n):
        xi = []
        w = []
        for i in range(n):
            mu = []
            for p in range(len(mu_range)):
                mu.append(np.random.uniform(mu_range[p][0], mu_range[p][1]))
            xi.append(tuple(mu))
            # w.append(1./n)
            w.append(1.)
        return xi
        
class BetaDistribution(Distribution):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def sample(self, mu_range, n):
        xi = []
        w = []
        for i in range(n):
            mu = []
            for p in range(len(mu_range)):
                mu.append(mu_range[p][0] + np.random.beta(self.alpha[p][0], self.alpha[p][1])*(mu_range[p][1]-mu_range[p][0]))
            xi.append(tuple(mu))
            # w.append(1./n)
            w.append(1.)
        return xi
        
# class NormalDistribution(Distribution):
# to be implemented

# class OtherDistribution(Distribution):
# to be implemented
