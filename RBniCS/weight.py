import numpy as np
import itertools # for linspace sampling
import scipy.stats
from quadrature.QuadratureRule import *

class Weight(object):
    def density(self, mu_range, xi):
        pass

class IndicatorWeight(Weight):
    def __init__(self, original_density, treshold):
        self.original_density = original_density
        self.treshold = treshold
        
    def density(self, mu_range, xi):
        p = self.original_density.density(mu_range, xi)
        for i in range(len(xi)):
            if p[i] < self.treshold:
                p[i] = 0.
            else:
                p[i] = 1. 
        return p

class UniformWeight(Weight):
    
    def density(self, mu_range, xi):
        p = []
        for i in range(len(xi)):
            p_mu = 1.0
            for j in range(len(mu_range)):
                p_mu *= scipy.stats.uniform.pdf(xi[i][j], mu_range[j][0], mu_range[j][1]) 
            p.append(p_mu)
        return p
        
class BetaWeight(Weight):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def density(self, mu_range, xi):
        p = []
        for i in range(len(xi)):
            p_mu = 1.0
            for j in range(len(mu_range)):
                p_mu *= scipy.stats.beta.pdf((xi[i][j]-mu_range[j][0])/(mu_range[j][1]-mu_range[j][0]), self.alpha[j][0], self.alpha[j][1]) 
            p.append(p_mu)
        return p
        
class QuadratureWeight(Weight):
    def __init__(self, original_density, rule, sparse_flag, alpha=0):
        self.original_density = original_density
        self.sparse_flag = sparse_flag
        self.rule = rule
        self.alpha = alpha
        
    def density(self, mu_range, xi):
        p = self.original_density.density(mu_range, xi)
        xi, w, m = QuadratureRule(len(p),self.rule,mu_range,self.sparse_flag,0,param=self.alpha)
        for i in range(len(p)):
            p[i] = p[i]*w[i]
        return p
        
# class NormalWeight(Weight):
# to be implemented

# class OtherWeight(Weight):
# to be implemented
