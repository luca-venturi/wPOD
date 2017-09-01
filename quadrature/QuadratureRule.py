# INPUT:
#			- n = maximum number of nodes, or, if order_flag != 0, order of the quadrature rule
#			- rule = string with the name of the rule {'Uniform','ClenshawCurtis','ClenshawCurtis_nested','GaussLegendre','GaussJacobi'}
#			- bounds = (d,2) array with bounds of the intervals
#			- sparse_flag = 0 for full tensor product rule, 1 for Smolyak rule
#			- print_flag != 0 -> show plot the interpolation points if d == 2
#			- param = eventual parameters associated with the rule
# OUTPUT:
#			- xi = nodes of the rule
#			- w = weights of the rule (ordered as xi)
#			- m = |xi|
#			- q = order of the rule

import numpy as np
from ausiliar_functions import *
import matplotlib.pyplot as plt
from sys import exit

def QuadratureRule(n,rule,bounds,order_flag=0,sparse_flag=0,print_flag=0,param=0):
	if sparse_flag == 1:
		if order_flag == 0:
			q = len(bounds)-1
			m = 0
			while m < n:
				q = q+1
				xi, w, m = SmolyakRule(len(bounds),q,rule,bounds,a=param)
			q = q -1
		else:
			xi, w, m = SmolyakRule(len(bounds),n+1,rule,bounds,a=param)
			q = n
	else:
		if order_flag == 0:
			n_d_root = int(np.ceil(n**(1./len(bounds))))
			l = [n_d_root for i in range(len(bounds))]
			xi, w, m = TensorProductRule(len(bounds),l,rule,bounds,a=param)
		else:
			l = [n for i in range(len(bounds))]
			xi, w, m = TensorProductRule(len(bounds),l,rule,bounds,a=param)
		q = l[0]

	if print_flag != 0:    
		if len(bounds) == 2:
			plt.plot(xi[0,:],xi[1,:],'.b',linestyle='None')
			plt.gca().set_aspect('equal', adjustable='box')
			plt.tight_layout()
			plt.show()

	xi = array2tuple(xi,1)	

	return xi, w, m, q

def SmolyakRule(d,q,rule,bounds,a=0):  

    param = []
    if a == 0:
        for i in range(d):
            param.append([0.,0.])
    else:
        param = a 
    
    bounds = np.array(bounds)
    nodes = np.zeros((d,1))
    weights = np.zeros((1,1))
    for l in range(max(d,q-d+1),q+1):
        
        if l == d:
            tmpnodes = np.zeros((d,1))
            tmpweights = np.ones((1,1))
            for i in range(d):
                tmp1, tmp2 = univariate_rule(1,rule,alpha=param[i][0],beta=param[i][1])
                tmpnodes[i,0] = (bounds[i,1]-bounds[i,0])*tmp1[0,0] + bounds[i,0]
                tmpweights[0,0] = (bounds[i,1]-bounds[i,0])*tmp2[0,0]*tmpweights[0,0]
            tmpweights[0,0] = (-1)**(q-l)*binom_coeff(d-1,q-l)*tmpweights[0,0]
            
            nodes = np.concatenate((nodes,tmpnodes),axis=1)
            weights = np.concatenate((weights,tmpweights),axis=1)
            
        else:
            
            ind, m = d_tuples(d,l)
            for i in range(m):
                
                gamma = ind[i,:]
                tmpnodes, tmpweights = univariate_rule(int(gamma[0]),rule,alpha=param[0][0],beta=param[0][1])         
                for j in range(1,d):
                    tmp1, tmp2 = univariate_rule(int(gamma[j]),rule,alpha=param[j][0],beta=param[j][1])
                    
                    tmpnodes = combvec(tmpnodes,tmp1)
                    tmpweights = combvec(tmpweights,tmp2)
                
                for j in range(d):
                    for k in range(tmpnodes.shape[1]):
                        tmpnodes[j,k] = (bounds[j,1]-bounds[j,0])*tmpnodes[j,k] + bounds[j,0]
                        tmpweights[j,k] = (bounds[j,1]-bounds[j,0])*tmpweights[j,k]    
                
                tmpweights = (-1)**(q-l)*binom_coeff(d-1,q-l)*np.array([np.prod(tmpweights, axis=0)])
                
                nodes = np.concatenate((nodes,tmpnodes),axis=1)
                weights = np.concatenate((weights,tmpweights),axis=1)            
    
    nodes = nodes[:,1:]
    weights = np.array([weights[0,1:]])
    
    count = 1
    if d != q:                
        nodes, indices, count = no_repetitions(nodes,1)
        weights = weight_compress(weights,indices,count)
    
    return nodes, weights, count

def TensorProductRule(d,n,rule,bounds,a=0):  
    
    param = []
    if a == 0:
        for i in range(d):
            param.append([0.,0.])
    else:
        param = a
    
    bounds = np.array(bounds)
    
    tmpnodes, tmpweights = univariate_rule(int(n[0]),rule,alpha=param[0][0],beta=param[0][1])         
    for j in range(1,d):
        tmp1, tmp2 = univariate_rule(int(n[j]),rule,alpha=param[j][0],beta=param[j][1])
                        
        tmpnodes = combvec(tmpnodes,tmp1)
        tmpweights = combvec(tmpweights,tmp2)
    
    nodes = np.zeros((d,tmpnodes.shape[1]))
    for j in range(d):
        for k in range(tmpnodes.shape[1]):
            nodes[j,k] = (bounds[j,1]-bounds[j,0])*tmpnodes[j,k] + bounds[j,0]
            tmpweights[j,k] = (bounds[j,1]-bounds[j,0])*tmpweights[j,k]    
                
    weights = np.array([np.prod(tmpweights, axis=0)])
    count = nodes.shape[1]
    weights = np.ndarray.tolist(weights[0,:])
    
    return nodes, weights, count

