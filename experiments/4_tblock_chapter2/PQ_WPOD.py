from dolfin import *
from RBniCS import *
import matplotlib.pyplot as plt

class Tblock(EllipticCoercivePODBase):

    def __init__(self, V, subd, bound):
        bc_list = [
            DirichletBC(V, 0.0, bound, 1),
            DirichletBC(V, 0.0, bound, 2),
            DirichletBC(V, 0.0, bound, 3),
            DirichletBC(V, 0.0, bound, 4),
            DirichletBC(V, 0.0, bound, 5),
            DirichletBC(V, 0.0, bound, 6),
            DirichletBC(V, 0.0, bound, 7),
            DirichletBC(V, 0.0, bound, 8)
        ]
        super(Tblock, self).__init__(V, bc_list)
        self.dx = Measure("dx")(subdomain_data=subd)
        self.ds = Measure("ds")(subdomain_data=bound)
        # Use the H^1 seminorm on V as norm, instead of the H^1 norm
        u = self.u
        v = self.v
        dx = self.dx
        scalar = inner(grad(u),grad(v))*dx
        self.S = assemble(scalar)
        [bc.apply(self.S) for bc in self.bc_list] # make sure to apply BCs to the inner product matrix
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        return min(self.compute_theta_a())
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        mu1 = self.mu[0]
    	mu2 = self.mu[1]
    	mu3 = self.mu[2]
    	mu4 = self.mu[3]
        theta_a0 = mu1
        theta_a1 = mu2
    	theta_a2 = mu3
        theta_a3 = mu4
        return (theta_a0, theta_a1, theta_a2, theta_a3)
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        return (1.0,)
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        u = self.u
        v = self.v
        dx = self.dx
        # Assemble A0
        a0 = inner(grad(u),grad(v))*dx(1) + 1e-15*inner(u,v)*dx
        A0 = assemble(a0)
        # Assemble A1
        a1 = inner(grad(u),grad(v))*dx(2) + 1e-15*inner(u,v)*dx
        A1 = assemble(a1)
    	# Assemble A2
        a2 = inner(grad(u),grad(v))*dx(3) + 1e-15*inner(u,v)*dx
        A2 = assemble(a2)
        # Assemble A3
        a3 = inner(grad(u),grad(v))*dx(4) + 1e-15*inner(u,v)*dx
        A3 = assemble(a3)
        # Return
        return (A0, A1, A2, A3)
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        v = self.v
        dx = self.dx
        ds = self.ds
        # Assemble F0
        f0 = v*dx
        F0 = assemble(f0)
        # Return
        return (F0,)


#~~~~~~~~~~~~~~~~~~~~~~~~~     MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 0.

Nmax = 10 
mean_error_u = np.zeros((Nmax,6))
mu_range = [(1.0, 3.0),(1.0, 3.0),(1.0, 3.0),(1.0, 3.0)]

mesh = Mesh("Data/4_tblock.xml")
subd = MeshFunction("size_t", mesh, "Data/4_tblock_physical_region.xml")
bound = MeshFunction("size_t", mesh, "Data/4_tblock_facet_region.xml")

V = FunctionSpace(mesh, "Lagrange", 1)

parameters.linear_algebra_backend = 'PETSc'

N_xi_train = 100
N_xi_test = 500

al = 10
be = al - 1.
param = [(al,al),(al,al),(al,al),(al,al),(al,al),(al,al),(al,al),(al,al),(al,al)]
Qparam = [(be,be),(be,be),(be,be),(be,be),(be,be),(be,be),(be,be),(be,be),(be,be)]
originalDistribution = BetaDistribution(param)
originalWeight = BetaWeight(param) 
'''
# 1.3
tb_3 = Tblock(V, subd, bound)

tb_3.setmu_range(mu_range)
tb_3.setNmax(Nmax)

distribution = QuadratureDistribution('ClenshawCurtis',0)
density = QuadratureWeight(originalWeight,'ClenshawCurtis',0)

tb_3.setxi_train(2000, sampling=distribution)
tb_3.set_density(weight=density)
tb_3.set_weighted_flag(1)

tb_3.offline() 

tb_3.setxi_test(N_xi_test, enable_import=True, sampling=originalDistribution)
mean_error_u[:,3] = tb_3.error_analysis()
'''
# 1.4
tb_4 = Tblock(V, subd, bound)

tb_4.setmu_range(mu_range)
tb_4.setNmax(Nmax)

distribution = QuadratureDistribution('GaussJacobi',0,alpha=param)
density = QuadratureWeight(IndicatorWeight(originalWeight,-1e8),'GaussJacobi',0,alpha=Qparam)

tb_4.setxi_train(500, sampling=distribution)
tb_4.set_density(weight=density)
tb_4.set_weighted_flag(1)

tb_4.offline() 

tb_4.setxi_test(N_xi_test, enable_import=True, sampling=originalDistribution)
mean_error_u[:,4] = tb_4.error_analysis()

# 1.5
tb_5 = Tblock(V, subd, bound)

tb_5.setmu_range(mu_range)
tb_5.setNmax(Nmax)

distribution = QuadratureDistribution('GaussJacobi',1,alpha=param)
density = QuadratureWeight(IndicatorWeight(originalWeight,-1e8),'GaussJacobi',1,alpha=Qparam)

tb_5.setxi_train(500, sampling=distribution)
tb_5.set_density(weight=density)
tb_5.set_weighted_flag(1)

tb_5.offline() 

tb_5.setxi_test(N_xi_test, enable_import=True, sampling=originalDistribution)
mean_error_u[:,5] = tb_5.error_analysis()

# 1.6
tb_6 = Tblock(V, subd, bound)

tb_6.setmu_range(mu_range)
tb_6.setNmax(Nmax)

distribution = QuadratureDistribution('GaussLegendre',0,alpha=param)
density = QuadratureWeight(IndicatorWeight(originalWeight,-1e8),'GaussJacobi',0,alpha=Qparam)

tb_6.setxi_train(500, sampling=distribution)
tb_6.set_density(weight=density)
tb_6.set_weighted_flag(1)

tb_6.offline() 

tb_6.setxi_test(N_xi_test, enable_import=True, sampling=originalDistribution)
mean_error_u[:,6] = tb_6.error_analysis()

# 1.7
tb_7 = Tblock(V, subd, bound)

tb_7.setmu_range(mu_range)
tb_7.setNmax(Nmax)

distribution = QuadratureDistribution('GaussLegendre',1)
density = QuadratureWeight(originalWeight,'GaussLegendre',1)

tb_7.setxi_train(500, sampling=distribution)
tb_7.set_density(weight=density)
tb_7.set_weighted_flag(1)

tb_7.offline() 

tb_7.setxi_test(N_xi_test, enable_import=True, sampling=originalDistribution)
mean_error_u[:,7] = tb_7.error_analysis()

# 7. Plot the errors

#plt.plot(np.log10(mean_error_u[:,3]),'r',label='QWPOD - ClenshawCurtis')
plt.plot(np.log10(mean_error_u[:,6]),'k-+',label='GaussLegendre POD')
plt.plot(np.log10(mean_error_u[:,4]),'y-*',label='Sparse GaussJacobi POD')
plt.plot(np.log10(mean_error_u[:,7]),'r',label='Sparse GaussLegendre POD')
plt.plot(np.log10(mean_error_u[:,5]),'y-*',label='Sparse GaussJacobi POD')
#plt.title('error comparison: first trial')
plt.title('error comparison: second trial')
plt.xlabel('$N$')
plt.ylabel('$\log_{10}(E[\parallel u_{N_\delta}(Y)-u_N(Y)\parallel_{H^1_0(\Omega)}^2])$')
plt.legend()
plt.show()

# 8. Returns the best method #
'''
mean_N_err = np.mean(np.log10(mean_error_u[:,3:6]), axis=0) #
m = min(mean_N_err) #
print [i for i in range(3,6) if mean_N_err[i-3] == m] #
for i in range(3,6): #
    sampling_cardinality = 'len(tb_'+str(i)+'.xi_train)' #
    print eval(sampling_cardinality) #
'''
