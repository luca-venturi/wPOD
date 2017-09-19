from dolfin import *
from RBniCS import *
import matplotlib.pyplot as plt

class Tblock(EllipticCoerciveRBBase):

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
first_mu = (2.0*np.random.beta(10.,10.)+1.0,2.0*np.random.beta(10.,10.)+1.0,2.0*np.random.beta(10.,10.)+1.0,2.0*np.random.beta(10.,10.)+1.0,)

mesh = Mesh("Data/4_tblock.xml")
subd = MeshFunction("size_t", mesh, "Data/4_tblock_physical_region.xml")
bound = MeshFunction("size_t", mesh, "Data/4_tblock_facet_region.xml")

V = FunctionSpace(mesh, "Lagrange", 1)

parameters.linear_algebra_backend = 'PETSc'

N_xi_train = 1000
N_xi_test = 500

param = [(10.,10.),(10.,10.),(10.,10.),(10.,10.)]
originalDistribution = BetaDistribution(param)
originalWeight = BetaWeight(param) 

# 1.0
tb_0 = Tblock(V, subd, bound)

tb_0.setmu_range(mu_range)
tb_0.setNmax(Nmax)

distribution = UniformDistribution()
density = originalWeight

tb_0.setxi_train(N_xi_train, sampling=distribution)
tb_0.set_density(weight=density)
tb_0.set_weighted_flag(1)

tb_0.setmu(first_mu)
tb_0.offline() 

tb_0.setxi_test(N_xi_test, sampling=originalDistribution)
mean_error_u[:,0] = tb_0.error_analysis()

# 1.1
tb_1 = Tblock(V, subd, bound)

tb_1.setmu_range(mu_range)
tb_1.setNmax(Nmax)

distribution = UniformDistribution()
density = originalWeight

tb_1.setxi_train(N_xi_train, sampling=distribution)
tb_1.set_density(weight=density)
tb_1.set_weighted_flag(2)

tb_1.setmu(first_mu)
tb_1.offline() 

tb_1.setxi_test(N_xi_test, sampling=originalDistribution)
mean_error_u[:,1] = tb_1.error_analysis()

# 1.2
tb_2 = Tblock(V, subd, bound)

tb_2.setmu_range(mu_range)
tb_2.setNmax(Nmax)

distribution = UniformDistribution()
density = originalWeight

tb_2.setxi_train(N_xi_train, sampling=distribution)
tb_2.set_density(weight=density)
tb_2.set_weighted_flag(3)

tb_2.setmu(first_mu)
tb_2.offline() 

tb_2.setxi_test(N_xi_test, sampling=originalDistribution)
mean_error_u[:,2] = tb_2.error_analysis()

# 7. Plot the errors

plt.plot(np.log10(mean_error_u[:,0]),'b',label='WGreedy1 - Uniform')
plt.plot(np.log10(mean_error_u[:,1]),'r',label='WGreedy2 - Uniform')
plt.plot(np.log10(mean_error_u[:,2]),'y',label='WGreedy3 - Uniform')
plt.legend()
plt.show()

# 8. Returns the best method #

mean_N_err = np.mean(np.log10(mean_error_u[:,:3]), axis=0) #
m = min(mean_N_err) #
print [i for i in range(3) if mean_N_err[i] == m] #
for i in range(3): #
    sampling_cardinality = 'len(tb_'+str(i)+'.xi_train)' #
    print eval(sampling_cardinality) #
