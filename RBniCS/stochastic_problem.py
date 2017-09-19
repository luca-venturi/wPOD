from dolfin import File, plot
import os # for path and makedir
import numpy as np
from quadrature.QuadratureRule import *
from weight import *
from stochastic_collocation_basis import *

class StochasticProblem(object):
    
    def __init__(self):
        self.mu = ()
        self.mu_range = []
        self.nodes = []
        self.weights = []
        self.cardinality = []
        self.density = None
        
        self.nodes_folder = "nodes/"
        self.weights_folder = "weights/"
        self.xi_test_folder = "xi_test/"
        
        self.xi_test = []
    
    def setmu_range(self, mu_range):
        self.mu_range = mu_range
        
    def set_density(self, weight=None):
        self.density = weight
    
    def setnodes_and_weights(self, order, sparse_flag, rule):
        if not os.path.exists(self.nodes_folder):
            os.makedirs(self.nodes_folder)
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)
        self.nodes, self.weights, self.cardinality = self.generate_nodes_and_weights_set(order, sparse_flag, rule)
        if self.density = None:
            self.density = IndicatorWeight(UniformWeight(), -1e8)
        rho = self.density.density(self.mu_range, self.nodes)
        for j in range(len(self.weights)):
            self.weights[j] = self.weights[j]*rho[j]
        np.save(self.nodes_folder + "nodes", self.nodes)
        np.save(self.weights_folder + "weights", self.weights)
    
    def setxi_test(self, ntest, enable_import=False, sampling=None):
        if not os.path.exists(self.xi_test_folder):
            os.makedirs(self.xi_test_folder)
        import_successful = False
        if enable_import and os.path.exists(self.xi_test_folder + "xi_test.npy"):
            xi_test = np.load(self.xi_test_folder + "xi_test.npy")
            import_successful = (len(np.asarray(xi_test)) == ntest)
        if import_successful:
            self.xi_test = list()
            for i in range(len(np.asarray(xi_test))):
                self.xi_test.append(tuple(xi_test[i, :]))
        else:
            self.xi_test = self.generate_test_set(ntest, sampling)
            np.save(self.xi_test_folder + "xi_test", self.xi_test)
    
    def generate_nodes_or_weights_set(self, order, sparse_flag, rule, alpha=0):
        if sparse_flag = 0:
            rule_name = 'TensorProductRule(len(self.mu_range),order,rule,self.mu_range,a=alpha)'
        elif sparse_flag = 1:
            rule_name = 'SmolyakRule(len(self.mu_range),order,rule,self.mu_range,a=alpha)'
        return eval(rule_name)
        
    def generate_test_set(self, n, sampling):
        if sampling == None:
            sampling = Uniform()
        return sampling.sample(self.mu_range, n)
    
    def setmu(self, mu):
        assert (len(mu) == len(self.mu_range)), "mu and mu_range must have the same lenght"
        self.mu = mu
    
    def _plot(self, solution, *args, **kwargs):
        self.move_mesh()
        preprocessed_solution = self.preprocess_solution_for_plot(solution)
        plot(preprocessed_solution, *args, **kwargs)
        self.reset_reference()
    
    def _export_vtk(self, solution, filename, output_options={}):
        if not "With mesh motion" in output_options:
            output_options["With mesh motion"] = False
        if not "With preprocessing" in output_options:
            output_options["With preprocessing"] = False
        file = File(filename + ".pvd", "compressed")
        if output_options["With mesh motion"]:
            self.move_mesh()
        if output_options["With preprocessing"]:
            preprocessed_solution = self.preprocess_solution_for_plot(solution)
            file << preprocessed_solution
        else:
            file << solution
        if output_options["With mesh motion"]:
            self.reset_reference()
        
    def preprocess_solution_for_plot(self, solution):
        return solution
        
    def move_mesh(self):
        pass
    
    def reset_reference(self):
        pass

