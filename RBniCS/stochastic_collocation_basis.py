from dolfin import *
import numpy as np
import sys # for exit
from stochastic_problem import *

class StochasticCollocationBasis(StochasticProblem):
    
    def __init__(self, V, bc_list):
        StochasticProblem.__init__(self)
        
        self.Qa = 0
        self.Qf = 0
        self.theta_a = ()
        self.theta_f = ()
        
        self.truth_A = ()
        self.truth_F = ()
        self.snapshot = Function(V)
        self.error = Function(V)
        
        self.bc_list = bc_list
        self.V = V
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        u = self.u
        v = self.v
        scalar = inner(u,v)*dx + inner(grad(u),grad(v))*dx
        self.S = assemble(scalar)
        if self.bc_list != None:
            [bc.apply(self.S) for bc in self.bc_list]
    
    def truth_solve(self):
        self.theta_a = self.compute_theta_a()
        self.theta_f = self.compute_theta_f()
        assembled_truth_A = self.affine_assemble_truth_matrix(self.truth_A, self.theta_a)
        assembled_truth_F = self.affine_assemble_truth_vector(self.truth_F, self.theta_f)
        truth_sol = solve(assembled_truth_A, self.snapshot.vector(), assembled_truth_F)
        return truth_sol
        
    def truth_output(self):
        self.theta_f = self.compute_theta_f()
        assembled_truth_F = self.affine_assemble_truth_vector(self.truth_F, self.theta_f)
        self.s = assembled_truth_F.inner(self.snapshot.vector())
        
    def affine_assemble_truth_matrix(self, vec, theta_v):
        A_ = vec[0]*theta_v[0]
        for i in range(1,len(vec)):
            A_ += vec[i]*theta_v[i]
        return A_
    
    def affine_assemble_truth_symmetric_part_matrix(self, vec, theta_v):
        A_ = self.affine_assemble_truth_matrix(vec, theta_v)
        AT_ = self.compute_transpose(A_)
        A_ += AT_
        A_ /= 2.
        return A_
        
    def affine_assemble_truth_vector(self, vec, theta_v):
        F_ = vec[0]*theta_v[0]
        for i in range(1,len(vec)):
            F_ += vec[i]*theta_v[i]
        return F_
    
    def apply_bc_to_matrix_expansion(self, vec):
        if self.bc_list != None:
            for i in range(len(vec)):
                [bc.apply(vec[i]) for bc in self.bc_list]
            for i in range(1,len(vec)):
                [bc.zero(vec[i]) for bc in self.bc_list]
        
    def apply_bc_to_vector_expansion(self, vec):
        if self.bc_list != None:
            for i in range(len(vec)):
                [bc.apply(vec[i]) for bc in self.bc_list]
    
    def compute_scalar(self,v1,v2,M):
        return v1.vector().inner(M*v2.vector())
        
    def compute_transpose(self, A):
        AT = A.copy()
        A = as_backend_type(A)
        AT = as_backend_type(AT)
        A.mat().transpose(AT.mat())
        return AT        
    
    def compute_theta_a(self):
        print "The function compute_theta_a() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function compute_theta_a(self)!")
    
    def export_solution(self, solution, filename):
        self._export_vtk(solution, filename, {"With mesh motion": True, "With preprocessing": True})
        
    def export_basis(self, basis, filename):
        self._export_vtk(basis, filename, {"With mesh motion": False, "With preprocessing": False})
    
    ###########################     PROBLEM SPECIFIC     ###########################
    
    def compute_theta_f(self):
        print "The function compute_theta_f() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function compute_theta_f(self)!")
    
    def assemble_truth_a(self):
        print "The function assemble_truth_a() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function assemble_truth_a(self)!")
    
    def assemble_truth_f(self):
        print "The function compute_truth_f() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function assemble_truth_f(self)!")
