import os # for path and makedir
import shutil # for rm
import sys # for exit
import random # to randomize selection in case of equal error bound
from stochastic_collocation_basis import *
from quadrature.QuadratureRule import *

class StochasticCollocation(StochasticCollocationBasis):
    
    def __init__(self, V, bc_list):
        StochasticCollocationBasis.__init__(self, V, bc_list)
        
        self.s = 0
        
        self.snapshot_matrix = np.array([])
        self.outputs_vector = []
        
        self.order = 0
        self.sparse_flag = 0
        self.rule = None
    
    def setnodes_and_weights(self, order, sparse_flag, rule):
        StochasticCollocationBasis.setnodes_and_weights(self, order, sparse_flag, rule)
        
    def build_solutions(self):
        if os.path.exists(self.post_processing_folder):
            shutil.rmtree(self.post_processing_folder)
        folders = (self.snapshots_folder, self.basis_folder, self.reduced_matrices_folder, self.post_processing_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
        
        self.truth_A = self.assemble_truth_a()
        self.apply_bc_to_matrix_expansion(self.truth_A)
        self.truth_F = self.assemble_truth_f()
        self.apply_bc_to_vector_expansion(self.truth_F)
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        
        for run in range(len(self.xi_train)):
            self.setmu(self.nodes[run])
            
            self.truth_solve()
            self.export_solution(self.snapshot, self.snapshots_folder + "truth_" + str(run))
            
            self.update_snapshot_matrix()
            
            run += 1
            
    def build_outputs(self):
        for run in range(len(self.xi_train))
            self.setmu(self.nodes[run])
            
            self.truth_output()
            self.outputs_vector.append(self.s) 
            
            run += 1
    
    def update_snapshot_matrix(self):
        self.store_single_snapshot(self.snapshot)
    
    def store_single_snapshot(self, snapshot):
        if self.snapshot_matrix.size == 0:
            self.snapshot_matrix = np.array(snapshot.vector()).reshape(-1, 1)
        else:
            self.snapshot_matrix = np.hstack((self.snapshot_matrix, np.array(snapshot.vector()).reshape(-1, 1)))
    
    def compute_u_mean(self): # compute E[u]
        u = self.weights[0]*np.array(self.snapshot_matrix[:,0]).reshape(-1, 1)
        for j in range(1,len(weights)):
            u = u + self.weights[j]*np.array(self.snapshot_matrix[:,j]).reshape(-1, 1)
        return u
    
    def compute_u_norm_mean(self): # output of interest: E[||u||]
        S = as_backend_type(self.S)
        dim = S.size(0)
        X_norm = S.mat().getValues(range(dim),range(dim))
        u = self.weights[0]*np.dot(np.array(self.snapshot_matrix[:,0]).reshape(-1, 1).T,np.matrix(np.dot(X_norm,np.array(self.snapshot_matrix[:,0]).reshape(-1, 1))))
        for j in range(1,len(self.weights)):
            u = u + self.weights[j]*np.dot(np.array(self.snapshot_matrix[:,j]).reshape(-1, 1).T,np.matrix(np.dot(X_norm,np.array(self.snapshot_matrix[:,j]).reshape(-1, 1))))
        return u
    
    def compute_norm_u_mean(self): # output of interest: ||E[u]|| : no devo fare differenza in error!!!
        u = compute_u_mean()
        S = as_backend_type(self.S)
        dim = S.size(0)
        X_norm = S.mat().getValues(range(dim),range(dim))
        norm_u = np.dot(np.array(u).reshape(-1, 1).T,np.matrix(np.dot(X_norm,np.array(u).reshape(-1, 1))))
        return norm_u
    
    def compute_s_mean(self):  # output of interest: E[s]
        s_mean = self.weights[0]*self.outputs_vector[0]
        for j in range(1,len(self.outputs_vector)):
            s_mean += self.weights[j]*self.outputs_vector[j]
        return u
    
    def online_solve(self, N=None, with_plot=True): # stampa E[s], E[||u||], ||E[u]||, plot E[u]
        sys.exit("Plase define function online_solve!")
    
    ###########################     ERROR ANALYSIS     ###########################
        
    def error_analysis(self, N=None):
        u_mean_scm = compute_u_mean()
        norm_u_mean_scm = compute_norm_u_mean()
        s_mean_scm = compute_s_mean()
        
        self.truth_A = self.assemble_truth_a()
        self.apply_bc_to_matrix_expansion(self.truth_A)
        self.truth_F = self.assemble_truth_f()
        self.apply_bc_to_vector_expansion(self.truth_F)
        self.Qa = len(self.truth_A)
        self.Qf = len(self.truth_F)
        
        S = as_backend_type(self.S)
        dim = S.size(0)
        X_norm = S.mat().getValues(range(dim),range(dim))
        
        self.setmu(self.xi_test[0])
            
        self.truth_solve()
        self.truth_output()
            
        u_norm_mean_mc += np.dot(np.array(self.snapshot.vector()).reshape(-1, 1).T,np.matrix(np.dot(X_norm,np.array(self.snapshot.vector()).reshape(-1, 1))))
        s_mean_mc = self.s
        u_mean_mc = self.snapshot.vector()
        
        for run in range(1,len(self.xi_test)):
            
            self.setmu(self.xi_test[run])
            
            self.truth_solve()
            self.truth_output()
            
            u_norm_mean_mc += np.dot(np.array(self.snapshot.vector()).reshape(-1, 1).T,np.matrix(np.dot(X_norm,np.array(self.snapshot.vector()).reshape(-1, 1))))
            s_mean_mc += self.s
            u_mean_mc = u_mean_mc + self.snapshot.vector()
            
        u_mean_mc = u_mean_mc/len(self.xi_test)
        u_norm_mean_mc = u_norm_mean_mc/len(self.xi_test)
        s_mean_mc = s_mean_mc/len(self.xi_test)
        
        err_mean = u_test-u_mean_scm
        err1 = norm_u_mean_mc = np.dot(np.array(err_mean).reshape(-1, 1).T,np.matrix(np.dot(X_norm,np.array(err_mean).reshape(-1, 1))))
        err2 = abs(u_norm_mean_mc-u_norm_mean_scm)
        errs = abs(s_mean_mc-s_mean_scm)
        
        return err1, err2, errs    
