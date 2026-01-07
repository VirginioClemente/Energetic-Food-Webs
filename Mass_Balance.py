import numpy as np
from numpy import linalg as LA

from scipy.optimize import least_squares
import scipy.stats


class Fluxes():
    
    '''
    This class solves the system of equations described in [...]
    
    
    Initiate the class by passing the following input:
    
    biomasses: Biomasses
    A: adjacency_matrix
    metabolisms: metabolisms
    e: efficiences 
    m: is the self-limitation term, it is assumed to be proportional to the death rate of each specie.
    '''
    
    def __init__(self,A,biomasses,metabolism,e,m):
        
        
        self.A = A
        self.biomasses = biomasses
        self.metabolism = metabolism
        self.e = e
        self.m = m
        
        
        
    def equation_to_solve(self, F,A,biomasses,metabolism,e,m):
    
        '''
        This is where the equations are defined.
        The equations so defined are passed as input to the solver. 
        
        
        
        Preferences are defined according to the following 
        
        W[i,j] = A[i,j] B[i] / sum_k A[k,j] B[k]
        
        where B[i] is the biomass to the element i.
        
        '''
        
        n_nodes = len(biomasses)

        f1 = np.zeros(n_nodes, dtype= np.float64)

        W = np.zeros((n_nodes,n_nodes),dtype= np.float64)

        for i in range(n_nodes):
            for j in range(n_nodes):
                termine_j = sum([A[k,j]*biomasses[k] for k in range(n_nodes)])
                if termine_j == 0: termine_j=1

                W[i,j] = A[i,j] *biomasses[i]/termine_j


        for i in range(n_nodes):
            if A.sum(0)[i] != 0:
                somma = 0
                for j in range(n_nodes):
                    somma += W[i,j]*F[j]

                f1[i] =  e[i] * F[i] -  metabolism[i] * biomasses[i] - somma - m[i]*(biomasses[i])**2 

        return f1




    def solve(self):
        
        '''
        Here the following Mass-Balance equations are solved:
        
        
        1) e[i] F_in[i] = x[i] B[i] + F_out[i] + m[i] B[i]^2  for the consumers;
        
        2) r[i] B[i] = F_out[i] + m[i] B[i]^2 for basal resources.
        
        
        Output:
        
        F: Adjacency matrix that captures the energy fluxes between consumer species and their corresponding resources.
        x_solved.cost : Loss function 
        
        '''

        n_nodes = len(self.biomasses)

        initial_values =  self.biomasses + np.ones(n_nodes)


        lower_bound = [0 for i in range(n_nodes) ]
        upper_bound = [np.inf for i in range(n_nodes) ] 
        boundslu = tuple(lower_bound), tuple(upper_bound)

        x_solved = least_squares(fun=self.equation_to_solve,
                                 x0=self.biomasses,
                                 jac='3-point',
                                 args=(self.A,self.biomasses,self.metabolism,self.e,self.m),
                                 bounds=boundslu,
                                 max_nfev=1e3,
                                 loss='linear',
                                 verbose=0,
                                 ftol=1e-9, xtol=1e-9, gtol=1e-9)


        W = np.zeros((n_nodes,n_nodes),dtype= np.float128)

        for i in range(n_nodes):
            for j in range(n_nodes):
                termine_j = sum([self.A[k,j]*self.biomasses[k] for k in range(n_nodes)])
                if termine_j == 0: termine_j=1

                W[i,j] = self.A[i,j] *self.biomasses[i]/termine_j

        F_in = x_solved.x

        F = np.zeros((n_nodes,n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):

                F[i,j] = F_in[j] * W[i,j]


        return F,x_solved.cost
    
    
    def compute_jacobian(self):
        
        
        '''        
        Output:
        
        jac: Jacobian associated with the population model
        '''
        
        
        F, cost = self.solve()
        n_nodes = len(self.biomasses)

        jac = np.zeros((n_nodes,n_nodes),dtype = np.float64)

        for i in range(n_nodes):
            for j in range(n_nodes):

                if i == j and self.A.sum(0)[i] == 0:
                    jac[i,j] =  -self.biomasses[i]*self.m[i]  
                    continue

                if i == j and self.A.sum(0)[i] != 0:
                    jac[i,j] =  -self.m[i]*self.biomasses[i]  

                else:                
                    jac[i,j] = self.e[i]*F[j,i]/(self.biomasses[j]) - F[i,j]/(self.biomasses[j])

        return jac
