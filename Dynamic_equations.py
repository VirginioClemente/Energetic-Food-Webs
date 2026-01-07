import numpy as np
import pandas as pd

from numpy import linalg as LA

from scipy.optimize import least_squares
import scipy.stats




def equations_dinamiche(number, F_out, F_in, x_i, e, gr, m):
    

    """
    Calculate dynamic equations for a system with various nodes.

    This function computes the system's dynamics based on the properties of each node, 
    incoming and outgoing fluxes, and certain coefficients. The equations govern 
    how each node interacts with the flux and other factors.

    Parameters:
    - number (ndarray): Numerical values representing specific properties (e.g., quantities) of system nodes.
    - F_out (ndarray): Outgoing fluxes for each node in the system.
    - F_in (ndarray): Incoming fluxes for each node in the system.
    - x_i (ndarray): Coefficients affecting the property represented by 'number' for each node.
    - e (ndarray): Coefficients for incoming flux for each node.
    - gr (ndarray): Growth rates or related factors for each node.
    - m (ndarray): Miscellaneous coefficients affecting each node, possibly related to decay, resistance, or other factors.

    Returns:
    - f1 (ndarray): The calculated dynamics (e.g., rate of change) for each node based on the input parameters.
    """

    n_nodes = len(number)

    f1 = np.zeros(n_nodes, dtype=np.float128)

    for i in range(n_nodes):
        if i < 10:
            f1[i] = -x_i[i] * number[i] + e[i] * F_in[i] - F_out[i] - m[i] * (number[i]) ** 2
        else:
            f1[i] = gr[i] * number[i] - F_out[i] - m[i] * (number[i]) ** 2
    # print(f1)
    # return
    return f1


def numerically_solve_equations_dinamiche(initial_0, F_out, F_in, x_i, e, gr, m):


    """
    Numerically solves the dynamic equations for a system of nodes.

    This function uses numerical methods to find solutions to the system of equations
    defined by the dynamics of each node. It considers node properties, fluxes, and 
    various coefficients to understand how the system evolves.

    Parameters:
    - initial_0 (ndarray): Initial conditions or values for the properties of the nodes.
    - F_out (ndarray): Outgoing fluxes for each node in the system.
    - F_in (ndarray): Incoming fluxes for each node in the system.
    - x_i (ndarray): Coefficients affecting the property represented by the initial conditions for each node.
    - e (ndarray): Coefficients for incoming flux for each node.
    - gr (ndarray): Growth rates or related factors for each node.
    - m (ndarray): Miscellaneous coefficients affecting each node.

    Returns:
    - number (ndarray): The numerically solved values for the properties of each node, given the system's dynamics.
    - x_solved.cost (float): The cost value from the numerical solving process, indicating the optimization's convergence quality.
    """

    n_nodes = len(F_out)

    initial = initial_0
    lower_bound = [0 for i in range(len(initial))]
    upper_bound = [np.inf for i in range(len(initial))]

    boundslu = tuple(lower_bound), tuple(upper_bound)

    x_solved = least_squares(fun=equations_dinamiche,
                             x0=initial,
                             jac='3-point',
                             args=(F_out, F_in, x_i, e, gr, m),
                             bounds=boundslu,
                             max_nfev=1e5,
                             loss='huber',
                             ftol=1e-13, xtol=1e-13, gtol=1e-13)

    number = x_solved.x
    number = np.array(number)

    return number, x_solved.cost


def compute_jacobian(A, F, vet_number, metabolism, e, m):
    """
    Computes the Jacobian matrix for a system of nodes based on various parameters.

    The function calculates the Jacobian matrix representing the first-order partial derivatives
    of the system's functions with respect to the nodes. This matrix is crucial for understanding
    the system's stability and dynamics during simulations and solving differential equations.

    Parameters:
    - A (ndarray): Adjacency matrix or similar, representing the connections between nodes.
    - F (ndarray): Matrix of fluxes between nodes.
    - vet_number (ndarray): Array representing specific properties (e.g., quantities) of system nodes.
    - metabolism (not used in the function, possibly deprecated): Intended for factors affecting node metabolism.
    - e (ndarray): Coefficients for node-specific terms, possibly related to energy or other resources.
    - m (ndarray): Coefficients affecting each node, possibly related to decay, resistance, or other factors.

    Returns:
    - jac (ndarray): The Jacobian matrix, showing how each function in the system changes with respect to the nodes' properties.
    """
    n_nodes = len(vet_number)

    jac = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for i in range(n_nodes):
        for j in range(n_nodes):

            if i == j and A.sum(0)[i] == 0:
                jac[i, j] = -vet_number[i] * m[i]
                continue

            if i == j and A.sum(0)[i] != 0:
                jac[i, j] = -m[i] * vet_number[i]

            else:
                jac[i, j] = e[i] * F[j, i] / (vet_number[j]) - F[i, j] / (vet_number[j])

    return jac


def frobenius_eigen_difference(M):    

    
    # Calculate the squared Frobenius norm
    frobenius_norm_squared = np.linalg.norm(M, 'fro')**2

    # Calculate eigenvalues and the sum of their squared magnitudes
    eigenvalues = np.linalg.eigvals(M)
    sum_squared_magnitudes = np.sum(np.abs(eigenvalues)**2)

    # Calculate the difference
    difference = np.sqrt(frobenius_norm_squared - sum_squared_magnitudes)

    # The result should be the difference, no square root is taken as per the formula in the screenshot
    return difference


def Numerical_abscissa(M):

    H = (M + M.T)/2
    
    eigenvalues = np.linalg.eigvals(H)

    # Prendi la parte reale di ciascun autovalore
    #real_parts = np.real(eigenvalues)

    # Trova e restituisce il massimo tra le parti reali
    return np.max(eigenvalues)


def compute_tot_fluxes(F):
    tot = 0
    for i in range(len(F[:,0])):
        for j in range(len(F[0,:])):
            tot += F[i,j]
    return tot
    
