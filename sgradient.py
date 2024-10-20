# importing useful librarys
import numpy as np
import pycutest as pqt
import matplotlib.pyplot as plt

import params 

#################################
'''
        TO-DOS:
        - STUDY PYCUTEST
        - CALCULATE GRADIENT OF F (GRADF)
        - CALCULATE F(X) (FX)
        - STOP CRITERIA: NUMBER OF ITERATIONS
        - SEARCH IMPLEMENTATIONS FOR COMPARISON



'''



#################################

def armijo1(fx, d, xk):
    fx.obj(xk)
    alpha = 1
    beta = params.BETA_GD
    while (fx.obj(xk+alpha *d) >= fx.obj(xk)): # finds the smallest 
        alpha *= beta #
    return alpha 

def armijo2(fx, d, xk, beta, eta):
    alpha = 1
    # beta, eta = params.BETA_BFGS , params.ETA_BFGS 

    term = eta *alpha * (fx.grad(xk)).T # VERIFICAR SE É .T mesmo

    while (fx.obj(xk+alpha *d) >= fx.obj(xk) +term ): # finds the smallest 
        alpha *= beta # 
        term *= beta 
    return alpha 

def armijo_mod(fx, d, xk, beta, eta, values):
    alpha = 1
    # beta, eta = params.BETA_BFGS , params.ETA_BFGS 

    term = eta *alpha * (fx.grad(xk)).T # VERIFICAR SE É .T mesmo
    max_fx  = max(values)

    while (fx.obj(xk+alpha *d) >= max_fx +term ): # finds the smallest 
        alpha *= beta # 
        term *= beta 
    return alpha 


def steepest_descent(x0,fx):
    # x0 is our initial point
    # fx is an pycutest object that contains the objective function fx.obj() as a pyhton method
    xk = x0
    k = 0

    MAX_ITER = params.MAX_ITERATIONS
    EPSILON = params.EPSILON
    status = False

    while (k < MAX_ITER): # stopping criterion
        if (np.linalg.norm(fx.grad(xk) >  EPSILON)): # Optimality condition
            status = True
            break
        d = -fx.grad(xk) # steepest descent direction

        alpha =  armijo1(fx, d, xk)
        xk = xk+alpha* d
        k +=1
    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k}

def bfgs_hessian(Hk_1, fx, xk_1, alpha, d):
    xk = xk_1 +alpha * d
    s = xk - xk_1
    y = fx.grad(xk) - fx.grad(xk_1)

    I = np.eye(Hk_1.shape[0])
    rho = 1/(y.T @ s)
    z = np.dot(s, y.T)
    Hk = (I - rho * z) @ Hk_1 (I - rho * z.T) + rho * np.dot(s, s.T)

    return Hk


def BFGS(x0, fx):
    k = 0
    Hk =  np.eye(len(x0))  # Initial aproximation to the hessian of fx
    xk = x0

    MAX_ITER = params.MAX_ITERATIONS
    EPSILON = params.EPSILON
    status = False

    while (k < MAX_ITER): # stopping criterion
        if (np.linalg.norm(fx.grad(xk) >  EPSILON)): # Optimality condition
            status = True
            break
        d = np.dot(Hk, -fx.grad(xk))

        alpha = armijo2(fx, d, xk)
        Hk = bfgs_hessian(Hk, fx, xk, alpha, d)
        xk = xk + alpha * d

        k+=1

    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k}

def sgradient_direction(xk_1, xk, fx):
    s = xk - xk_1
    y = fx.grad(xk) - fx.grad(xk_1)

    sigma = (s.T @ y)/ (s.T @ s)
    d = -(1/sigma) * fx.grad(fx)

    return d

def spectral_gradient(x0, x1, fx,  step = None):
    MAX_ITER = params.MAX_ITERATIONS
    EPSILON = params.EPSILON

    k=1
    xk_1 = x0
    xk = x1
    alpha = 1

    if step == 'armijo':
        beta, eta  = params.BETA_SG, params.ETA_SG

        while (k < MAX_ITER): # stopping criterion
            if (np.linalg.norm(fx.grad(xk) >  EPSILON)): # Optimality condition
                status = True
                break

            sigma = sgradient_direction(xk_1, xk, fx)
            
            alpha =  armijo2(fx, d, xk, beta, eta)
            xk = xk+alpha* d
            k +=1

    elif step == 'armijo_mod':
        values = [] 
        beta, eta  = params.BETA_SG, params.ETA_SG

        while (k < MAX_ITER): # stopping criterion
            if (np.linalg.norm(fx.grad(xk) >  EPSILON)): # Optimality condition
                status = True
                break
            
            if len(values) >= 10:
                values.pop(0)
            values.append(fx.obj(xk))

            d = sgradient_direction(xk_1, xk, fx)

            alpha = armijo_mod(fx, d, xk, beta, eta,values)
            xk = xk+alpha* d
            k +=1

    else:
        while (k < MAX_ITER): # stopping criterion
            if (np.linalg.norm(fx.grad(xk) >  EPSILON)): # Optimality condition
                status = True
                break
            
            sigma = sgradient_direction(xk_1, xk, fx)
            d = -(1/sigma) * fx.grad(fx)
            xk = xk+alpha* d
            k +=1

    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k-1}

