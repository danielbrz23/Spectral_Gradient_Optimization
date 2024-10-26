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

def armijo(fx, d, xk, beta, eta, values= None):
    alpha = 1
    if values == None:
        values = [fx.obj(xk)]

    term = eta *alpha * (fx.grad(xk)).T @ d
    max_fx  = float(max(values))

    while (fx.obj(xk+alpha *d) >= max_fx +term ): # finds the smallest 
        alpha *= beta # 
        term *= beta 
    return alpha 


def gradient_descent(x0,fx, eta = params.ETA_GD, beta = params.BETA_GD, MAX_ITER = params.MAX_ITERATIONS, EPSILON = params.EPSILON):
    # x0 is our initial point
    # fx is an pycutest object that contains the objective function fx.obj() as a pyhton method
    xk = x0
    k = 0


    status = False

    while (k < MAX_ITER): # stopping criterion
        if (np.linalg.norm(fx.grad(xk) < EPSILON)): # Optimality condition
            status = True
            break
        d = -fx.grad(xk) # steepest descent direction

        alpha =  armijo(fx, d, xk, beta, eta)
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


def BFGS(x0, fx, eta = params.ETA_BFGS, beta = params.BETA_BFGS, MAX_ITER = params.MAX_ITERATIONS, EPSILON = params.EPSILON):
    k = 0
    Hk =  np.eye(len(x0))  # Initial aproximation to the hessian of fx
    xk = x0

    beta, eta =  params.BETA_BFGS, params.ETA_BFGS
    MAX_ITER = params.MAX_ITERATIONS
    EPSILON = params.EPSILON
    status = False

    while (k < MAX_ITER): # stopping criterion
        if (np.linalg.norm(fx.grad(xk) < EPSILON)): # Optimality condition
            status = True
            break
        d = np.dot(Hk, -fx.grad(xk))

        alpha = armijo(fx, d, xk, beta, eta)
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

def spectral_gradient(x0, x1, fx,  step = None, eta = params.ETA_SG, beta = params.BETA_SG, MAX_ITER = params.MAX_ITERATIONS, EPSILON = params.EPSILON):

    k=1
    xk_1 = x0
    xk = x1
    alpha = 1

    if step == 'armijo':
      

        while (k < MAX_ITER): # stopping criterion
            if (np.linalg.norm(fx.grad(xk)) < EPSILON): # Optimality condition
                status = True
                break

            sigma = sgradient_direction(xk_1, xk, fx)
            
            alpha =  armijo(fx, d, xk, beta, eta)
            xk = xk+alpha* d
            k +=1

    elif step == 'armijo_mod':
        values = [] 

        while (k < MAX_ITER): # stopping criterion
            if (np.linalg.norm(fx.grad(xk)) < EPSILON): # Optimality condition
                status = True
                break
            
            if len(values) >= 10:
                values.pop(0)
            values.append(fx.obj(xk))

            d = sgradient_direction(xk_1, xk, fx)

            alpha = armijo(fx, d, xk, beta, eta,values)
            xk = xk+alpha* d
            k +=1

    else:
        while (k < MAX_ITER): # stopping criterion
            if (np.linalg.norm(fx.grad(xk)) < EPSILON): # Optimality condition
                status = True
                break
            
            d= sgradient_direction(xk_1, xk, fx)
            xk = xk+alpha* d
            k +=1

    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k-1}

