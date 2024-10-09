# importing useful librarys
import numpy as np
import pycutest as pqt
import matplotlib.pyplot as plt

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
    alpha = 1
    beta = 0.8 # perguntar para o Silva Silva qual seria um bom valor
    while (fx(xk+alpha *d) >= fx(xk)): # finds the smallest 
        alpha *= beta #
    return alpha 

def armijo2(fx, d, xk,graadf):
    alpha = 1
    beta, eta = 0.5 , 0.5 # perguntar para o Silva Silva qual seria um bom valor

    term = eta *alpha * gradf(xk).T

    while (fx(xk+alpha *d) >= fx(xk) +term ): # finds the smallest 
        alpha *= beta #
    return alpha 

def steepest_descent(x0,fx,gradf):
    xk = x0
    k = 0
    while (gradf(xk) != 0 or k>100): # stop condition
        d = -gradf(xk) # steepest descent direction
        alpha =  armijo (fx, d, xk)
        xk = x0+alpha* d
        k +=1

def bfgs_hessian(Bk_1, xk_1, alpha, d):
    xk = xk_1 +alpha * d
    s = xk - xk_1
    y = gradf(xk) - gradf(xk_1)

    rho1 = 1/(y.T @ s)
    rho2 = 1/(s.T @ Bk_1 @ s)
    Bk = Bk_1 +(y @y.T)*rho + (Bk_1 @ s @ s.T @ Bk_1) * rho2


def BFGS(x0, fx, gradf):
    k = 0 ## PERGUNTAR PARA O PROFESSOR E AJUSTAR!!
    Bk = gradf(gradf(fx)) # Initial aproximation to the hessian of fx
    xk = x0

    while (gradf(xk) != 0): # stop condition
        d = np.linalg.solve(Bk, -gradf(fx))

        alpha = armijo2(fx, d, xk, gradf)

        Bk = bfgs_hessian(Bk, gradf, xk, alpha, d)