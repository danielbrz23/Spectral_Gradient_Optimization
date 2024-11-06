# importing useful librarys
import numpy as np
import pycutest as pqt
import matplotlib.pyplot as plt
import params 

def armijo(fx, d, xk, evalf, evalgf, values= None, MAX_ITER =  params.MAX_ITER_ARMIJO):
    eta = params.ETA
    beta = params.BETA
    alpha = 1
    if values == None:
        values = [fx.obj(xk)]
        evalf +=1

    term = eta *alpha * (fx.grad(xk)).T @ d
    evalgf+=1

    max_fx  = float(max(values))
    i=0
    while (fx.obj(xk+alpha *d) >= max_fx +term and i<=MAX_ITER): # finds the smallest 
        evalf +=1
        alpha *= beta # 
        term *= beta 
        i+=1
    if i>MAX_ITER:
        return False
    return alpha, evalf, evalgf


def gradient_descent(x0,fx, MAX_ITER = params.MAX_ITER_GD, EPSILON = params.EPSILON):
    # x0 is our initial point
    # fx is an pycutest object that contains the objective function fx.obj() as a pyhton method
    xk = x0
    k = 0
    status = False

    evalf = 0
    evalgf = 0

    while (k < MAX_ITER): # stopping criterion
        gradf = fx.grad(xk)
        evalgf +=1
        if (np.linalg.norm(gradf, ord=np.inf) < EPSILON): # Optimality condition
            status = True
            break
        d = -gradf # steepest descent direction

        armijo = armijo(fx, d, xk, evalf, evalgf)
        if armijo == False:
            return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
        else:
            alpha, evalf, evalgf =  armijo

        xk = xk+alpha* d
        k +=1
    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}

def bfgs_hessian(Hk_1, gradfk, gradfk_1, xk_1, alpha, d):
    s = alpha * d # ( xk - xk_1 )
    y = gradfk - gradfk_1

    I = np.eye(Hk_1.shape[0])
    rho = 1/(y.T @ s)
    z = np.dot(s, y.T)
    Hk = (I - rho * z) @ Hk_1 @(I - rho * z.T) + rho * np.dot(s, s.T)

    return Hk


def BFGS(xk, fx, MAX_ITER = params.MAX_ITERATIONS, EPSILON = params.EPSILON):
    k = 0
    Hk =  np.eye(len(xk))  # Initial aproximation to the hessian of fx

    status = False
    evalf, evalgf = 0,0

    gradfk = fx.grad(xk)
    evalgf +=1

    while (k < MAX_ITER): # stopping criterion
        if (np.linalg.norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
            status = True
            break
        d = np.dot(Hk, -fx.grad(xk))

        armijo = armijo(fx, d, xk, evalf, evalgf)
        if armijo == False:
            return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
        else:
            alpha, evalf, evalgf =  armijo
        
        xk = xk + alpha * d
        gradfk_1 = gradfk
        gradfk =  fx.grad(xk)
        evalgf+=1

        Hk = bfgs_hessian(Hk, gradfk, gradfk_1,  xk, alpha, d)
        
        k+=1

    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k}

def sgradient_direction(xk_1, xk,  gradfk, gradfk_1):
    s = xk - xk_1
    y = gradfk - gradfk_1

    if (s @ s) > 0:
        sigma = (s @ y)/ (s @ s)
    else:
        sigma =1e-4 * gradfk/max(1.0, np.linalg.norm(xk))

    sigma = max(1e-30, min(sigma, 1e30))

    d = -(1/sigma) * gradfk

    return d


def spectral_gradient(xk_1, xk, fx,  step = 'simple') :
    k=1
    MAX_ITER = params.MAX_ITER_SG
    evalf, evalgf = 0,0

    gradfk_1 = fx.grad(xk_1)
    evalgf +=1

    EPSILON = 1e-4
    status = False
    match step:
        case 'armijo':
            while (k < MAX_ITER): # stopping criterion
                gradfk = fx.grad(xk)
                evalgf +=1
                if (np.linalg.norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
                    status = True
                    break
                d = sgradient_direction(xk_1, xk, gradfk, gradfk_1)
                
                armijo = armijo(fx, d, xk, evalf, evalgf)
                if armijo == False:
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
                else:
                    alpha, evalf, evalgf =  armijo

                xk = xk+alpha* d
                k +=1
                gradfk_1 = gradfk

        case 'armijo_mod':
            values = [] 

            while (k < MAX_ITER): # stopping criterion
                gradfk = fx.grad(xk)
                evalgf +=1
                if (np.linalg.norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
                    status = True
                    break
                d = sgradient_direction(xk_1, xk, gradfk, gradfk_1)
                
                if len(values) >= 10:
                    values.pop(0)
                values.append(fx.obj(xk))

                d = sgradient_direction(xk_1, xk, fx)

                armijo = armijo(fx, d, xk, evalf, evalgf, values = values)
                if armijo == False:
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
                else:
                    alpha, evalf, evalgf =  armijo

                xk = xk+alpha* d
                k +=1
                gradfk_1 = gradfk

        case 'simple':
            while (k < MAX_ITER): # stopping criterion
                gradfk = fx.grad(xk)
                evalgf +=1
                if (np.linalg.norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
                    status = True
                    break
                d = sgradient_direction(xk_1, xk, gradfk, gradfk_1)

                xk = xk+ d
                k +=1
                gradfk_1 = gradfk
    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k-1, 'evalf': evalf, 'evalgf': evalgf}