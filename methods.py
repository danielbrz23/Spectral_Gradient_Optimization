import numpy as np
from numpy.linalg import norm
import params 
from math import isnan

def armijo(fx, d, xk, gradfk, evalf, values,  eta = params.ETA,  beta = params.BETA, MAX_ITER =  params.MAX_ITER_ARMIJO):
    alpha = 1
    term = eta * np.dot(gradfk, d)

    max_fx  = max(values)
    i=0
    while (i<=MAX_ITER): 
        if fx.obj(xk+alpha*d) < ( max_fx + alpha * term ):
            return alpha, evalf+1
        evalf+=1
        alpha *= beta 
        i+=1
    return False
    
def gradient_descent(x0,fx, MAX_ITER = params.MAX_ITER_GD, EPSILON = params.EPSILON):
    """
    Gradient descent with Armijo Line Search
    
    Parameters:
    - fx: an pycutest object with methods `obj(x)` (function evaluation) and `grad(x)` (gradient evaluation)
    - xk: initial point
    - max_iter: maximum number of iterations
    - EPSILON: tolerance for stopping criterion

    Returns:
    - A dictionary with the final point, function value, iteration count and evaluations count.
    """

    xk = x0
    k = 0

    evalf,evalgf = 0, 0

    while (k < MAX_ITER): # stopping criterion
        gradf = fx.grad(xk)
        evalgf +=1
        if (norm(gradf, ord=np.inf) < EPSILON): # Optimality condition
            return  { 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
        d = -gradf # steepest descent direction
        d = d/np.max(np.abs(d)) if k == 0 else d* 0.1

        arm= armijo(fx, d, xk, gradf, evalf, values = [fx.obj(xk)])
        evalf+=1
        if arm == False:
            return { 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
        else:
            alpha, evalf=  arm

        xk = xk+alpha* d
        k +=1
    return { 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf':- evalf, 'evalgf': -evalgf}

def bfgs_hessian(Hk_1, gradfk, gradfk_1, alpha, d):
    s = alpha * d # ( xk - xk_1 )
    y = gradfk - gradfk_1

    I = np.eye(Hk_1.shape[0])
    rho = 1/(y.T @ s) if (y.T @ s) != 0 else 1e-10
    z = np.outer(s, y.T)
    Hk = (I - rho * z) @ Hk_1 @(I - rho * z.T) + rho * np.outer(s, s.T)

    return Hk


def BFGS(xk, fx, MAX_ITER = params.MAX_ITER_BFGS, EPSILON = params.EPSILON):
    """
    Quasi-Newton BFGS with Armijo Line Search
    
    Parameters:
    - fx: an pycutest object with methods `obj(x)` (function evaluation) and `grad(x)` (gradient evaluation)
    - xk: initial point
    - max_iter: maximum number of iterations
    - EPSILON: tolerance for stopping criterion

    Returns:
    - A dictionary with the final point, function value, iteration count and evaluations count.
    """
    k = 0
    Hk =  np.eye(len(xk))  # Initial aproximation to the hessian of fx
    evalf, evalgf = 0,0
    gradfk = fx.grad(xk)
    evalgf +=1

    while (k < MAX_ITER): # stopping criterion
        if (norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
            return  { 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
        d = np.dot(Hk, -gradfk)

        arm= armijo(fx, d, xk, gradfk, evalf, values = [fx.obj(xk)])
        evalf+=1
        if arm == False:
            return {  'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
        else:
            alpha, evalf=  arm
        
        xk = xk + alpha * d
        gradfk_1 = gradfk
        gradfk =  fx.grad(xk)
        evalgf+=1

        Hk = bfgs_hessian(Hk, gradfk, gradfk_1,  alpha, d)
        
        k+=1

    return {  'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}

def sgradient_direction(xk_1, xk,  gradfk, gradfk_1):
    s = xk - xk_1
    y = gradfk - gradfk_1
    skyk = np.dot(s, y)
    sksk = np.dot(s, s)

    if  skyk> 0 and sksk!= 0:
        sigma = skyk/sksk

    else:
        n_xk = (np.max(np.abs(xk)))
        sigma  = (1.0e-4 * (np.max(np.abs(gradfk_1))))/ (max(1.0, n_xk ))

    sigma = max(1.0e-30, min(sigma, 1.0e30))
    d = -(1/sigma) * gradfk
    return d


def spectral_gradient(xk_1, fx,  step = 'simple', EPSILON = params.EPSILON, MAX_ITER = params.MAX_ITER_SG) :
    """
    Spectral Gradient Descent with Armijo Line Search
    
    Parameters:
    - fx: an pycutest object with methods `obj(x)` (function evaluation) and `grad(x)` (gradient evaluation)
    - xk_1: initial point
    - step: a parameter that determines the type of search
    - max_iter: maximum number of iterations
    - EPSILON: tolerance for stopping criterion

    Returns:
    - A dictionary with the final point, function value, iteration count and evaluations count.
    """
    k=1
    evalf, evalgf = 0,0
    gradfk_1 = fx.grad(xk_1)
    evalgf +=1

    ## Finding x1
    n_g = np.max(np.abs(gradfk_1))
    n_x = np.max(np.abs(xk_1))
    eps = 2.22e-16  # machine epsilon
    a = (np.sqrt(eps) * max(np.sqrt(eps), n_x))/n_g
    xk = xk_1 - a*gradfk_1 
    values = [fx.obj(xk_1)] 

    while (k < MAX_ITER): 
        gradfk = fx.grad(xk)
        evalgf +=1
        if (np.max(np.abs(gradfk)) < EPSILON): 
            return  { 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
        
        d = sgradient_direction(xk_1, xk, gradfk, gradfk_1)       
        fxk = fx.obj(xk)
        if isnan(fxk):
            return {  'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
        evalf+=1

        match step:
            case 'simple':
                alpha =1 
            case 'armijo':
                arm=  armijo(fx, d, xk, gradfk, evalf, values = [fxk])
                if arm == False:
                    return {  'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
                else:
                    alpha, evalf=  arm

            case 'armijo_mod':
                if len(values) >= 10:
                    values.pop(0)
                values.append(fxk)
                     
                arm=  armijo(fx, d, xk, gradfk, evalf, values = values)
                if arm == False:
                    return {  'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
                else:
                    alpha, evalf=  arm
            case _:
                print('Error: Variable *step* only recieves one of the following options: ["simple", "armijo", "armijo_mod"]! ')
                return
        xk_1 = xk
        xk = xk+alpha* d
        gradfk_1 = gradfk
        k +=1
    return {  'xk': xk, 'fx': fx.obj(xk), 'iter': k-1, 'evalf': -evalf, 'evalgf': -evalgf}