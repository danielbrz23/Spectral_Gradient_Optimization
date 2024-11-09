import numpy as np
import params 
import math

def armijo(fx:object, d, xk, evalf, evalgf, values, MAX_ITER =  params.MAX_ITER_ARMIJO):
    eta = params.ETA
    beta = params.BETA
    alpha = 1


    term = eta * np.dot((fx.grad(xk)), d)
    evalgf+=1

    max_fx  = max(values)
    i=0
    while (i<=MAX_ITER): 
        if fx.obj(xk+alpha*d) <= ( max_fx + alpha * term ):
            return alpha, evalf+i, evalgf
        alpha *= beta 
        i+=1
        
    return False
    


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
            return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
        d = -gradf # steepest descent direction

        arm= armijo(fx, d, xk, evalf, evalgf, values = [fx.obj(xk)])
        evalf+=1
        if arm == False:
            return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
        else:
            alpha, evalf, evalgf =  arm

        xk = xk+alpha* d
        k +=1
    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf':- evalf, 'evalgf': -evalgf}

def bfgs_hessian(Hk_1, gradfk, gradfk_1, alpha, d):
    s = alpha * d # ( xk - xk_1 )
    y = gradfk - gradfk_1

    I = np.eye(Hk_1.shape[0])
    rho = 1/(y.T @ s) if (y.T @ s) != 0 else 1e-10
    z = np.outer(s, y.T)
    Hk = (I - rho * z) @ Hk_1 @(I - rho * z.T) + rho * np.outer(s, s.T)

    return Hk


def BFGS(xk, fx, MAX_ITER = params.MAX_ITER_BFGS, EPSILON = params.EPSILON):
    k = 0
    Hk =  np.eye(len(xk))  # Initial aproximation to the hessian of fx

    status = False
    evalf, evalgf = 0,0

    gradfk = fx.grad(xk)
    evalgf +=1

    while (k < MAX_ITER): # stopping criterion
        if (np.linalg.norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
            status = True
            return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
        d = np.dot(Hk, -fx.grad(xk))

        arm= armijo(fx, d, xk, evalf, evalgf, values = [fx.obj(xk)])
        evalf+=1
        if arm == False:
            return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
        else:
            alpha, evalf, evalgf =  arm
        
        xk = xk + alpha * d
        gradfk_1 = gradfk
        gradfk =  fx.grad(xk)
        evalgf+=1

        Hk = bfgs_hessian(Hk, gradfk, gradfk_1,  alpha, d)
        
        k+=1

    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}

def sgradient_direction(xk_1, xk,  gradfk, gradfk_1):
    s = xk - xk_1
    y = gradfk - gradfk_1
    skyk = np.dot(s, y)
    sksk = np.dot(s, s)

    if  skyk> 0 and sksk != 0:
        sigma = skyk/sksk
        if sigma == np.nan:
            norm = np.linalg.norm(xk, ord=np.inf)
            sigma  = 1e-4 * np.linalg.norm(gradfk, ord=np.inf) / (max(1.0, norm ))

    else:
        norm = np.linalg.norm(xk, ord=np.inf)
        sigma  = 1e-4 * np.linalg.norm(gradfk, ord=np.inf) / (max(1.0, norm ))


    sigma = max(1e-30, min(sigma, 1e30))

    d = -(1/sigma) *gradfk

    return d


def spectral_gradient(xk_1, fx,  step = 'simple', EPSILON = params.EPSILON, MAX_ITER = params.MAX_ITER_SG) :
    k=1
    evalf, evalgf = 0,0

    gradfk_1 = fx.grad(xk_1)
    evalgf +=1

    status = False

    ## Finding x1
    delta = (2.22e-16)*(1/2)
    norm =  np.linalg.norm(xk_1, ord = np.inf)
    alpha = (delta * max(delta,norm) )/ (np.linalg.norm(gradfk_1, ord=np.inf))
    d = -alpha* gradfk_1
    alpha, evalf, evalgf = armijo(fx, d, xk_1, evalf, evalgf, values = [fx.obj(xk_1)])
    evalf+=1
    xk = xk_1 + alpha * d
    

    match step:
        case 'armijo':
            while (k < MAX_ITER): # stopping criterion
                gradfk = fx.grad(xk)
                evalgf +=1
                if (np.linalg.norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
                    status = True
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
                d = sgradient_direction(xk_1, xk, gradfk, gradfk_1)
                
                arm= armijo(fx, d, xk, evalf, evalgf, values = [fx.obj(xk)])
                evalf+=1
                if arm == False:
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
                else:
                    alpha, evalf, evalgf =  arm

                xk_1 = xk
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
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
                
                d = sgradient_direction(xk_1, xk, gradfk, gradfk_1)

                if len(values) >= 10:
                    values.pop(0)
                values.append(fx.obj(xk))
                evalf+=1

                arm=  armijo(fx, d, xk, evalf, evalgf, values = values)
                if arm == False:
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
                else:
                    alpha, evalf, evalgf =  arm


                xk_1 = xk
                xk = xk+alpha* d
                k +=1
                gradfk_1 = gradfk

        case 'simple':
            while (k < MAX_ITER): # stopping criterion
                gradfk = fx.grad(xk)
                evalgf +=1
                if (np.linalg.norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
                    status = True
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k-1, 'evalf': evalf, 'evalgf': evalgf}
                d = sgradient_direction(xk_1, xk, gradfk, gradfk_1)

                xk_1 = xk
                xk = xk + d
                if math.isnan(fx.obj(xk)):
                    print(fx.obj(xk), xk_1, xk, gradfk, gradfk_1, d)
                k +=1
                gradfk_1 = gradfk
    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k-1, 'evalf': -evalf, 'evalgf': -evalgf}