import numpy as np
import params 

def armijo(fx, d, xk, evalf, evalgf, values= None, MAX_ITER =  params.MAX_ITER_ARMIJO):
    eta = params.ETA
    beta = params.BETA
    alpha = 1
    if values == None:
        values = [fx.obj(xk)]
        evalf +=1

    term = eta * np.dot((fx.grad(xk)), d)
    evalgf+=1

    max_fx  = float(max(values))
    i=0
    while (fx.obj(xk+alpha *d) >= max_fx + alpha * term and i<=MAX_ITER): 
        alpha *= beta 
        i+=1
    evalf +=i
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
            return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
        d = -gradf # steepest descent direction

        arm = armijo(fx, d, xk, evalf, evalgf)
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

        arm= armijo(fx, d, xk, evalf, evalgf)
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
    if np.dot(s, y) > 0:
        sigma = np.dot(s, y)/np.dot(s, s)
    else:
        sigma =1e-4 *  np.linalg.norm(gradfk,ord = np.inf)/max(1.0, np.linalg.norm(xk, ord=np.inf))


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
    alpha = ( (EPSILON**(1/2))* max((EPSILON**(1/2)), np.linalg.norm(xk_1, ord = np.inf)) )/ (np.linalg.norm(gradfk_1, ord=np.inf))
    d = -alpha* gradfk_1
    alpha, evalf, evalgf = armijo(fx, d, xk_1, evalf, evalgf)
    xk = alpha * d
    

    match step:
        case 'armijo':
            while (k < MAX_ITER): # stopping criterion
                gradfk = fx.grad(xk)
                evalgf +=1
                if (np.linalg.norm(gradfk, ord=np.inf) < EPSILON): # Optimality condition
                    status = True
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': evalf, 'evalgf': evalgf}
                d = sgradient_direction(xk_1, xk, gradfk, gradfk_1)
                
                arm = armijo(fx, d, xk, evalf, evalgf)
                if arm == False:
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
                else:
                    alpha, evalf, evalgf =  arm

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

                arm=  armijo(fx, d, xk, evalf, evalgf, values = values)
                if arm == False:
                    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k, 'evalf': -evalf, 'evalgf': -evalgf}
                else:
                    alpha, evalf, evalgf =  arm

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

                xk = xk+ d
                k +=1
                gradfk_1 = gradfk
    return {'status': status, 'xk': xk, 'fx': fx.obj(xk), 'iter': k-1, 'evalf': -evalf, 'evalgf': -evalgf}