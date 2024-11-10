from tabulate import tabulate
import pycutest as pqt
import methods
import pandas as pd
import time

def df_evaluations(tests: list, meths: dict):
    data =[]
    for problem in tests:
        evals = {('Problem', ''): problem}
        for name, method in meths.items():
            fx = pqt.import_problem(problem)
            x0 =fx.x0

            if name == 'Spectral':
                for step in ['simple', 'armijo', 'armijo_mod']:
                    start = time.time()
                    dict = methods.spectral_gradient(x0, fx, step=step)
                    end = time.time()
                    exec_time = end-start

                    evalf =  dict['evalf']
                    evalgf =  dict['evalgf']
                    grad_n = max(abs(fx.grad(dict['xk'])))

                    evals = evals |{(f'SG step: {step}', 'f(x)'): fx.obj(dict['xk']), (f'SG step: {step}', '|\u2207f(x)|'): grad_n,(f'SG step: {step}', 'k'): dict['iter'],(f'SG step: {step}', 'evalf(x)'): evalf, (f'SG step: {step}', ' eval\u2207f(x)'): evalgf, (f'SG step: {step}', 'tempo (s)'): exec_time}

            else:
                start = time.time()
                dict = method(x0, fx)
                end = time.time()
                exec_time = end-start

                evalf =  dict['evalf']
                evalgf =  dict['evalgf']
                grad_n = max(abs(fx.grad(dict['xk'])))
                evals = evals |{(f'{name}', 'f(x)'): fx.obj(dict['xk']), (f'{name}', '|\u2207f(x)|'): grad_n, (f'{name}', 'k'): dict['iter'], (f'{name}', 'evalf(x)'): evalf, (f'{name}', ' eval\u2207f(x)'): evalgf, (f'{name}', 'tempo (s)'): exec_time}
        data.append(evals)
    df = pd.DataFrame(data)
    df.columns= pd.MultiIndex.from_tuples(df.columns)
        
    return df


meths = {'Cauchy':methods.gradient_descent, 'BFGS':methods.BFGS, 'Spectral':methods.spectral_gradient}
tests = ["ARGLINA", "ARGLINB", "BA-L1SPLS", "BIGGS6", "BROWNAL", "COATING", 
         "FLETCHCR", "GAUSS2LS", "GENROSE", "HAHN1LS", "HEART6LS", "HILBERTB", 
        "HYDCAR6LS", "LANCZOS1LS", "LANCZOS2LS", "LRIJCNN1", "LUKSAN12LS", 
        "LUKSAN16LS", "OSBORNEA", "PALMER1C", "PALMER3C", "PENALTY2", "PENALTY3", 
        "QING", "ROSENBR", "STRTCHDV", "TESTQUAD", "THURBERLS", "TRIGON1", "TOINTGOR"]

df = df_evaluations(tests, meths)
df.to_csv('TESTE.csv')