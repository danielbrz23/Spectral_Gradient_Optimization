from tabulate import tabulate
import pycutest as pqt
import methods
import pandas as pd
import time

def df_evaluations(tests):
    meths = {'Cauchy':methods.gradient_descent, 'BFGS':methods.BFGS, 'Spectral':methods.spectral_gradient}
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
                    evalgf =  dict['evalf']

                    evals = evals |{(f'SG step: {step}', 'f(x)'): evalf, (f'SG step: {step}', ' \u2207f(x)'): evalgf, (f'SG step: {step}', 'tempo (s)'): round(exec_time,3)}

            else:
                start = time.time()
                dict = method(x0, fx)
                end = time.time()
                exec_time = end-start

                evalf =  dict['evalf']
                evalgf =  dict['evalf']

                evals = evals |{(f'{name}', 'f(x)'): evalf, (f'{name}', ' \u2207f(x)'): evalgf, (f'{name}', 'tempo (s)'): exec_time}
        data.append(evals)
    df = pd.DataFrame(data)
    df.columns= pd.MultiIndex.from_tuples(df.columns)
        
    return df



tests = [
        "ARGLINA",
        "ARGLINB",
        "BA-L1SPLS",
        "BIGGS6",
        "BROWNAL",
        "COATING",
        "FLETCHCR",
        "GAUSS2LS",
        "GENROSE",
        "HAHN1LS",
        "HEART6LS",
        "HILBERTB",
        "HYDCAR6LS",
        "LANCZOS1LS",
        "LANCZOS2LS",
        "LRIJCNN1",
        "LUKSAN12LS",
        "LUKSAN16LS",
        "OSBORNEA",
        "PALMER1C",
        "PALMER3C",
        "PENALTY2",
        "PENALTY3",
        "QING",
        "ROSENBR",
        "STRTCHDV",
        "TESTQUAD",
        "THURBERLS",
        "TRIGON1",
        "TOINTGOR",
    ]

meths = {'Spectral':methods.spectral_gradient}
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
                evalgf =  dict['evalf']

                evals = evals |{(f'SG step: {step}', 'f(x)'): evalf, (f'SG step: {step}', ' \u2207f(x)'): evalgf, (f'SG step: {step}', 'tempo (s)'): round(exec_time,3)}

        else:
            start = time.time()
            dict = method(x0, fx)
            end = time.time()
            exec_time = end-start

            evalf =  dict['evalf']
            evalgf =  dict['evalf']

            evals = evals |{(f'{name}', 'f(x)'): evalf, (f'{name}', ' \u2207f(x)'): evalgf, (f'{name}', 'tempo (s)'): exec_time}
    data.append(evals)
df = pd.DataFrame(data)
df.columns= pd.MultiIndex.from_tuples(df.columns)


df.to_csv('results.csv')
print(df)
