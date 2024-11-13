from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Lognormal
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# TENSION/COMPRESSION SPRING

def lsf(x, g):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0], 0.001**2))
    stochastic_model.addVariable(Normal('x2', x[1], 0.010**2))
    stochastic_model.addVariable(Normal('x3', x[2], 0.800**2))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(x):
    return (x[2] + 2.) * x[1] * x[0]**2

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_x1 = 2. * (x[2] + 2.) * x[1] * x[0]
    df_x2 = (x[2] + 2.) * x[0]**2
    df_x3 = x[1] * x[0]**2
    return np.array([df_x1, df_x2, df_x3])

# LIMIT STATES FUNCTIONS
def g1(x1, x2, x3):
    global Geval
    Geval += 1
    return x2**3 * x3 / (71785. * x1**4) - 1.
def g2(x1, x2, x3):
    global Geval
    Geval += 1
    return 1. - (4. * x2**2 - x1 * x2) / (12566. * (x2 * x1**3 - x1**4)) + 1. / (5108. * x1**2)
def g3(x1, x2, x3):
    global Geval
    Geval += 1
    return 140.45 * x1 / (x2**2 * x3) - 1.
def g4(x1, x2, x3):
    global Geval
    Geval += 1
    return 1. - (x1 + x2) / 1.5

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g2) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g3) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g4) - betaT})

# TARGET RELIABILITY INDEX
betaT = [3.]

# BOUNDS
bnds = [[0.01, 0.10], [0.10, 1.00], [5.00, 15.0]]

# INITIAL POINTS
x0 = [0.05, 0.50, 10.0]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)