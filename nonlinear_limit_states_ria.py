from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Constant
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# NONLINEAR LIMIT STATE

def lsf(d, G):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Constant('d1', d[0]))
    stochastic_model.addVariable(Constant('d2', d[1]))
    stochastic_model.addVariable(Normal('p1', 5., 0.3*5.))
    stochastic_model.addVariable(Normal('p2', 3., 0.3*3.))
    limit_state = LimitState(G)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(d):
    return d[0] ** 2 + d[1] ** 2

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d):
    df_d1 = 2. * d[0]
    df_d2 = 2. * d[1]
    return np.array([df_d1, df_d2])

# LIMIT STATES FUNCTIONS
def G1(d1, d2, p1, p2):
    global Geval
    Geval += 1
    return (d1 * d2 * p2 ** 2)/5. - p1

cons = ({'type': 'ineq', 'fun': lambda d: lsf(d, G1) / betaT - 1})

# TARGET RELIABILITY INDEX
betaT = [2.32]

# BOUNDS
bnds = [[0, 15.], [0, 15.]]

# INITIAL POINTS
d0 = [[2., 2.], [12., 12.]] # Modify according to the problem

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=d0[1], method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)