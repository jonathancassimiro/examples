from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Lognormal, Gumbel
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# CONICAL STRUCTURE

def lsf(x, g):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0], 0.10 * x[0]))
    stochastic_model.addVariable(Normal('x2', x[1], 0.12 * x[1]))
    stochastic_model.addVariable(Normal('x3', x[2], 0.08 * x[2]))
    stochastic_model.addVariable(Gumbel('p1', 7e4, 0.08 * 7e4))
    stochastic_model.addVariable(Gumbel('p2', 8e4, 0.08 * 8e4))
    stochastic_model.addVariable(Lognormal('p3', 7e10, 0.05 * 7e10))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(x):
    return np.pi * x[0] * (2 * x[1] + np.sin(x[2]))

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_x1 = np.pi * (2 * x[1] + np.sin(x[2]))
    df_x2 = 2. * np.pi * x[0]
    df_x3 = np.pi * x[0] * np.cos(x[2])
    return np.array([df_x1, df_x2, df_x3])

# LIMIT STATES FUNCTIONS
def g1(x1, x2, x3, p1, p2, p3):
    global Geval
    Geval += 1
    return 1. - 1.6523 / (np.pi * p3 * x1**2 * np.cos(x3)**2) * ((p1 / 0.66) + (p2 / (0.41 * x2)))

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1) - betaT})

# TARGET RELIABILITY INDEX
betaT = [4.]

# BOUNDS
bnds = [[0.001, 0.005], [0.800, 1.000], [np.pi/6, np.pi/4]]

# INITIAL POINTS
x0 = [0.0025, 0.900, 0.524]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)