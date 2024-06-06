from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# HIGHLY NONLINEAR

def lsf(x, g):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0],  0.3**2))
    stochastic_model.addVariable(Normal('x2', x[1],  0.3**2))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(x):
    return -(x[0] + x[1] - 10.)**2 / 30. - (x[0] - x[1] + 10.)**2 / 120.

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_dx1 = -x[0] / 12. - x[1] / 20. + 1/2.
    df_dx2 = -x[0] / 20. - x[1] / 12. + 5/6.
    return np.array([[df_dx1, df_dx2]])

# LIMIT STATES FUNCTIONS
def g1(x1, x2):
    global Geval
    Geval += 1
    return x1 ** 2 * x2 / 20. - 1.
def g2(x1, x2):
    global Geval
    Geval += 1
    Y = 0.9063 * x1 + 0.4226 * x2
    Z = 0.4226 * x1 - 0.9063 * x2
    return 1. - (Y - 6.)**2 - (Y - 6.)**3 + 0.6 * (Y - 6.)**4 - Z
def g3(x1, x2):
    global Geval
    Geval += 1
    return 80. / (x1**2 + 8. * x2 + 5.) - 1.

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g2) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g3) - betaT})

# TARGET RELIABILITY INDEX
betaT = [3.]

# BOUNDS
bnds = [[0, 10.], [0, 10.]]

# INITIAL POINT
x0 = [5., 5.]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)