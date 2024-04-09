from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Constant
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# MULTIPLE LIMIT STATE

def lsf(x, G):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0],  0.3))
    stochastic_model.addVariable(Normal('x2', x[1],  0.3))
    limit_state = LimitState(G)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(x):
    return x[0] + x[1]

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_dx1 = 1.
    df_dx2 = 1.
    return np.array([[df_dx1, df_dx2]])

# LIMIT STATES FUNCTIONS
def G1(x1, x2):
    global Geval
    Geval += 1
    return x1**2 * x2/20. - 1.
def G2(x1, x2):
    global Geval
    Geval += 1
    return (x1 + x2 - 5.)**2/30. + (x1 - x2 - 12.)**2/120. - 1.
def G3(x1, x2):
    global Geval
    Geval += 1
    return 80./(x1**2 + 8.*x2 + 5.) - 1.

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, G1) / betaT[2] - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x, G2) / betaT[2] - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x, G3) / betaT[2] - 1})

# TARGET RELIABILITY INDEX
betaT = [2., 3., 4.] # Modify according to the problem

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