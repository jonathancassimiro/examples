from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# AUTOMOBILE WITH FRONT I-BEAM AXLE

def lsf(x, g):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1',  x[0],  0.060))
    stochastic_model.addVariable(Normal('x2',  x[1],  0.325))
    stochastic_model.addVariable(Normal('x3',  x[2],  0.070))
    stochastic_model.addVariable(Normal('x4',  x[3],  0.425))
    stochastic_model.addVariable(Normal('p1', 3.5e6, 1.75e5))
    stochastic_model.addVariable(Normal('p2', 3.1e6, 1.55e5))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(x):
    return x[3] * x[0] + 2. * (x[1] - x[0]) * x[2]

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_x1 = x[3] - 2. * x[2]
    df_x2 = 2. * x[2]
    df_x3 = 2. * x[1] - 2. * x[0]
    df_x4 = x[0]
    return np.array([df_x1, df_x2, df_x3, df_x4])

# LIMIT STATES FUNCTIONS
def g1(x1, x2, x3, x4, p1, p2):
    global Geval
    Geval += 1
    Wx = x1 * (x4 - 2. * x3) ** 3 / (6. * x4) + x2 * (x4 ** 3 - (x4 - 2. * x3) ** 3) / (6. * x4)
    Wp = 0.8 * x2 * x3 ** 2 + 0.4 * x1 **3 * (x4 - 2. * x3) / x3
    gamas = 460
    gamam = p1 / Wx
    tau = p2 / Wp
    return gamas - np.sqrt((gamam ** 2) + 3. * (tau ** 2))

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1) - betaT})

# TARGET RELIABILITY INDEX
betaT = [3.]

# BOUNDS
bnds = [[10., 20.], [60., 100.], [10., 20.], [70., 120.]]

# INITIAL POINTS
x0 = [12., 75., 12., 85.]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)