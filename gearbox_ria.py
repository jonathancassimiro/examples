from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Constant
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# GEAR BOX

def lsf(x, G):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0], 0.005))
    stochastic_model.addVariable(Normal('x2', x[1], 0.005))
    stochastic_model.addVariable(Normal('x3', x[2], 0.005))
    stochastic_model.addVariable(Normal('x4', x[3], 0.005))
    stochastic_model.addVariable(Normal('x5', x[4], 0.005))
    stochastic_model.addVariable(Normal('x6', x[5], 0.005))
    stochastic_model.addVariable(Normal('x7', x[6], 0.005))
    limit_state = LimitState(G)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(x):
    return 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_dx0 = 0.7854 * x[1]**2 * (3.3333 * x[2]**2 + 14.9334 * x[2] - 43.0934) - 1.508 * (x[5]**2 + x[6]**2)
    df_dx1 = 2 * 0.7854 * x[0] * x[1] * (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934)
    df_dx2 = 2 * 0.7854 * x[0] * x[1] ** 2 * (3.3333 * x[2] + 14.9334)
    df_dx3 = 0.7854 * x[5] ** 2
    df_dx4 = 0.7854 * x[6] ** 2
    df_dx5 = -2 * 1.508 * x[0] * x[5] + 3 * 7.477 * x[5] ** 2 + 2 * 0.7854 * x[3] * x[5]
    df_dx6 = -2 * 1.508 * x[0] * x[6] + 3 * 7.477 * x[6] ** 2 + 2 * 0.7854 * x[4] * x[6]
    return np.array([df_dx0, df_dx1, df_dx2, df_dx3, df_dx4, df_dx5, df_dx6])

# LIMIT STATES FUNCTIONS
def G1(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - 27. * (x1 * x2**2 * x3)**-1
def G2(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - 397.5 * (x1 * x2**2 * x3**2)**-1
def G3(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - 1.93 * x4**3 * (x2 * x3 * x6**4)**-1
def G4(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - 1.93 * x5**3 * (x2 * x3 * x7**4)**-1
def G5(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1100. - np.sqrt((745*x4/(x2*x3))**2 + 16.9e6) / (0.10 * x6**3)
def G6(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 850.  - np.sqrt((745*x5/(x2*x3))**2 + 157.5e6) * (0.10 * x7**3)**-1
def G7(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 40. - x2 * x3
def G8(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return x1 * x2**-1 - 5.
def G9(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 12. - x1 * x2**-1
def G10(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - (1.5 * x6 + 1.9) * x4**-1
def G11(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - (1.1 * x7 + 1.9) * x5**-1

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x,  G1) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x,  G2) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x,  G3) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x,  G4) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x,  G5) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x,  G6) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x,  G7) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x,  G8) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x,  G9) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x, G10) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: lsf(x, G11) / betaT - 1})

# TARGET RELIABILITY INDEX
betaT = [3.]

# BOUNDS
bnds = [[2.60, 3.60], [0.70, 0.80], [17.0, 28.0], [7.30, 8.30], [7.30, 8.30], [2.90, 3.90], [5.00, 5.50]]

# INITIAL POINT
x0 = [3.50, 0.70, 17.0, 7.30, 7.72, 3.35, 5.29]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)