from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# HOCK AND SCHITTKOWSKI 113

def lsf(x, g):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0],  0.02**1))
    stochastic_model.addVariable(Normal('x2', x[1],  0.02**1))
    stochastic_model.addVariable(Normal('x3', x[2],  0.02**1))
    stochastic_model.addVariable(Normal('x4', x[3],  0.02**1))
    stochastic_model.addVariable(Normal('x5', x[4],  0.02**1))
    stochastic_model.addVariable(Normal('x6', x[5],  0.02**1))
    stochastic_model.addVariable(Normal('x7', x[6],  0.02**1))
    stochastic_model.addVariable(Normal('x8', x[7],  0.02**1))
    stochastic_model.addVariable(Normal('x9', x[8],  0.02**1))
    stochastic_model.addVariable(Normal('x10', x[9],  0.02**1))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(x):
    return x[0]**2 + x[1]**2 + x[0] * x[1] - 14. * x[0] - 16. * x[1] + (x[2] - 10.)**2 + 4. * (x[3] - 5.)**2 + (x[4] - 3.)**2 + 2. * (x[5] - 1.)**2 + 5. * x[6]**2 + 7. * (x[7] - 11.)**2 + 2. * (x[8] - 10.)**2 + (x[9] - 7.)**2 + 45.

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_dx1 = 2. * x[0] + x[1] - 14.
    df_dx2 = 2. * x[1] + x[0] - 16.
    df_dx3 = 2. * x[2] - 20.
    df_dx4 = 8. * x[3] - 40.
    df_dx5 = 2. * x[4] - 6.
    df_dx6 = 4. * x[5] - 4.
    df_dx7 = 10. * x[6]
    df_dx8 = 14. * x[7] - 154.
    df_dx9 = 4. * x[8] - 40.
    df_dx10 = 2. * x[9] - 14.
    return np.array([[df_dx1, df_dx2, df_dx3, df_dx4, df_dx5, df_dx6, df_dx7, df_dx8, df_dx9, df_dx10]])

# LIMIT STATES FUNCTIONS
def g1(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    global Geval
    Geval += 1
    return 1. - (4. * x1 + 5. * x2 - 3. * x7 + 9. * x8) / 105.
def g2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    global Geval
    Geval += 1
    return -10. * x1 + 8. * x2 + 17. * x7 - 2. * x8
def g3(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    global Geval
    Geval += 1
    return 1. + (8. * x1 - 2. * x2 - 5. * x9 + 2. * x10) / 12.
def g4(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    global Geval
    Geval += 1
    return 1. - (3. * (x1 - 2.)**2 + 4. * (x2 - 3.)**2 + 2. * x3**2 - 7. * x4) / 120.
def g5(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    global Geval
    Geval += 1
    return 1. - (5. * x1**2 + 8. * x2 + (x3 - 6)**2 - 2. * x4) / 40.
def g6(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    global Geval
    Geval += 1
    return 1. - (0.5 * (x1 - 8.)**2 + 2. * (x2 - 4.)**2 + 3. * x5**2 - x6) / 30.
def g7(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    global Geval
    Geval += 1
    return -x1**2 - 2. * (x2 - 2.)**2 + 2. * x1 * x2 - 14. * x5 + 6. * x6
def g8(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    global Geval
    Geval += 1
    return 3. * x1 - 6. * x2 - 12. * (x9 - 8.)**2 + 7. * x10

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g2) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g3) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g4) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g5) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g6) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g7) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g8) - betaT})

# TARGET RELIABILITY INDEX
betaT = [3.]

# BOUNDS
bnds = [[0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf]]

# INITIAL POINT
x0 = [2.17, 2.36, 8.77, 5.10, 0.99, 1.43, 1.32, 9.83, 8.28, 8.38]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)