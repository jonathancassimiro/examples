from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Gumbel, Lognormal, Weibull
from scipy.optimize import minimize
import numpy as np
import time

start_time = time.time()
Geval = 0

# AUTOMOBILE WITH FRONT I-BEAM AXLE

def lsf(x, g):
    options = AnalysisOptions()
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0],  0.05 * x[0]))
    stochastic_model.addVariable(Normal('x2', x[1],  0.05 * x[1]))
    stochastic_model.addVariable(Normal('x3', x[2],  0.05 * x[2]))
    stochastic_model.addVariable(Gumbel('p1', 100, 0.15 * 100))
    stochastic_model.addVariable(Gumbel('p2', 200e6, 0.08 * 200e6))
    stochastic_model.addVariable(Lognormal('p3', 225e3, 0.08 * 225e3))
    stochastic_model.addVariable(Weibull('p4', 7.86, 0.10 * 7.86))
    stochastic_model.addVariable(Normal('p5',  5., 0.05 * 5.))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    beta = Analysis.getBeta()
    return beta

# OBJECTIVE FUNCTION
def f(x):
    return 7.86 * x[2] * 5. * ((4. * np.sqrt(3) / 9.) * x[0] + x[1])

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_x1 = 52.4 * x[2] * np.sqrt(3) / 3.
    df_x2 = 39.3 * x[2]
    df_x3 = 52.4 * x[0] * np.sqrt(3) / 3. + 39.3 * x[1]
    return np.array([df_x1, df_x2, df_x3])

# LIMIT STATES FUNCTIONS
def g1(x1, x2, x3, p1, p2, p3, p4, p5):
    global Geval
    Geval += 1
    Mb = p1 * p5 / 3. + p4 * 9.81 * x2 * x3 * p5**2 / 18.
    sigb = 6. * Mb / (x2 * x3**2)
    return p3 - sigb
def g2(x1, x2, x3, p1, p2, p3, p4, p5):
    global Geval
    Geval += 1
    Fb = 9. * np.pi**2 * p2 * x3 * x1**3 * np.sin(np.deg2rad(60))**2 / (48. * p5**2)
    Fab = (1 / np.cos(np.deg2rad(60))) * ((3. * p1 / 2.) + (3. * p4 * 9.81 * x2 * x3 * p5 / 4.))
    return Fb - Fab

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1) - betaT},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g2) - betaT})

# TARGET RELIABILITY INDEX
betaT = [2.]

# BOUNDS
bnds = [[0.050, 0.300], [0.050, 0.300], [0.050, 0.300]]

# INITIAL POINTS
x0 = [0.061, 0.202, 0.269]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - start_time),3), 's')
print('Limit State Functions Evaluations:', Geval)