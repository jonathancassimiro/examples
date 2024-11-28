from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Gumbel, Lognormal, Weibull
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# STEEL T-COLUMN

def lsf(x, g):
    PMA = True
    options = AnalysisOptions()
    options.is_PMA = PMA
    options.beta_targ = betaT
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Lognormal('x1', x[0], 3.))
    stochastic_model.addVariable(Lognormal('x2', x[1], 2.))
    stochastic_model.addVariable(Lognormal('x3', x[2], 5.))
    stochastic_model.addVariable(Lognormal('p1', 0.400, 0.035))
    stochastic_model.addVariable(Normal('p2', 30., 10.))
    stochastic_model.addVariable(Weibull('p3', 21., 4.2))
    stochastic_model.addVariable(Normal('p4', 500., 50.))
    stochastic_model.addVariable(Gumbel('p5', 600., 90.))
    stochastic_model.addVariable(Gumbel('p6', 600., 90.))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    return Analysis.G

# OBJECTIVE FUNCTION
def f(x):
    return x[0] * x[1] + 5 * x[2]

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_x1 = x[1]
    df_x2 = x[0]
    df_x3 = 5.
    return np.array([df_x1, df_x2, df_x3])

# LIMIT STATES FUNCTIONS
def g1(x1, x2, x3, p1, p2, p3, p4, p5, p6):
    global Geval
    Geval += 1
    L = 7500
    F = p4 + p5 + p6
    As = 2. * x1 * x2
    ms = x1 * x2 * x3
    eb = np.pi**2 * p3 * x1 * x2 * x3**2 / (2. * L**2)
    return p1 - F * ((1. / As) + (p2 * eb / (ms * (eb - F))))

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1)})

# TARGET RELIABILITY INDEX
betaT = [3.13217]

# BOUNDS
bnds = [[200., 400.], [10., 30.], [100., 500.]]

# INITIAL POINTS
x0 = [300., 20., 300.]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)