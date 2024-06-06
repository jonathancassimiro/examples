from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Constant
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# SHORT COLUMN

def lsf(x, g):
    PMA = True
    options = AnalysisOptions()
    options.is_PMA = PMA
    options.beta_targ = betaT
    stochastic_model = StochasticModel()
    # stochastic_model.addVariable(Constant('x1',  x[0]))                # Em caso de otimização deterministica
    # stochastic_model.addVariable(Constant('x2',  x[1]))                # Em caso de otimização deterministica
    stochastic_model.addVariable(Normal('x1',  x[0], CoV[2]*x[0]))
    stochastic_model.addVariable(Normal('x2',  x[1], CoV[2]*x[1]))
    stochastic_model.addVariable(Normal('p1',  4e4,  4e4*0.10))
    stochastic_model.addVariable(Normal('p2', 2500, 2500*0.20))
    stochastic_model.addVariable(Normal('p3',  250,  250*0.30))
    stochastic_model.addVariable(Normal('p4',  125,  125*0.30))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    return Analysis.G

# OBJECTIVE FUNCTION
def f(x):
    return x[0] * x[1]

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(x):
    df_dx1 = x[1]
    df_dx2 = x[0]
    return np.array([[df_dx1, df_dx2]])

# LIMIT STATES FUNCTIONS
def g1(x1, x2, p1, p2, p3, p4):
    global Geval
    Geval += 1
    return 1 - 4*p3/(x1*x2**2*p1) - 4*p4/(x1**2*x2*p1) - p2**2/((x1*x2*p1)**2)

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1)})

# TARGET RELIABILITY INDEX
betaT = [3.]

# COEFFICIENTS OF VARIATION
CoV = [0.05, 0.10, 0.15] # Modify according to the problem

# INITIAL POINT
x0 = [0.5, 0.5]

# OPTIMIZATION
res = minimize(fun=f, jac=df, x0=x0, method='slsqp', constraints=cons, options={'disp': True, 'maxiter':300, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)