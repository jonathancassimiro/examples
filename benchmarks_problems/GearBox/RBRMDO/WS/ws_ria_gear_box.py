from scipy.optimize import minimize
from pystra import Normal, Lognormal, LimitState, AnalysisOptions, StochasticModel, Form
import numpy as np
import time

execucao = time.time()
Geval = 0

volume = []
stress = []

# RELIABILITY INDEX APPROACH (RIA)
def RIA(x, G):
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

# OBJECTIVE FUNCTION 1
def f1(x):
    np.random.seed(seed=999)
    N = 1000
    x1 = np.random.normal(loc=x[0], scale=0.005, size=N)
    x2 = np.random.normal(loc=x[1], scale=0.005, size=N)
    x3 = np.random.normal(loc=x[2], scale=0.005, size=N)
    x4 = np.random.normal(loc=x[3], scale=0.005, size=N)
    x5 = np.random.normal(loc=x[4], scale=0.005, size=N)
    x6 = np.random.normal(loc=x[5], scale=0.005, size=N)
    x7 = np.random.normal(loc=x[6], scale=0.005, size=N)
    V1 = []
    for i in range(N):
        V = 0.7854*x1[i]*x2[i]**2*(3.3333*x3[i]**2 + 14.9334*x3[i] - 43.0934) - 1.508*x1[i]*(x6[i]** 2 + x7[i]**2) + 7.477*(x6[i]**3 + x7[i]**3) + 0.7854*(x4[i]*x6[i]**2 + x5[i]*x7[i]**2)
        V1.append(V)
    Vmed = np.mean(V1, axis=0)
    Vmin = np.min(np.abs(Vmed))
    return Vmin

# OBJECTIVE FUNCTION 2
def f2(x):
    np.random.seed(seed=999123)
    N = 1000
    x1 = np.random.normal(loc=x[0], scale=0.005, size=N)
    x2 = np.random.normal(loc=x[1], scale=0.005, size=N)
    x3 = np.random.normal(loc=x[2], scale=0.005, size=N)
    x4 = np.random.normal(loc=x[3], scale=0.005, size=N)
    x5 = np.random.normal(loc=x[4], scale=0.005, size=N)
    x6 = np.random.normal(loc=x[5], scale=0.005, size=N)
    x7 = np.random.normal(loc=x[6], scale=0.005, size=N)
    S1 = []
    for i in range(N):
        S = 1100. - np.sqrt((745*x4[i]/(x2[i]*x3[i]))**2 + 16.9 * 10e6) / (0.10*x6[i]**3)
        S1.append(S)
    Smed = np.mean(S1, axis=0)
    Smax = np.min(np.abs(Smed))
    return Smax

# ROBUST OBJECTIVE FUNCTION
def f(x):
    Vmin = f1(x)
    Smax = f2(x)
    return w1 * Vmin / Vmin0 + w2 * Smax / Smax0

# LIMIT STATES FUNCTIONS
def G1(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - 27. * (x1 * x2 ** 2 * x3) ** -1
def G2(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - 397.5 * (x1 * x2 ** 2 * x3 ** 2) ** -1
def G3(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - 1.93 * x4 ** 3 * (x2 * x3 * x6 ** 4) ** -1
def G4(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - 1.93 * x5 ** 3 * (x2 * x3 * x7 ** 4) ** -1
def G5(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1100. - np.sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (0.10 * x6 ** 3)
def G6(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 850. - np.sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) * (0.10 * x7 ** 3) ** -1
def G7(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 40. - x2 * x3
def G8(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return x1 * x2 ** -1 - 5.
def G9(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 12. - x1 * x2 ** -1
def G10(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - (1.5 * x6 + 1.9) * x4 ** -1
def G11(x1, x2, x3, x4, x5, x6, x7):
    global Geval
    Geval += 1
    return 1. - (1.1 * x7 + 1.9) * x5 ** -1

cons = ({'type': 'ineq', 'fun': lambda x: RIA(x,  G1) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x,  G2) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x,  G3) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x,  G4) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x,  G5) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x,  G6) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x,  G7) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x,  G8) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x,  G9) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x, G10) / betaT - 1},
        {'type': 'ineq', 'fun': lambda x: RIA(x, G11) / betaT - 1})

# TARGET RELIABILITY INDEX
betaT = [3.]

# BOUNDS
bnds = [[2.60, 3.60], [0.70, 0.80], [17.0, 28.0], [7.30, 8.30], [7.30, 8.30], [2.90, 3.90], [5.00, 5.50]]

# INITIAL POINT
x0 = [3.50, 0.70, 17.0, 7.30, 7.72, 3.35, 5.29]

# NORMALIZATION
Vmin0 = f1(x0)
Smax0 = f2(x0)

# DETERMINISTIC OPTIMIZATION
def optimization(x0):
    res = minimize(f, x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp':True, 'maxiter': 1000, 'ftol': 1e-4})
    mean_x = res.x
    volume.append(f1(mean_x))
    stress.append(f2(mean_x))
    return volume, stress, res

# CONVERGENCE HISTORY
def history(w1, w2, volume, stress, res):
    print(res)
    print('w1:', w1)
    print('w2:', w2)
    print('f1:', volume)
    print('f2:', stress)
    print('Processing Time:', round((time.time() - execucao), 3), 's')
    print('Limit State Functions Evaluations:', Geval)
    print('-----------------------------------------------')

# PARETO POINTS
nPar = 15
W1 = np.linspace(1., 0., num=nPar)
W2 = np.linspace(0., 1., num=nPar)

for w1, w2 in zip(W1, W2):
    if w1 + w2 == 1:
        volume, stress, res = optimization(x0)
        history(w1, w2, volume, stress, res)

# RESULTS
print('Mean volume:', volume)
print('Mean stress:', stress)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)