from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# ROOF TRUSS

betaT_list    = np.array([2.0, 2.5, 3.0, 3.5, 4.0])             # TARGET RELIABILITY INDEX
lambda_r_list = np.array([0.00, 0.25, 0.50, 0.75, 1.00])        # ROBUSTNESS WEIGHT

cost_opt_list  = []
cost_mean_list = []
cost_std_list  = []

# ROBUST OBJECTIVE FUNCTION
def f(x):
    np.random.seed(999)
    N = 1000
    x1 = np.random.normal(x[0], 5.9853e-5, N)
    x2 = np.random.normal(x[1], 480e-5, N)
    f = 20224. * x1 + 364. * x2
    μf = np.mean(f)
    σf = np.std(f)
    return μf + λr * σf

# LIMIT STATES FUNCTIONS
def lsf(x, g):
    PMA = True
    options = AnalysisOptions()
    options.is_PMA = PMA
    options.beta_targ = betaT
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0], 5.9853e-5))
    stochastic_model.addVariable(Normal('x2', x[1], 480e-5))
    stochastic_model.addVariable(Normal('p1', 2e4, 1.4e3))
    stochastic_model.addVariable(Normal('p2', 12., 0.12))
    stochastic_model.addVariable(Normal('p3', 1e11, 6e9))
    stochastic_model.addVariable(Normal('p4', 2e10, 1.2e9))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    return Analysis.G
def g1(x1, x2, p1, p2, p3, p4):
    return 0.03 - (p1 * p2**2 / 2.) * (3.81 / (x2 * p4) + 1.13 / (x1 * p3))
constraints = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1)})

# BOUNDS
bounds = [[0.0006, 0.0012], [0.0180, 0.0630]]

# INITIAL POINTS
x0 = [0.001, 0.042]

# OPTIMIZATION
for λr in lambda_r_list:

    cost_opt_row  = []
    cost_mean_row = []
    cost_std_row  = []

    for bt in betaT_list:
        betaT = [bt]
        results = minimize(fun=f, x0=x0, method='slsqp', bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})
        np.random.seed(999)
        N = 1000
        x1 = np.random.normal(results.x[0], 5.9853e-5, N)
        x2 = np.random.normal(results.x[1], 480e-5, N)
        f_val = 20224. * x1 + 364. * x2

        cost_opt_row.append(results.fun)
        cost_mean_row.append(np.mean(f_val))
        cost_std_row.append(np.std(f_val))

    cost_opt_list.append(cost_opt_row)
    cost_mean_list.append(cost_mean_row)
    cost_std_list.append(cost_std_row)

Beta, Lambda = np.meshgrid(betaT_list, lambda_r_list)

# RESULTS

# COST TRADE-OFF SURFACE
plt.figure(figsize=(6.5, 5.0))
plt.contourf(Beta, Lambda, np.array(cost_opt_list).T)
plt.colorbar()
plt.xlabel(r'$\beta^T_i$')
plt.ylabel(r'$\lambda_r$')
plt.xticks(betaT_list)
plt.yticks(lambda_r_list)
plt.show()

# PARETO FRONT
plt.figure(figsize=(6.5, 5.0))
plt.scatter(np.ravel(cost_mean_list), np.ravel(cost_std_list))
for j, bt in enumerate(betaT_list):
    plt.annotate(f'{bt:.1f}', (cost_mean_list[0][j], cost_std_list[0][j]), textcoords='offset points', xytext=(6, 6))
plt.xlabel(r'$\mu_f$')
plt.ylabel(r'$\sigma_f$')
plt.show()