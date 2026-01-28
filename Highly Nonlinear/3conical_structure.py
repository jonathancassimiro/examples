from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Lognormal, Gumbel
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# CONICAL STRUCTURE

betaT_list    = np.array([2.0, 2.5, 3.0, 3.5, 4.0])             # TARGET RELIABILITY INDEX
lambda_r_list = np.array([0.00, 0.25, 0.50, 0.75, 1.00])        # ROBUSTNESS WEIGHT

cost_opt_list  = []
cost_mean_list = []
cost_std_list  = []

# ROBUST OBJECTIVE FUNCTION
def f(x):
    np.random.seed(999)
    N = 1000
    x1 = np.random.normal(x[0], 0.10, N)
    x2 = np.random.normal(x[1], 0.12, N)
    x3 = np.random.normal(x[2], 0.08, N)
    f = np.pi * x1 * (2 * x2 + np.sin(x3))
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
    stochastic_model.addVariable(Normal('x1', x[0], 0.10 * x[0]))
    stochastic_model.addVariable(Normal('x2', x[1], 0.12 * x[1]))
    stochastic_model.addVariable(Normal('x3', x[2], 0.08 * x[2]))
    stochastic_model.addVariable(Gumbel('p1', 7e4, 0.08 * 7e4))
    stochastic_model.addVariable(Gumbel('p2', 8e4, 0.08 * 8e4))
    stochastic_model.addVariable(Lognormal('p3', 7e10, 0.05 * 7e10))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    return Analysis.G
def g1(x1, x2, x3, p1, p2, p3):
    return 1. - 1.6523 / (np.pi * p3 * x1**2 * np.cos(x3)**2) * ((p1 / 0.66) + (p2 / (0.41 * x2)))
constraints = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1)})

# BOUNDS
bounds = [[0.0010, 0.0050], [0.800, 1.000], [np.pi/6, np.pi/4]]

# INITIAL POINTS
x0 = [0.0025, 0.900, 0.524]

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
        x1 = np.random.normal(results.x[0], results.x[0] * 0.10, N)
        x2 = np.random.normal(results.x[1], results.x[1] * 0.12, N)
        x3 = np.random.normal(results.x[2], results.x[2] * 0.08, N)
        f_val = np.pi * x1 * (2 * x2 + np.sin(x3))

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