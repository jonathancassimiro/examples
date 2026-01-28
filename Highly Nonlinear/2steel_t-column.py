from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Gumbel, Lognormal, Weibull
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# STEEL T-COLUMN

betaT_list    = np.array([2.0, 2.5, 3.0, 3.5, 4.0])             # TARGET RELIABILITY INDEX
lambda_r_list = np.array([0.00, 0.25, 0.50, 0.75, 1.00])        # ROBUSTNESS WEIGHT

cost_opt_list  = []
cost_mean_list = []
cost_std_list  = []

# ROBUST OBJECTIVE FUNCTION
def f(x):
    np.random.seed(999)
    N = 1000
    x1 = np.random.normal(x[0], 30, N)
    x2 = np.random.normal(x[1], 2, N)
    x3 = np.random.normal(x[2], 50, N)
    f = x1 * x2 + 5 * x3
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
    stochastic_model.addVariable(Normal('x1', x[0], 30.))
    stochastic_model.addVariable(Normal('x2', x[1], 2.))
    stochastic_model.addVariable(Normal('x3', x[2], 50.))
    stochastic_model.addVariable(Normal('p1', 0.400, 0.035))
    stochastic_model.addVariable(Normal('p2', 30., 10.))
    stochastic_model.addVariable(Weibull('p3', 21., 4.2))
    stochastic_model.addVariable(Normal('p4', 500., 50.))
    stochastic_model.addVariable(Gumbel('p5', 600., 90.))
    stochastic_model.addVariable(Gumbel('p6', 600., 90.))
    stochastic_model.addVariable(Normal('p7', 3000., 300.))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    return Analysis.G
def g1(x1, x2, x3, p1, p2, p3, p4, p5, p6, p7):
    F = p4 + p5 + p6
    As = 2. * x1 * x2
    ms = x1 * x2 * x3
    eb = np.pi**2 * p3 * x1 * x2 * x3**2 / (2. * p7**2)
    return p1 - F * ((1. / As) + (p2 * eb / (ms * (eb - F))))
constraints = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1)})

# BOUNDS
bounds = [[200., 400.], [10., 30.], [100., 500.]]

# INITIAL POINTS
x0 = [300., 20., 300.]

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
        x1 = np.random.normal(results.x[0], 30, N)
        x2 = np.random.normal(results.x[1], 2, N)
        x3 = np.random.normal(results.x[2], 50, N)
        f_val = x1 * x2 + 5 * x3

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