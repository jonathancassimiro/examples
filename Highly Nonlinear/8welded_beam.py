from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Lognormal
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# WELDED BEAM

betaT_list    = np.array([2.0, 2.5, 3.0, 3.5, 4.0])             # TARGET RELIABILITY INDEX
lambda_r_list = np.array([0.00, 0.25, 0.50, 0.75, 1.00])        # ROBUSTNESS WEIGHT

cost_opt_list  = []
cost_mean_list = []
cost_std_list  = []

# SYSTEM PARAMETERS
z1 = 2.6688e4
z2 = 3.556e2
z3 = 2.0685e5
z4 = 8.274e4
z5 = 6.35
z6 = 9.377e1
z7 = 2.0685e2
S1 = 6.74135e-5
S2 = 2.93585e-6

# ROBUST OBJECTIVE FUNCTION
def f(x):
    np.random.seed(999)
    N = 1000
    x1 = np.random.normal(x[0], 0.1693**2, N)
    x2 = np.random.normal(x[1], 0.1693**2, N)
    x3 = np.random.normal(x[2], 0.0107**2, N)
    x4 = np.random.normal(x[3], 0.0107**2, N)
    f = S1 * x1**2 * x2 + S2 * x3 * x4 * (z2 + x2)
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
    stochastic_model.addVariable(Normal('x1', x[0], 0.1693**2))
    stochastic_model.addVariable(Normal('x2', x[1], 0.1693**2))
    stochastic_model.addVariable(Normal('x3', x[2], 0.0107**2))
    stochastic_model.addVariable(Normal('x4', x[3], 0.0107**2))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    return Analysis.G
def g1(x1, x2, x3, x4):
    M = z1 * (z2 + x2 / 2.)
    R = np.sqrt(x2**2 + (x1 + x3)**2) / 2.
    J = np.sqrt(2) * x1 * x2 * (x2**2 / 12. + (x1 + x3)**2 / 4.)
    t = z1 / (np.sqrt(2) * x1 * x2)
    T = M * R / J
    tau = np.sqrt(t**2 + 2 * t * T * x2 / (2 * R) + T**2)
    return 1. - tau / z6
def g2(x1, x2, x3, x4):
    sigma = 6. * z1 * z2 / (x3**2 * x4)
    return 1. - sigma / z7
def g3(x1, x2, x3, x4):
    return 1. - x1 / x4
def g4(x1, x2, x3, x4):
    delta = 4. * z1 * z2**3 / (z3 * x3**3 * x4)
    return 1. - delta / z5
def g5(x1, x2, x3, x4):
    Fb = (4.013 * x3 * x4**3 * np.sqrt(z3 * z4) / (6. * z2**2)) * (1. - (x3 / (4. * z2)) * np.sqrt(z3 / z4))
    return Fb / z1 - 1.
constraints = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1)},
               {'type': 'ineq', 'fun': lambda x: lsf(x, g2)},
               {'type': 'ineq', 'fun': lambda x: lsf(x, g3)},
               {'type': 'ineq', 'fun': lambda x: lsf(x, g4)},
               {'type': 'ineq', 'fun': lambda x: lsf(x, g5)})

# BOUNDS
bounds = [[3.175, 50.80], [0.000, 254.0], [0.000, 254.0], [0.000, 50.80]]

# INITIAL POINTS
x0 = [6.208, 157.82, 210.62, 6.208]

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
        x1 = np.random.normal(results.x[0], 0.1693 ** 2, N)
        x2 = np.random.normal(results.x[1], 0.1693 ** 2, N)
        x3 = np.random.normal(results.x[2], 0.0107 ** 2, N)
        x4 = np.random.normal(results.x[3], 0.0107 ** 2, N)
        f_val = S1 * x1 ** 2 * x2 + S2 * x3 * x4 * (z2 + x2)

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