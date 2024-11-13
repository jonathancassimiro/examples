from codes import SLA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# STEEL T-COLUMN

# OBJECTIVE FUNCTION
def f(d, x):
    return x[0] * x[1] + 5. * x[2]

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_x1 = x[1]
    df_x2 = x[0]
    df_x3 = 5.
    return np.array([df_x1, df_x2, df_x3])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    L = 7500
    F = p[3] + p[4] + p[5]
    As = 2. * x[0] * x[1]
    ms = x[0] * x[1] * x[2]
    eb = np.pi**2 * p[2] * x[0] * x[1] * x[2]**2 / (2. * L**2)
    return [p[0] - F * ((1. / As) + (p[1] * eb / (ms * (eb - F))))]

# TARGET RELIABILITY INDEX
betaT = [3.13217]

# BOUNDS
bndsX = [[200., 400.], [10., 30.], [100., 500.]]

# INITIAL POINTS
x0 = [300., 20., 300.]

# RANDOM DESIGN VARIABLES, x = {x1, x2, x3}
x1 = VA('Lognormal', mean = x0[0], sdev = 3.)
x2 = VA('Lognormal', mean = x0[1], sdev = 2.)
x3 = VA('Lognormal', mean = x0[2], sdev = 5.)
x = [x1, x2, x3]

# RANDOM DESIGN PARAMETERS, p = {p1, p2, p3, p4, p5, p6}
p1 = VA('Lognormal', mean = .4, sdev = .035)
p2 = VA('Normal', mean = 30., sdev = 10.)
p3 = VA('Weibull_MIN', mean = 21., sdev = 4.2)
p4 = VA('Normal', mean = 500., sdev = 50.)
p5 = VA('Gumbel_MIN', mean = 600., sdev = 90.)
p6 = VA('Gumbel_MIN', mean = 600., sdev = 90.)
p = [p1, p2, p3, p4, p5, p6]

# SLA FUNCTION
res = SLA(f, df, g, betaT, x = x, p = p, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)