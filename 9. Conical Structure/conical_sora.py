from codes import SORA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# CONICAL STRUCTURE

# OBJECTIVE FUNCTION
def f(d, x):
    return np.pi * x[0] * (2 * x[1] + np.sin(x[2]))

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_x1 = np.pi * (2 * x[1] + np.sin(x[2]))
    df_x2 = 2. * np.pi * x[0]
    df_x3 = np.pi * x[0] * np.cos(x[2])
    return np.array([df_x1, df_x2, df_x3])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    return [1. - 1.6523 / (np.pi * p[2] * x[0]**2 * np.cos(x[2])**2) * ((p[0] / 0.66) + (p[1] / (0.41 * x[1])))]

# TARGET RELIABILITY INDEX
betaT = [3.5]

# BOUNDS
bndsX = [[0.001, 0.005], [0.800, 1.000], [np.pi/6, np.pi/4]]

# INITIAL POINTS
x0 = [0.0025, 0.900, 0.524]

# RANDOM DESIGN VARIABLES, x = {x1, x2, x3}
x1 = VA('Lognormal', mean = x0[0], cv = 0.050)
x2 = VA('Lognormal', mean = x0[1], cv = 0.025)
x3 = VA('Lognormal', mean = x0[2], cv = 0.020)
x = [x1, x2, x3]

# RANDOM DESIGN PARAMETERS, p = {p1, p2, p3}
p1 = VA('Normal', mean = 7e4, cv = 0.080)
p2 = VA('Normal', mean = 8e4, cv = 0.080)
p3 = VA('Normal', mean = 7e10, cv = 0.050)
p = [p1, p2, p3]

# SORA FUNCTION
res = SORA(f, df, g, betaT, x = x, p = p, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)