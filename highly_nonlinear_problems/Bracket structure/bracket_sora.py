from codes import SORA, VA
import numpy as np
import time

start_time = time.time()
Geval = 0

# AUTOMOBILE WITH FRONT I-BEAM AXLE

# OBJECTIVE FUNCTION
def f(d, x):
    return 7.86 * x[2] * 5. * ((4. * np.sqrt(3) / 9.) * x[0] + x[1])

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_x1 = 52.4 * x[2] * np.sqrt(3) / 3.
    df_x2 = 39.3 * x[2]
    df_x3 = 52.4 * x[0] * np.sqrt(3) / 3. + 39.3 * x[1]
    return np.array([df_x1, df_x2, df_x3])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    Mb = p[0] * p[4] / 3. + p[3] * 9.81 * x[1] * x[2] * p[4]**2 / 18.
    Fb = (9. * np.pi**2 * p[1] * x[2] * x[0]**3 * np.sin(np.deg2rad(60))**2) / (48. * p[4]**2)
    Fab = (1 / np.cos(np.deg2rad(60))) * ((3. * p[0] / 2.) + (3. * p[3] * 9.81 * x[1] * x[2] * p[4] / 4.))
    g1 = p[2] - 6. * Mb / (x[1] * x[2]**2)
    g2 = Fb - Fab
    return g1, g2

# TARGET RELIABILITY INDEX
betaT = [2., 2.]

# BOUNDS
bndsX = [[0.050, 0.300], [0.050, 0.300], [0.050, 0.300]]

# INITIAL POINTS
x0 = [0.061, 0.202, 0.269]

# RANDOM DESIGN VARIABLES, x = {x1, x2, x3}
x1 = VA('Normal', mean = x0[0], cv = 0.05)
x2 = VA('Normal', mean = x0[1], cv = 0.05)
x3 = VA('Normal', mean = x0[2], cv = 0.05)
x = [x1, x2, x3]

# RANDOM DESIGN PARAMETERS, p = {p1, p2, p3, p4, p5}
p1 = VA('Gumbel_MIN', mean = 100, cv = 0.15)
p2 = VA('Gumbel_MIN', mean = 200e6, cv = 0.08)
p3 = VA('Lognormal', mean = 225e3, cv = 0.08)
p4 = VA('Weibull_MIN', mean = 7.86, cv = 0.10)
p5 = VA('Normal', mean = 5, cv = 0.05)
p = [p1, p2, p3, p4, p5]

# SORA FUNCTION
res = SORA(f, df, g, betaT, x = x, p = p, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - start_time),3), 's')
print('Limit State Functions Evaluations:', Geval)