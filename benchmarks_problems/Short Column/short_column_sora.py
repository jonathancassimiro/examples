from codes import SORA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# SHORT COLUMN

# OBJECTIVE FUNCTION
def f(d, x):
    return x[0] * x[1]

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_dx1 = x[1]
    df_dx2 = x[0]
    return np.array([[df_dx1, df_dx2]])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    return [1. - 4. * p[2] / (x[0] * x[1] ** 2 * p[0]) - 4. * p[3] /(x[0] ** 2 * x[1] * p[0]) - p[1] ** 2 / ((x[0] * x[1] * p[0]) ** 2)]

# DETERMINISTIC CONSTRAINTS
def cons(d, x):
    return [x[0] / x[1]]

# BOUNDS
bnds_cons = [[0.5, 2.0]]

# TARGET RELIABILITY INDEX
betaT = [3.]

# COEFFICIENTS OF VARIATION
CoV = [0.05, 0.10, 0.15] # Modify according to the problem

# INITIAL POINT
d0 = [0.5, 0.5]
x0 = [0.5, 0.5]

# RANDOM DESIGN VARIABLES, x = {x1, x2}
x1 = VA('Normal', mean = x0[0], cv = CoV[1])
x2 = VA('Normal', mean = x0[1], cv = CoV[1])
x = [x1, x2]

# RANDOM DESIGN PARAMETERS, p = {p1, p2, p3, p4}
p1 = VA('Normal', mean =  4e4, cv = 0.1)
p2 = VA('Normal', mean = 2500, cv = 0.2)
p3 = VA('Normal', mean =  250, cv = 0.3)
p4 = VA('Normal', mean =  125, cv = 0.3)
p = [p1, p2, p3, p4]

# SORA FUNCTION
res = SORA(f, df, g, betaT, x = x, p = p, cons = cons, bnds_cons = bnds_cons, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)