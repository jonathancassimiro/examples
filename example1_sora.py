from codes import SORA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# NONLINEAR LIMIT STATE

# OBJECTIVE FUNCTION
def f(d, x):
    return d[0] ** 2 + d[1] ** 2

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_d1 = 2. * d[0]
    df_d2 = 2. * d[1]
    return np.array([df_d1, df_d2])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    return [(d[0] * d[1] * p[1]**2)/5. - p[0]]

# TARGET RELIABILITY INDEX
betaT = [2.32]

# BOUNDS
bndsd = [[0, 15.], [0, 15.]]

# INITIAL POINTS
d0 = [[2., 2.], [12., 12.]] # Modify according to the problem

# RANDOM DESIGN PARAMETERS, p = {p1, p2}
p1 = VA('Normal', mean = 5., sdev = 1.5)
p2 = VA('Normal', mean = 3., sdev = 0.9)
p = [p1, p2]

# SORA FUNCTION
res = SORA(f, df, g, betaT, d = d0[0], p = p, bndsd = bndsd, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)