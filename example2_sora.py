from codes import SORA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# MULTIPLE LIMIT STATE

# OBJECTIVE FUNCTION
def f(d, x):
    return x[0] + x[1]

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_dx1 = 1.
    df_dx2 = 1.
    return np.array([[df_dx1, df_dx2]])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    g1 = x[0] ** 2 * x[1] / 20. - 1.
    g2 = (x[0] + x[1] - 5.) ** 2 / 30. + (x[0] - x[1] - 12.) ** 2 / 120. - 1.
    g3 = 80.  / (x[0] ** 2 + 8. * x[1] + 5.) - 1.
    return g1, g2, g3

# TARGET RELIABILITY INDEX
betaT = [[2., 2., 2.], [3., 3., 3.], [4., 4., 4.]] # Modify according to the problem

# BOUNDS
bndsX = [[0, 10.], [0, 10.]]

# RANDOM DESIGN VARIABLES, x = {x1, x2}
x1 = VA('Normal', mean = 5., sdev = 0.3)
x2 = VA('Normal', mean = 5., sdev = 0.3)
x = [x1, x2]

# SORA FUNCTION
res = SORA(f, df, g, betaT[2], x = x, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)