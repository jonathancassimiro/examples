from codes import SORA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# TENSION/COMPRESSION SPRING

# OBJECTIVE FUNCTION
def f(d, x):
    return (x[2] + 2.) * x[1] * x[0]**2

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_x1 = 2. * (x[2] + 2.) * x[1] * x[0]
    df_x2 = (x[2] + 2.) * x[0]**2
    df_x3 = x[1] * x[0]**2
    return np.array([df_x1, df_x2, df_x3])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    g1 = x[1]**3 * x[2] / (71785. * x[0]**4) - 1.
    g2 = 1. - (4. * x[1]**2 - x[0] * x[1]) / (12566. * (x[1] * x[0]**3 - x[0]**4)) + 1. / (5108. * x[0]**2)
    g3 = 140.45 * x[0] / (x[1]**2 * x[2]) - 1.
    g4 = 1. - (x[0] + x[1]) / 1.5
    return g1, g2, g3, g4

# TARGET RELIABILITY INDEX
betaT = [3., 3., 3., 3.]

# BOUNDS
bndsX = [[0.01, 0.10], [0.10, 1.00], [5.00, 15.0]]

# INITIAL POINTS
x0 = [0.05, 0.50, 10.0]

# RANDOM DESIGN VARIABLES, x = {x1, x2, x3}
x1 = VA('Normal', mean = x0[0], sdev = 0.001**2)
x2 = VA('Normal', mean = x0[1], sdev = 0.010**2)
x3 = VA('Normal', mean = x0[2], sdev = 0.800**2)
x = [x1, x2, x3]

# SORA FUNCTION
res = SORA(f, df, g, betaT, x = x, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)