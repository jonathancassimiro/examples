from codes import SORA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# AUTOMOBILE WITH FRONT I-BEAM AXLE

# OBJECTIVE FUNCTION
def f(d, x):
    return x[3] * x[0] + 2. * (x[1] - x[0]) * x[2]

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_x1 = x[3] - 2. * x[2]
    df_x2 = 2. * x[2]
    df_x3 = 2. * x[1] - 2. * x[0]
    df_x4 = x[0]
    return np.array([df_x1, df_x2, df_x3, df_x4])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    Wx = x[0] * (x[3] - 2. * x[2]) ** 3 / (6. * x[3]) + x[1] * (x[3] ** 3 - (x[3] - 2. * x[2]) ** 3) / (6. * x[3])
    Wp = 0.8 * x[1] * x[2] ** 2 + 0.4 * x[0] **3 * (x[3] - 2. * x[2]) / x[2]
    gamas = .460
    gamam = p[0] / Wx
    tau = p[1] / Wp
    return [gamas - np.sqrt((gamam ** 2) + 3. * (tau ** 2))]

# TARGET RELIABILITY INDEX
betaT = [3.]

# BOUNDS
bndsX = [[10., 20.], [60., 100.], [10., 20.], [70., 120.]]

# INITIAL POINTS
x0 = [12., 75., 12., 85.]

# RANDOM DESIGN VARIABLES, x = {x1, x2, x3, x4}
x1 = VA('Normal', mean = x0[0], sdev = 0.060)
x2 = VA('Normal', mean = x0[1], sdev = 0.325)
x3 = VA('Normal', mean = x0[2], sdev = 0.070)
x4 = VA('Normal', mean = x0[3], sdev = 0.425)
x = [x1, x2, x3, x4]

# RANDOM DESIGN PARAMETERS, p = {p1, p2}
p1 = VA('Normal', mean = 3.5e3, sdev = 1.75e2)
p2 = VA('Normal', mean = 3.1e3, sdev = 1.55e2)
p = [p1, p2]

# SORA FUNCTION
res = SORA(f, df, g, betaT, x = x, p = p, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)