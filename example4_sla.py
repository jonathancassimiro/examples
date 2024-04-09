from codes import SLA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# GEAR BOX

# OBJECTIVE FUNCTION
def f(d, x):
    return 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_dx0 = 0.7854 * x[1]**2 * (3.3333 * x[2]**2 + 14.9334 * x[2] - 43.0934) - 1.508 * (x[5]**2 + x[6]**2)
    df_dx1 = 2 * 0.7854 * x[0] * x[1] * (3.3333 * x[2]**2 + 14.9334 * x[2] - 43.0934)
    df_dx2 = 2 * 0.7854 * x[0] * x[1]**2 * (3.3333 * x[2] + 14.9334)
    df_dx3 = 0.7854 * x[5]**2
    df_dx4 = 0.7854 * x[6]**2
    df_dx5 = -2 * 1.508 * x[0] * x[5] + 3 * 7.477 * x[5]**2 + 2 * 0.7854 * x[3] * x[5]
    df_dx6 = -2 * 1.508 * x[0] * x[6] + 3 * 7.477 * x[6]**2 + 2 * 0.7854 * x[4] * x[6]
    return np.array([df_dx0, df_dx1, df_dx2, df_dx3, df_dx4, df_dx5, df_dx6])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    g1 = 1. - 27. * (x[0] * x[1]**2 * x[2])**-1
    g2 = 1. - 397.5 * (x[0] * x[1]**2 * x[2]**2)**-1
    g3 = 1. - 1.93 * x[3]**3 * (x[1] * x[2] * x[5]**4)**-1
    g4 = 1. - 1.93 * x[4]**3 * (x[1] * x[2] * x[6]**4)**-1
    g5 = 110. * x[5]**3 - np.sqrt((745*x[3]/(x[1]*x[2]))**2 + 16.9e6)
    g6 = 85.  * x[6]**3 - np.sqrt((745*x[4]/(x[1]*x[2]))**2 + 157.5e6)
    g7 = 40. - x[1] * x[2]
    g8 = x[0] - 5. * x[1]
    g9 = 12. - (x[0] * x[1]**-1)
    g10 = 1. - (1.5 * x[5] + 1.9) * x[3]**-1
    g11 = 1. - (1.1 * x[6] + 1.9) * x[4]**-1
    return g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11

# TARGET RELIABILITY INDEX
betaT = [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]

# BOUNDS
bndsX = [[2.60, 3.60], [0.70, 0.80], [17.0, 28.0], [7.30, 8.30], [7.30, 8.30], [2.90, 3.90], [5.00, 5.50]]             # Valores-limites para as variaveis aleatorias de projeto

# INITIAL POINT
x0 = [3.50, 0.70, 17.0, 7.30, 7.72, 3.35, 5.29]

# RANDOM DESIGN VARIABLES, x = {x1, x2, x3, x4, x5, x6, x7}
x1 = VA('Normal', mean = x0[0], sdev = 0.005)
x2 = VA('Normal', mean = x0[1], sdev = 0.005)
x3 = VA('Normal', mean = x0[2], sdev = 0.005)
x4 = VA('Normal', mean = x0[3], sdev = 0.005)
x5 = VA('Normal', mean = x0[4], sdev = 0.005)
x6 = VA('Normal', mean = x0[5], sdev = 0.005)
x7 = VA('Normal', mean = x0[6], sdev = 0.005)
x = [x1, x2, x3, x4, x5, x6, x7]

# SLA FUNCTION
res = SLA(f, df, g, betaT, x = x, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)