from codes import SORA, VA
import numpy as np
import time

execucao = time.time()
Geval = 0

# HOCK AND SCHITTKOWSKI 113

# OBJECTIVE FUNCTION
def f(d, x):
    return x[0]**2 + x[1]**2 + x[0]*x[1] - 14. * x[0] - 16. * x[1] + (x[2] - 10.)**2 + 4. * (x[3] - 5.)**2 + (x[4] - 3.)**2 + 2. * (x[5] - 1.)**2 + 5. * x[6]**2 + 7. * (x[7] - 11.)**2 + 2. * (x[8] - 10.)**2 + (x[9] - 7.)**2 + 45.

# JACOBIAN OF THE OBJECTIVE FUNCTION
def df(d, x):
    df_dx1 = 2. * x[0] + x[1] - 14.
    df_dx2 = 2. * x[1] + x[0] - 16.
    df_dx3 = 2. * x[2] - 20.
    df_dx4 = 8. * x[3] - 40.
    df_dx5 = 2. * x[4] - 6.
    df_dx6 = 4. * x[5] - 4.
    df_dx7 = 10. * x[6]
    df_dx8 = 14. * x[7] - 154.
    df_dx9 = 4. * x[8] - 40.
    df_dx10 = 2. * x[9] - 14.
    return np.array([[df_dx1, df_dx2, df_dx3, df_dx4, df_dx5, df_dx6, df_dx7, df_dx8, df_dx9, df_dx10]])

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    g1 = 1. - (4. * x[0] + 5. * x[1] - 3. * x[6] + 9. * x[7]) / 105.
    g2 = -10. * x[0] + 8. * x[1] + 17. * x[6] - 2 * x[7]
    g3 = 1. + (8 * x[0] - 2. * x[1] - 5. * x[8] + 2. * x[9]) / 12.
    g4 = 1. - (3. * (x[0] - 2.) ** 2 + 4. * (x[1] - 3.) ** 2 + 2. * x[2] ** 2 - 7 * x[3]) / 120.
    g5 = 1. - (5. * x[0] ** 2 + 8. * x[1] + (x[2] - 6.) ** 2 - 2. * x[3]) / 40.
    g6 = 1. - (0.5 * (x[0] - 8.) ** 2 + 2. * (x[1] - 4.) ** 2 + 3 * x[4] ** 2 - x[5]) / 30.
    g7 = -x[0] ** 2 - 2. * (x[1] - 2.) ** 2 + 2. * x[0] * x[1] - 14. * x[4] + 6 * x[5]
    g8 = 3. * x[0] - 6. * x[1] - 12. * (x[8] - 8.) ** 2 + 7. * x[9]
    return g1, g2, g3, g4, g5, g6, g7, g8

# TARGET RELIABILITY INDEX
betaT = [3., 3., 3., 3., 3., 3., 3., 3.]

# BOUNDS
bndsX = [[0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf]]

# INITIAL POINT
x0 = [2.17, 2.36, 8.77, 5.10, 0.99, 1.43, 1.32, 9.83, 8.28, 8.38]

# RANDOM DESIGN VARIABLES, x = {x1, x2, x3, x4, x5, x6, x7, x8, x9, x10}
x1 = VA('Normal', mean = x0[0], sdev = 0.02**1)
x2 = VA('Normal', mean = x0[1], sdev = 0.02**1)
x3 = VA('Normal', mean = x0[2], sdev = 0.02**1)
x4 = VA('Normal', mean = x0[3], sdev = 0.02**1)
x5 = VA('Normal', mean = x0[4], sdev = 0.02**1)
x6 = VA('Normal', mean = x0[5], sdev = 0.02**1)
x7 = VA('Normal', mean = x0[6], sdev = 0.02**1)
x8 = VA('Normal', mean = x0[7], sdev = 0.02**1)
x9 = VA('Normal', mean = x0[8], sdev = 0.02**1)
x10 = VA('Normal', mean = x0[9], sdev = 0.02**1)
x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

# SORA FUNCTION
res = SORA(f, df, g, betaT, x = x, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)