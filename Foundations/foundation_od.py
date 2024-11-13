from scipy.optimize import minimize

import numpy as np
import time

execucao = time.time()

# INPUT DATA
fck0 = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
siga0 = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030])
lb0 = np.array([38, 33, 30, 28, 25, 24])
c0 = np.array([3, 4, 4, 5, 5, 5])
ρmin0 = np.array([0.0015, 0.0015, 0.00164, 0.00179, 0.00194, 0.00208])
cc0 = np.array([99.49, 104.51, 108.53, 118.05, 122.27, 128.16])
ec0 = np.array([224.34, 224.34, 265.28, 265.28, 265.91, 265.95])
C = [0.0000, 0.0513, 0.0597, 0.0665, 0.0722, 0.0773]                  # Percentage of cement
ρs = 7850.0                                                           # Steel specific density, kg/m3
ρc = 1200.0                                                           # Cement specific density, kg/m3
phi = 2                                                               # Diameter reinforcement, in cm
H = 150                                                               # Excavation depth, in cm
L = 50                                                                # Excavation line, in cm
D = 0                                                                 # Soil-cement excavation depth, in cm

# resistences
fck = fck0[0] / (1 - 1.645 * 0.10)                                    # Concrete compressive strength, in kN/cm2
siga = siga0[0]                                                       # Allowable bearing stress, in kN/cm2

# loads and column geometry
N = 300                                                               # Column axial load, in kN
eca = 0.                                                              # Eccentricity of the axial load in direction a
ecb = 0.                                                              # Eccentricity of the axial load in direction b
lc = 30 * (1.)                                                       # Column length, in cm
wc = 30                                                               # Column width, in cm

index = np.searchsorted(fck0, fck)
if index >= len(fck0):
    index = len(fck0) - 1

# costs of materials
cc = cc0[index]                                                       # Concrete cost, US$/m3
cs = 1.54                                                             # Steel cost, US$/kg
cf = 36.82                                                            # Formwork cost, US$/m2
ce = 11.14                                                            # Excavation cost, US$/m3
cb = 38.10                                                            # Compacted backfill cost, US$/m3
csc = 0.14                                                            # Cement cost, US$/kg

# emissions of materials
ec = ec0[index]                                                       # Concrete emission, kg/m3
es = 3.02                                                             # Steel emission, kg/kg
ef = 14.55                                                            # Formwork emission, kg/m2
ee = 13.16                                                            # Excavation emission, kg/m3
eb = 27.20                                                            # Compacted backfill emission, kg/m3
esc = 0.83                                                            # Cement cost, kg/kg

ρmin = ρmin0[index]                                                   # Minimum steel rate
lb = lb0[index]*phi                                                   # Anchorage lenght of the column, cm
c = c0[index]                                                         # Concrete cover to reinforcement, in cm

# OBJECTIVE FUNCTION
def f(x):

    # volume of concrete, in m3
    Vc = ((x[3] - x[2]) * (x[0] * x[1] + lc * wc + np.sqrt(x[0] * x[1] * lc * wc)) / 3 + x[0] * x[1] * x[2] + lc * wc * (H - x[3])) * 10**(-6)

    # mass of steel, in kg
    Ms = ρs * (x[4] * (x[0] - 2 * c + 15 * phi) + x[5] * (x[1] - 2 * c + 15 * phi)) * 10**(-6)

    # area of formwork, in m2
    Af = (2 * x[2] *(x[0] + x[1])) * 10**(-4)

    # volume of excavation and volume of soil-cement excavation and backfill, in m3
    Ve = (H * (x[0] + 2 * L) * (x[1] + 2 * L) + D * (x[0] + 2 * L) * (x[1] + 2 * L)) * 10**(-6)

    # volume of backfill and volume of soil-cement excavation and backfill, in m3
    Vb = (Ve - Vc) + (D * (x[0] + 2 * L) * (x[1] + 2 * L)) * 10**(-6)

    # volume of soil-cement excavation and backfill, in m3
    Vsc = (D * (x[0] + 2 * L) * (x[1] + 2 * L)) * 10**(-6)

    # mass of cement, in kg
    Mc = C[0] * ρc * Vsc

    # objective functions
    ECO2 = ec * Vc + es * Ms + ef * Af + ee * Ve + eb * Vb + esc * Mc
    cost = cc * Vc + cs * Ms + cf * Af + ce * Ve + cb * Vb + csc * Mc
    return ECO2

# LIMIT STATES FUNCTIONS
cons = ({'type': 'ineq', 'fun': lambda x: siga - ((1.1 * N) / (x[0] * x[1])) * (1 + 6 * eca/x[0] + 6 * ecb/x[1])},
        {'type': 'ineq', 'fun': lambda x: 1/9 - (eca/x[0])**2 - (ecb/x[1])**2},
        {'type': 'ineq', 'fun': lambda x: 0.27 * (1 - fck0[0]*10 / 250) * fck/1.4 - (1.4 * N) / (2 * (lc + wc) * (x[3] - c - phi/2))},
        {'type': 'ineq', 'fun': lambda x: x[4] - ρmin * (x[1] * x[2] + (x[1] + wc)/2 * (x[3] + x[2]))},
        {'type': 'ineq', 'fun': lambda x: x[5] - ρmin * (x[0] * x[2] + (x[0] + lc)/2 * (x[3] + x[2]))},
        {'type': 'ineq', 'fun': lambda x: x[3] - (x[0] - lc) / 3},
        {'type': 'ineq', 'fun': lambda x: x[3] - (x[1] - wc) / 3},
        {'type': 'ineq', 'fun': lambda x: x[3] - lb - c},
        {'type': 'ineq', 'fun': lambda x: x[2] - 0.4 * x[3]})

# BOUNDS
bnds = [[60, np.inf], [60, np.inf], [15, np.inf], [37.5, np.inf], [1.35, np.inf], [1.35, np.inf]]

# INITIAL POINTS
x0 = [60, 60, 15, 37.5, 1.35, 1.35]

res = minimize(f, x0 = x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': False, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
