from scipy.optimize import minimize
from montoya_chart import find_alphas
import numpy as np
import pdb, time

execucao = time.time()

# INPUT DATA
fck0 = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
fyk0 = np.array([40, 50])
σall0 = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030])
lb0 = np.array([38, 33, 30, 28, 25, 24])
c0 = np.array([4.5, 4.5, 4.5, 5, 5, 5])
ρmin0 = np.array([0.0015, 0.0015, 0.00164, 0.00179, 0.00194, 0.00208])
cc0 = np.array([99.49, 104.51, 108.53, 118.05, 122.27, 128.16])
ec0 = np.array([224.34, 224.34, 265.28, 265.28, 265.91, 265.95])
cs0 = np.array([1.54, 1.51])
es0 = np.array([3.02, 2.82])
C0 = [0.0000, 0.0513, 0.0597, 0.0665, 0.0722, 0.0773]
phi0 = [1.00, 1.25, 1.60, 2.00, 2.50]

# DATA DESIGN
fck = fck0[0]                                                         # Concrete compressive strength, in kN/cm2
fyk = fyk0[0]                                                         # Yield strength, in kN/cm2
σall = σall0[0]                                                       # Allowable bearing stress, in kN/cm2
N = 300                                                               # Column axial load, in kN
eca, ecb = 0., 0.                                                     # Eccentricity of the axial load in direction a and b
ac, bc = 30 * (1.), 30.                                               # Column length/widht, in cm

index1 = np.searchsorted(fck0, fck)
if index1 >= len(fck0):
    index1 = len(fck0) - 1

index2 = np.searchsorted(fyk0, fyk)
if index2 >= len(fyk0):
    index2 = len(fyk0) - 1

index3 = np.searchsorted(σall0, σall)
if index3 >= len(σall0):
    index3 = len(σall0) - 1

# COSTS OF MATERIALS
cc = cc0[index1]                                                      # Concrete cost, US$/m3
cs = cs0[index2]                                                      # Steel cost B500 & B400, US$/kg
cf = 36.82                                                            # Formwork cost, US$/m2
ce = 11.14                                                            # Excavation cost, US$/m3
cb = 38.10                                                            # Compacted backfill cost, US$/m3
csc = 0.14                                                            # Cement cost, US$/kg

# EMISSIONS OF MATERIALS
ec = ec0[index1]                                                      # Concrete emission, kg/m3
es = es0[index2]                                                      # Steel emission B500 & B400, kg/kg
ef = 14.55                                                            # Formwork emission, kg/m2
ee = 13.16                                                            # Excavation emission, kg/m3
eb = 27.20                                                            # Compacted backfill emission, kg/m3
esc = 0.83                                                            # Cement cost, kg/kg

# CONSTANTS
C = C0[index3]                                                        # Percentage of cement
ρs = 7850.0                                                           # Steel specific density, kg/m3
ρc = 2400.0                                                           # Concrete specific density, kg/m3
ρcem = 1200.0                                                         # Cement specific density, kg/m3
phi_c = 2.0                                                           # Column diameter reinforcement, in cm
phi_f = phi0[0]                                                       # Footing diameter reinforcement, in cm
H = 150                                                               # Excavation depth, in cm
L = 50                                                                # Excavation line, in cm
hl = 5                                                                # Lean concrete layer, in cm
D = 0                                                                # Soil-cement excavation depth, in cm

ρmin = ρmin0[index1]                                                  # Minimum steel rate
lb = lb0[index1]*phi_c                                                # Anchorage lenght of the column, cm
c = c0[index1]                                                        # Concrete cover to reinforcement, in cm

# MONTOYA'S CHARTS
def montoya(x):

    # Self-weight of the footing, in kN
    G = (ρc * (((x[3] - x[2]) * (x[0] * x[1] + ac * bc + np.sqrt(x[0] * x[1] * ac * bc)) / 3 + x[0] * x[1] * x[2] + ac * bc * (H - x[3])) - (x[4] * (x[0] - 4 * c + x[2]) + x[5] * (x[1] - 4 * c + x[2]))) * 10**(-6) + ρs * (x[4] * (x[0] - 4 * c + x[2]) + x[5] * (x[1] - 4 * c + x[2])) * 10**(-6)) * 0.00981

    ηa = eca / x[0]
    ηb = ecb / x[1]

    η1, η2 = sorted([ηa, ηb], reverse=True)
    α1, α2, α3 = find_alphas(η1, η2)

    σ1 = (100 / α1) * ((N + G) / (x[0] * x[1]))
    σ2 = (α2 / 100) * σ1
    σ3 = (α3 / 100) * σ1
    σ4 = σ2 + σ3 - σ1

    return σ1, σ2, σ3, σ4

# OBJECTIVE FUNCTION
def f(x):

    # mass of steel, in kg
    Vs = (x[4] * (x[0] - 4 * c + x[2]) + x[5] * (x[1] - 4 * c + x[2])) * 10**(-6)
    Ms = ρs * Vs

    # volume of concrete, in m3
    Vc = ((x[3] - x[2]) * (x[0] * x[1] + ac * bc + np.sqrt(x[0] * x[1] * ac * bc)) / 3 + x[0] * x[1] * x[2] + ac * bc * (H - x[3])) * 10**(-6) - Vs

    # volume of lean concrete, in m³
    Vl = (hl * (x[0] + 2 * L) * (x[1] + 2 * L) + D * (x[0] + 2 * L) * (x[1] + 2 * L)) * 10**(-6)

    # area of formwork, in m2
    Af = (2 * x[2] *(x[0] + x[1])) * 10**(-4)

    # volume of excavation and volume of soil-cement excavation and backfill, in m3
    Ve = ((H + hl) * (x[0] + 2 * L) * (x[1] + 2 * L) + D * (x[0] + 2 * L) * (x[1] + 2 * L)) * 10**(-6)

    # volume of backfill and volume of soil-cement excavation and backfill, in m3
    Vb = (Ve - Vc - Vl) + (D * (x[0] + 2 * L) * (x[1] + 2 * L)) * 10**(-6)

    # volume of soil-cement excavation and backfill, in m3
    Vsc = (D * (x[0] + 2 * L) * (x[1] + 2 * L)) * 10**(-6)

    # mass of cement, in kg
    Mc = C * ρcem * Vsc

    # objective functions
    ECO2 = es * Ms + ec * Vc + ec * Vl + ef * Af + ee * (Ve + Vsc) + eb * (Vb + Vsc) + esc * Mc
    cost = cs * Ms + cc * Vc + cc * Vl + cf * Af + ce * (Ve + Vsc) + cb * (Vb + Vsc) + csc * Mc

    print(ECO2, cost)

    return ECO2

# LIMIT STATES FUNCTIONS
def g1(x): # soil-bearing capacity
    σ1, σ2, σ3, σ4 = montoya(x)
    return 1.25 * σall - σ1
def g2(x): # tipping
    return 1/9 - (eca/x[0])**2 - (ecb/x[1])**2
def g3(x): # simple bending
    d = x[3] - c - phi_f / 2
    σ1, σ2, σ3, σ4 = montoya(x)
    λ = 0.8
    σcd = 0.85 * fck/1.
    xa = (x[4] * fyk/1.) / (λ * x[0] * σcd)
    MRa = λ * x[0] * xa * (d - 0.5 * λ * xa) * σcd
    σa = (σ1 + σ2) / 2
    MSa = (σa / 2) * x[1] * ((x[0] / 2 - 0.35 * ac) ** 2)
    return MRa - MSa
def g4(x): # simple bending
    d = x[3] - c - phi_f / 2
    σ1, σ2, σ3, σ4 = montoya(x)
    λ = 0.8
    σcd = 0.85 * fck / 1.
    xb = (x[5] * fyk / 1.) / (λ * x[1] * σcd)
    MRb = λ * x[1] * xb * (d - 0.5 * λ * xb) * σcd
    σb = (σ1 + σ3) / 2
    MSb = (σb / 2) * x[0] * ((x[1] / 2 - 0.35 * bc) ** 2)
    return MRb - MSb
def g5(x): # punching shear
    d = x[3] - c - phi_f/2
    r_table = [0.50, 1.00, 2.00, 3.00]
    K_table = [0.45, 0.60, 0.70, 0.80]
    r1 = ac / bc
    r2 = bc / ac
    idx1 = min(range(len(r_table)), key=lambda i: abs(r1 - r_table[i]))
    idx2 = min(range(len(r_table)), key=lambda i: abs(r2 - r_table[i]))
    Ka = K_table[idx1]
    Kb = K_table[idx2]
    Wa = (ac ** 2) / 2 + ac * bc + 4 * bc * d + 16 * d ** 2 + 2 * np.pi * d * ac
    Wb = (bc ** 2) / 2 + ac * bc + 4 * ac * d + 16 * d ** 2 + 2 * np.pi * d * bc
    MSa = N * eca
    MSb = N * ecb
    τRd2 = 0.27 * (1 - fck0[0]*10 / 250) * fck/1.
    τs = (1. * N) / (2 * (ac + bc) * d) + ((Ka*MSa)/(Wa*d)) + ((Kb*MSb)/(Wb*d))
    return τRd2 - τs
def g6(x): # shear force
    d = x[3] - c - phi_f / 2
    σ1, σ2, σ3, σ4 = montoya(x)
    σ13a = σ3 + (σ1 - σ3) * ((x[0] + ac + d) / (2 * x[0]))
    σ24a = σ4 + (σ2 - σ4) * ((x[0] + ac + d) / (2 * x[0]))
    σa = (σ1 + σ2 + σ13a + σ24a) / 4
    Aa = 0.25 * (x[1] + bc + d) * (x[0] + ac + d)
    QR2a = 0.27 * (1 - fck*10 / 250) * fck/1. * x[1] * d
    QSa = 1.5 * σa * Aa
    return QR2a - QSa
def g7(x): # shear force
    d = x[3] - c - phi_f / 2
    σ1, σ2, σ3, σ4 = montoya(x)
    σ12b = σ2 + (σ1 - σ2) * ((x[1] + bc + d) / (2 * x[1]))
    σ34b = σ4 + (σ3 - σ4) * ((x[1] + bc + d) / (2 * x[1]))
    σb = (σ1 + σ3 + σ12b + σ34b) / 4
    Ab = 0.25 * (x[0] + ac + d) * (x[1] + bc + d)
    QR2b = 0.27 * (1 - fck*10 / 250) * fck/1. * x[0] * d
    QSb = 1.5 * σb * Ab
    return QR2b - QSb
def g8(x): # reinforcement limitations
    Asa_min = ρmin * x[1] * x[2]
    return x[4] - Asa_min
def g9(x): # reinforcement limitations
    Asb_min = ρmin * x[0] * x[2]
    return x[5] - Asb_min
def g10(x): # geometric limitations
    return x[3] - (x[0] - ac) / 3
def g11(x): # geometric limitations
    return x[3] - (x[1] - bc) / 3
def g12(x): # geometric limitations
    return x[3] - lb - c
def g13(x): # geometric limitations
    n = (4 * x[4]) / (np.pi * phi_f ** 2) - 1
    sa = ((x[0] - 2 * c) / n) - phi_f
    return 20  - sa
def g14(x): # geometric limitations
    n = (4 * x[5]) / (np.pi * phi_f ** 2) - 1
    sb = ((x[1] - 2 * c) / n) - phi_f
    return 20  - sb
def g15(x):
    return x[2] - 0.4 * x[3]

cons = ({'type': 'ineq', 'fun': lambda x: g1(x)},
        {'type': 'ineq', 'fun': lambda x: g2(x)},
        {'type': 'ineq', 'fun': lambda x: g3(x)},
        {'type': 'ineq', 'fun': lambda x: g4(x)},
        {'type': 'ineq', 'fun': lambda x: g5(x)},
        {'type': 'ineq', 'fun': lambda x: g6(x)},
        {'type': 'ineq', 'fun': lambda x: g7(x)},
        {'type': 'ineq', 'fun': lambda x: g8(x)},
        {'type': 'ineq', 'fun': lambda x: g9(x)},
        {'type': 'ineq', 'fun': lambda x: g10(x)},
        {'type': 'ineq', 'fun': lambda x: g11(x)},
        {'type': 'ineq', 'fun': lambda x: g12(x)},
        {'type': 'ineq', 'fun': lambda x: g13(x)},
        {'type': 'ineq', 'fun': lambda x: g14(x)},
        {'type': 'ineq', 'fun': lambda x: g15(x)})

# BOUNDS
bnds = [[60, np.inf], [60, np.inf], [15, np.inf], [37.5, np.inf], [1.35, np.inf], [1.35, np.inf]]

# INITIAL POINTS
x0 = [60, 60, 15, 37.5, 1.35, 1.35]

# DETERMINISTIC OPTIMIZATION
res = minimize(f, x0 = x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': False, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')