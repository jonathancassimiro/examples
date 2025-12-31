from codes import SLA, VA
from montoya_chart import find_alphas
import numpy as np
import pdb, time

execucao = time.time()
Geval = 0

# DATA DESIGN
fck = 2.5                                                            # Concrete compressive strength, in kN/cm2
siga = 0.005                                                         # Allowable bearing stress, in kN/cm2
fyk = 40                                                             # Yield strength, in kN/cm2
N = 300.                                                             # Column axial load, in kN
eca, ecb = 0., 0.                                                    # Eccentricity of the axial load in direction a and b
ac, bc = 30 * (1.), 30.                                              # Column length/widht, in cm

# COSTS OF MATERIALS
cc = 99.49                                                           # Concrete cost, US$/m3
cs = 1.54                                                            # Steel cost, US$/kg
cf = 36.82                                                           # Formwork cost, US$/m2
ce = 11.14                                                           # Excavation cost, US$/m3
cb = 38.10                                                           # Compacted backfill cost, US$/m3
csc = 0.14                                                           # Cement cost, US$/kg

# EMISSIONS OF MATERIALS
ec = 224.34                                                          # Concrete emission, kg/m3
es = 3.02                                                            # Steel emission, kg/kg
ef = 14.55                                                           # Formwork emission, kg/m2
ee = 13.16                                                           # Excavation emission, kg/m3
eb = 27.20                                                           # Compacted backfill emission, kg/m3
esc = 0.83                                                           # Cement cost, kg/kg

# INPUT DATA
C = 0.                                                               # Percentage of cement
ρs = 7850.                                                           # Steel specific density, kg/m3
ρc = 2400.0                                                          # Concrete specific density, kg/m3
ρcem = 1200.0                                                        # Cement specific density, kg/m3
phi_c = 2.0                                                          # Column diameter reinforcement, in cm
phi_f = 1.0                                                          # Footing diameter reinforcement, in cm
H = 150.                                                             # Excavation depth, in cm
L = 50.                                                              # Excavation line, in cm
hl = 5                                                               # Lean concrete layer, in cm
D = 0                                                                # Soil-cement excavation depth, in cm

ρmin = 0.0015                                                        # Minimum steel rate
lb = 38*phi_c                                                        # Anchorage lenght of the column, cm
c = 4.5                                                              # Concrete cover to reinforcement, in cm

def montoya(x, p):

    # Self-weight of the footing, in kN
    G = (ρc * (((x[3] - x[2]) * (x[0] * x[1] + ac * bc + np.sqrt(x[0] * x[1] * ac * bc)) / 3 + x[0] * x[1] * x[2] + ac * bc * (H - x[3]) - (x[4] * (x[0] - 4 * c + x[2]) + x[5] * (x[1] - 4 * c + x[2]))) * 1e-6) + ρs * (x[4] * (x[0] - 4 * c + x[2]) + x[5] * (x[1] - 4 * c + x[2])) * 1e-6) * 0.00981
    ηa = eca / x[0]
    ηb = ecb / x[1]

    η1 = np.maximum(ηa, ηb)
    η2 = np.minimum(ηa, ηb)
    α1, α2, α3 = find_alphas(η1, η2)

    σ1 = (100 / α1) * ((p[3] + G) / (x[0] * x[1]))
    σ2 = (α2 / 100) * σ1
    σ3 = (α3 / 100) * σ1
    σ4 = σ2 + σ3 - σ1

    return σ1, σ2, σ3, σ4

# OBJECTIVE FUNCTION
def f(d, x):

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

    return cost

# LIMIT STATES FUNCTIONS
def g(d, x, p):

    global Geval
    Geval += 1

    d = x[3] - c - phi_f / 2
    σ1, σ2, σ3, σ4 = montoya(x, p)
    σ12b = σ2 + (σ1 - σ2) * ((x[1] + bc + d) / (2 * x[1]))
    σ34b = σ4 + (σ3 - σ4) * ((x[1] + bc + d) / (2 * x[1]))
    σ13a = σ3 + (σ1 - σ3) * ((x[0] + ac + d) / (2 * x[0]))
    σ24a = σ4 + (σ2 - σ4) * ((x[0] + ac + d) / (2 * x[0]))

    r_ = [0.50, 1.00, 2.00, 3.00]
    K_ = [0.45, 0.60, 0.70, 0.80]
    ia = min(range(len(r_)), key=lambda i: abs((ac / bc) - r_[i]))
    ib = min(range(len(r_)), key=lambda i: abs((bc / ac) - r_[i]))
    Wa = (ac ** 2) / 2 + ac * bc + 4 * bc * d + 16 * d ** 2 + 2 * np.pi * d * ac
    Wb = (bc ** 2) / 2 + ac * bc + 4 * ac * d + 16 * d ** 2 + 2 * np.pi * d * bc
    xa = (x[4] * p[1] / 1.) / (0.8 * x[0] * (0.85 * p[0] / 1.))
    xb = (x[5] * p[1] / 1.) / (0.8 * x[1] * (0.85 * p[0] / 1.))

    g1  = 1.25 * p[2] - σ1
    g2  = 1/9 - (eca / x[0])**2 - (ecb / x[1])**2
    g3 = (0.8 * x[0] * xa * (d - 0.5 * 0.8 * xa) * (0.85 * p[0] / 1.) - (((σ1 + σ2) / 2) / 2) * x[1] * ((x[0]/2 - 0.35*ac)**2))
    g4 = (0.8 * x[1] * xb * (d - 0.5 * 0.8 * xb) * (0.85 * p[0] / 1.) - (((σ1 + σ3) / 2) / 2) * x[0] * ((x[1]/2 - 0.35*bc)**2))
    g5 = (0.27 * (1 - p[0]*10/250) * p[0] / 1. - (p[3] / (2 * (ac + bc) * d)) + ((K_[ia]) * (p[3] * eca)) / (Wa * d) + ((K_[ib]) * (p[3] * ecb)) / (Wb * d))
    g6 = (0.27 * (1 - p[0]*10/250) * p[0] / 1. * x[1] * d - 1.5 * ((σ1 + σ2 + σ13a + σ24a) / 4) * (0.25 * (x[1] + bc + d) * (x[0] + ac + d)))
    g7 = (0.27 * (1 - p[0]*10/250) * p[0] / 1. * x[0] * d - 1.5 * ((σ1 + σ3 + σ12b + σ34b) / 4) * (0.25 * (x[0] + ac + d) * (x[1] + bc + d)))
    g8  = x[4] - ρmin * x[1] * x[2]
    g9  = x[5] - ρmin * x[0] * x[2]
    g10 = x[3] - (x[0] - ac) / 3
    g11 = x[3] - (x[1] - bc) / 3
    g12 = x[3] - lb - c
    g13 = 20 - ((x[0] - 2*c) / ((4*x[4])/(np.pi*phi_f**2) - 1)) - phi_f
    g14 = 20 - ((x[1] - 2*c) / ((4*x[5])/(np.pi*phi_f**2) - 1)) - phi_f
    g15 = x[2] - 0.4 * x[3]

    return g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15

# TARGET RELIABILITY INDEX
betaT = [2.0] * 15

# BOUNDS
bndsX = [[60, np.inf], [60, np.inf], [15, np.inf], [37.5, np.inf], [1.35, np.inf], [1.35, np.inf]]

# INITIAL POINTS
x0 = [60, 60, 15, 37.5, 1.35, 1.35]

# RANDOM DESIGN VARIABLES, x = {x1, x2, x3, x4, x5, x6}
x1  = VA('Normal', mean = x0[0], cv = 0.025)     # End footing cross-section length
x2  = VA('Normal', mean = x0[1], cv = 0.025)     # End footing cross-section width
x3  = VA('Normal', mean = x0[2], cv = 0.025)     # End footing base height
x4  = VA('Normal', mean = x0[3], cv = 0.025)     # End footing total height
x5  = VA('Normal', mean = x0[4], cv = 0.025)     # End footing reinforcement area, cross-section length
x6  = VA('Normal', mean = x0[5], cv = 0.025)     # End footing reinforcement area, cross-section width
x = [x1, x2, x3, x4, x5, x6]

# RANDOM DESIGN PARAMETERS, p = {p1, p2, p3}
p1 = VA('Lognormal', mean = fck, cv = 0.100)    # Design compressive strength of the concrete
p2 = VA('Lognormal', mean = fyk, cv = 0.100)    # Design strength of the steel
p3 = VA('Lognormal', mean = siga, cv = 0.125)   # Design allowable bearing stress of the soil
p4 = VA('Normal', mean = N, cv = 0.100)         # End column axial load
p = [p1, p2, p3, p4]

# SLA FUNCTION
res = SLA(f, g, betaT, x = x, p = p, bndsX = bndsX, disp = True)

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)