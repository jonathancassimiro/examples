from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Lognormal
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

# INPUT DATA
ρs = 7850.                                                           # Steel specific density, kg/m3
ρmin = 0.0015                                                        # Minimum steel rate
phi = 2.                                                             # Diameter reinforcement, in cm
H = 150.                                                             # Excavation depth, in cm
L = 50.                                                              # Excavation line, in cm
lb = 38*phi                                                          # Anchorage lenght of the column, cm
c = 3.                                                               # Concrete cover to reinforcement, in cm

# resistences
fck = 2.5                                                            # Concrete compressive strength, in kN/cm2
siga = 0.005                                                         # Allowable bearing stress, in kN/cm2

# loads and column geometry
N = 300.                                                             # Column axial load, in kN
eca = 0.                                                             # Eccentricity of the axial load in direction a
ecb = 0.                                                             # Eccentricity of the axial load in direction b
lc = 30.                                                             # Column length, in cm
wc = 30.                                                             # Column width, in cm

# costs of materials
cc = 99.49                                                           # Concrete cost, US$/m3
cs = 1.54                                                            # Steel cost, US$/kg
cf = 36.82                                                           # Formwork cost, US$/m2
ce = 11.14                                                           # Excavation cost, US$/m3
cb = 38.10                                                           # Compacted backfill cost, US$/m3

# emissions of materials
ec = 224.34                                                          # Concrete emission, kg/m3
es = 3.02                                                            # Steel emission, kg/kg
ef = 14.55                                                           # Formwork emission, kg/m2
ee = 13.16                                                           # Excavation emission, kg/m3
eb = 27.20                                                           # Compacted backfill emission, kg/m3

def lsf(x, g):
    PMA = True
    options = AnalysisOptions()
    options.is_PMA = PMA
    options.beta_targ = betaT
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('x1', x[0], 0.025*x[0]))
    stochastic_model.addVariable(Normal('x2', x[1], 0.025*x[1]))
    stochastic_model.addVariable(Normal('x3', x[2], 0.025*x[2]))
    stochastic_model.addVariable(Normal('x4', x[3], 0.025*x[3]))
    stochastic_model.addVariable(Normal('x5', x[4], 0.025*x[4]))
    stochastic_model.addVariable(Normal('x6', x[5], 0.025*x[5]))
    stochastic_model.addVariable(Lognormal('p1', fck, 0.100*fck))
    stochastic_model.addVariable(Lognormal('p2', siga, 0.125*siga))
    stochastic_model.addVariable(Normal('p3', N, 0.100*N))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    return Analysis.G

# OBJECTIVE FUNCTION
def f(x):

    # volume of concrete, in m3
    Vc = ((x[3] - x[2]) * (x[0] * x[1] + lc * wc + np.sqrt(x[0] * x[1] * lc * wc)) / 3 + x[0] * x[1] * x[2] + lc * wc * (H - x[3])) * 10**(-6)

    # mass of steel, in kg
    Ms = ρs * (x[4] * (x[0] - 2 * c + 15 * phi) + x[5] * (x[1] - 2 * c + 15 * phi)) * 10**(-6)

    # area of formwork, in m2
    Af = (2 * x[2] *(x[0] + x[1])) * 10**(-4)

    # volume of excavation
    Ve = (H * (x[0] + 2 * L) * (x[1] + 2 * L)) * 10**(-6)

    # volume of backfill
    Vb = Ve - Vc

    print(Vc, Ms, Af, Ve, Vb)

    # objective functions
    ECO2 = ec * Vc + es * Ms + ef * Af + ee * Ve + eb * Vb
    cost = cc * Vc + cs * Ms + cf * Af + ce * Ve + cb * Vb

    return cost

# LIMIT STATES FUNCTIONS
def g1(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return p2 - ((1.1 * p3) / (x1 * x2)) * (1 + 6 * eca/x1 + 6 * ecb/x2)
def g2(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return 1/9 - (eca/x1)**2 - (ecb/x2)**2
def g3(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return 0.27 * (1 - p1*10 / 250) * p1/1.4 - (1.4 * p3) / (2 * (lc + wc) * (x4 - c - phi/2))
def g4(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return x5 - ρmin * (x2 * x3 + (x2 + wc)/2 * (x4 + x3))
def g5(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return x6 - ρmin * (x1 * x3 + (x1 + lc)/2 * (x4 + x3))
def g6(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return x4 - (x1 - lc) / 3
def g7(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return x4 - (x2 - wc) / 3
def g8(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return x4 - lb - c
def g9(x1, x2, x3, x4, x5, x6, p1, p2, p3):
    global Geval
    Geval += 1
    return x3 - 0.4 * x4

cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1)},
        #{'type': 'ineq', 'fun': lambda x: lsf(x, g2)},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g3)},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g4)},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g5)},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g6)},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g7)},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g8)},
        {'type': 'ineq', 'fun': lambda x: lsf(x, g9)})

# TARGET RELIABILITY INDEX
betaT = [4]

# BOUNDS
bnds = [[60, np.inf], [60, np.inf], [15, np.inf], [37.5, np.inf], [1.35, np.inf], [1.35, np.inf]]

# INITIAL POINTS
x0 = [60, 60, 15, 37.5, 1.35, 1.35]

# OPTIMIZATION
res = minimize(fun=f, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': True, 'maxiter':1000, 'ftol': 1e-4})

# RESULTS
print(res)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)