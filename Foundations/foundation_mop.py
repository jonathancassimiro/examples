from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Lognormal
from scipy.optimize import minimize
import numpy as np
import time

execucao = time.time()
Geval = 0

CO2 = []
cost = []

# INPUT DATA
fck0 = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
lb0 = np.array([38, 33, 30, 28, 25, 24])
c0 = np.array([3, 4, 4, 5, 5, 5])
ρmin0 = np.array([0.0015, 0.0015, 0.00164, 0.00179, 0.00194, 0.00208])
cc0 = np.array([99.49, 104.51, 108.53, 118.05, 122.27, 128.16])
ec0 = np.array([224.34, 224.34, 265.28, 265.28, 265.91, 265.95])
ρs = 7850.                                                           # Steel specific density, kg/m3
phi = 2.                                                             # Diameter reinforcement, in cm
H = 150.                                                             # Excavation depth, in cm
L = 50.                                                              # Excavation line, in cm

# resistences
fck = fck0[0]                                                        # Concrete compressive strength, in kN/cm2
siga = 0.005                                                         # Allowable bearing stress, in kN/cm2

index = np.searchsorted(fck0, fck)
if index >= len(fck0):
    index = len(fck0) - 1

# loads and column geometry
N = 300.                                                             # Column axial load, in kN
eca = 0.                                                             # Eccentricity of the axial load in direction a
ecb = 0.                                                             # Eccentricity of the axial load in direction b
lc = 30.                                                             # Column length, in cm
wc = 30.                                                             # Column width, in cm

# costs of materials
cc = cc0[index]                                                      # Concrete cost, US$/m3
cs = 1.54                                                            # Steel cost, US$/kg
cf = 36.82                                                           # Formwork cost, US$/m2
ce = 11.14                                                           # Excavation cost, US$/m3
cb = 38.10                                                           # Compacted backfill cost, US$/m3

# emissions of materials
ec = ec0[index]                                                      # Concrete emission, kg/m3
es = 3.02                                                            # Steel emission, kg/kg
ef = 14.55                                                           # Formwork emission, kg/m2
ee = 13.16                                                           # Excavation emission, kg/m3
eb = 27.20                                                           # Compacted backfill emission, kg/m3

ρmin = ρmin0[index]                                                  # Minimum steel rate
lb = lb0[index]*phi                                                  # Anchorage lenght of the column, cm
c = c0[index]                                                        # Concrete cover to reinforcement, in cm

def lsf(x, g):
    PMA = True
    options = AnalysisOptions()
    options.is_PMA = PMA
    options.beta_targ = betaT[0] # Modify according to the problem
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

# OBJECTIVE FUNCTION 1
def f1(x):

    np.random.seed(seed=999)
    N = 1000
    x1 = np.random.normal(loc=x[0], scale=0.025, size=N)
    x2 = np.random.normal(loc=x[1], scale=0.025, size=N)
    x3 = np.random.normal(loc=x[2], scale=0.025, size=N)
    x4 = np.random.normal(loc=x[3], scale=0.025, size=N)
    x5 = np.random.normal(loc=x[4], scale=0.025, size=N)
    x6 = np.random.normal(loc=x[5], scale=0.025, size=N)
    V1 = []
    for i in range(N):

        # Volume of concrete, in m³
        Vc = ((x4[i] - x3[i]) * (x1[i] * x2[i] + lc * wc + np.sqrt(x1[i] * x2[i] * lc * wc)) / 3 + x1[i] * x2[i] * x3[i] + lc * wc * (H - x4[i])) * 10 ** (-6)

        # Mass of steel, in kg
        Ms = ρs * (x5[i] * (x1[i] - 2 * c + 15 * phi) + x6[i] * (x2[i] - 2 * c + 15 * phi)) * 10 ** (-6)

        # Area of formwork, in m²
        Af = (2 * x3[i] * (x1[i] + x2[i])) * 10 ** (-4)

        # Volume of excavation, in m³
        Ve = (H * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6)

        # Volume of backfill, in m³
        Vb = Ve - Vc

        ECO2 = ec * Vc + es * Ms + ef * Af + ee * Ve + eb * Vb

        V1.append(ECO2)
    Vmed = np.mean(V1, axis=0)
    Vmin = np.min(np.abs(Vmed))

    return Vmin

# OBJECTIVE FUNCTION 2
def f2(x):

    np.random.seed(seed=999123)
    N = 1000
    x1 = np.random.normal(loc=x[0], scale=0.025, size=N)
    x2 = np.random.normal(loc=x[1], scale=0.025, size=N)
    x3 = np.random.normal(loc=x[2], scale=0.025, size=N)
    x4 = np.random.normal(loc=x[3], scale=0.025, size=N)
    x5 = np.random.normal(loc=x[4], scale=0.025, size=N)
    x6 = np.random.normal(loc=x[5], scale=0.025, size=N)
    V1 = []
    for i in range(N):

        # Volume of concrete, in m³
        Vc = ((x4[i] - x3[i]) * (x1[i] * x2[i] + lc * wc + np.sqrt(x1[i] * x2[i] * lc * wc)) / 3 + x1[i] * x2[i] * x3[i] + lc * wc * (H - x4[i])) * 10 ** (-6)

        # Mass of steel, in kg
        Ms = ρs * (x5[i] * (x1[i] - 2 * c + 15 * phi) + x6[i] * (x2[i] - 2 * c + 15 * phi)) * 10 ** (-6)

        # Area of formwork, in m²
        Af = (2 * x3[i] * (x1[i] + x2[i])) * 10 ** (-4)

        # Volume of excavation, in m³
        Ve = (H * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6)

        # Volume of backfill, in m³
        Vb = Ve - Vc

        cost = cc * Vc + cs * Ms + cf * Af + ce * Ve + cb * Vb

        V1.append(cost)
    Vmed = np.mean(V1, axis=0)
    Vmin = np.min(np.abs(Vmed))

    return Vmin

# ROBUST OBJECTIVE FUNCTION
def f(x):
    CO2 = f1(x)
    cost = f2(x)
    return w1 * CO2 / CO2_0 + w2 * cost / cost_0

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
betaT = [2., 3., 4.] # Modify according to the problem

# BOUNDS
bnds = [[60, np.inf], [60, np.inf], [15, np.inf], [37.5, np.inf], [1.35, np.inf], [1.35, np.inf]]

# INITIAL POINTS
x0 = [60, 60, 15, 37.5, 1.35, 1.35]

# NORMALIZATION
CO2_0 = f1(x0)
cost_0 = f2(x0)

# DETERMINISTIC OPTIMIZATION
def optimization(x0):
    res = minimize(f, x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp':True, 'maxiter': 1000, 'ftol': 1e-4})
    mean_x = res.x
    CO2.append(f1(mean_x))
    cost.append(f2(mean_x))
    return CO2, cost, res

# CONVERGENCE HISTORY
def history(w1, w2, CO2, cost, res):
    print(res)
    print('w1:', w1)
    print('w2:', w2)
    print('f1:', CO2)
    print('f2:', cost)
    print('Processing Time:', round((time.time() - execucao), 3), 's')
    print('Limit State Functions Evaluations:', Geval)
    print('-----------------------------------------------')

# PARETO POINTS
nPar = 1
W1 = np.linspace(1., 0., num=nPar)
W2 = np.linspace(0., 1., num=nPar)

for w1, w2 in zip(W1, W2):
    if w1 + w2 == 1:
        CO2, cost, res = optimization(x0)
        history(w1, w2, CO2, cost, res)

# RESULTS
print('CO2 Emissions:', CO2)
print('Cost:', cost)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)