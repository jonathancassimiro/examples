from scipy.optimize import minimize, NonlinearConstraint, OptimizeResult
from distributions import VA, phi, Phi, InvPhi
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

# SINGLE LOOP APPROACH (SLA)
def SLA(f, g, betaT, d = [], x = [], p = [], R = None, bndsd = None, bndsX = None, bnds_cons = None, cons = None, OPTmaxiter=1000, OPTtol=1e-4):

    nVDt = len(d)
    nVAP = len(x)
    nPAP = len(p)
    nVP = nVDt + nVAP
    nVA = nVAP + nPAP
    nLS = len(betaT)

    global w1
    global w2

    if nVP == 0:
        raise ValueError('At least one design variable must be specified!')

    if bnds_cons != None:
        ncons = np.array(bnds_cons).shape[0]
    else:
        ncons = 0

    if R == None:
        R = np.eye(nVA)

    # Objective Function
    def faux(var):
        d = var[:nVDt]
        x = var[nVDt:]
        return f(d, x)

    # Gradient LSF
    def grad_g(d, x, p, ls, step=1e-6):
        grad = np.empty(nVA)
        for i in range(nVAP):
            dx = np.zeros(nVAP)
            dx[i] = step
            grad[i] = (g(d, x + dx, p)[ls].item() - g(d, x - dx, p)[ls].item()) / (2 * step)
        for i in range(nPAP):
            dp = np.zeros(nPAP)
            dp[i] = step
            grad[nVAP + i] = (g(d, x, p + dp)[ls].item() - g(d, x, p - dp)[ls].item()) / (2 * step)
        return grad

    # Bound
    if bndsd == None:
        bndsd = [[None, None], ] * nVDt
    if bndsX == None:
        bndsX = [[None, None], ] * nVAP
    bnds = bndsd + bndsX

    # Jacobian transformation Z --> Y
    def JacobZY(cx):
        L = np.linalg.cholesky(cx)
        Jyz = np.linalg.pinv(L)
        Jzy = L
        return [Jyz, Jzy]

    # Jacobian transformation X --> Z
    def JacobXZ(RV, rvi):
        mean_eq = np.empty(nVA)
        sdev_eq = np.empty(nVA)
        for i in range(nVA):
            zi = InvPhi(RV[i].cdf(rvi[i]))
            if RV[i].pdf(rvi[i]) == 0. or np.isnan(zi):
                sdev_eq[i] = 0.
                mean_eq[i] = rvi[i]
            else:
                sdev_eq[i] = phi(zi) / RV[i].pdf(rvi[i])
                mean_eq[i] = rvi[i] - zi * sdev_eq[i]
        D_eq = np.diag(sdev_eq)
        Jxz = D_eq
        Jzx = np.linalg.pinv(D_eq)
        return [Jxz, Jzx, mean_eq]

    for i in range(nVAP):
        x[i].calcpar()
    for i in range(nPAP):
        p[i].calcpar()

    # Inicialization
    xMPP = np.empty((nLS, nVAP))
    pMPP = np.empty((nLS, nPAP))
    alfa = np.empty((nLS,  nVA))
    Jyz, Jzy = JacobZY(R)
    d = np.array(d)
    mean_x = np.array([xi.mean for xi in x])
    mean_p = np.array([pi.mean for pi in p])
    VP = np.concatenate((d, mean_x))
    for i in range(nLS):
        xMPP[i,:] = mean_x
        pMPP[i,:] = mean_p

    # Deterministic inequality constraints
    def cons_aux(var):
        constr = np.empty(nLS)
        d = var[:nVDt]
        x = var[nVDt:]
        for i in range(nLS):
            Xi = x - np.matmul(Jxy[:nVAP, :], alfa[i, :]) * betaT[i]
            constr[i] = g(d, Xi, pMPP[i, :])[i].item()
        return constr

    # Deterministic equality constraints
    def cons_aux_eq(var):
        constr = np.empty(ncons)
        d = var[:nVDt]
        x = var[nVDt:]
        for i in range(ncons):
            constr[i] = cons(d, x)[i]
        return constr

    # Bounds
    if bnds_cons != None:
        bdsc = np.array(bnds_cons, dtype=object)
        bdsc[:, 0][bdsc[:, 0] == None] = -np.inf
        bdsc[:, 1][bdsc[:, 1] == None] = np.inf
        inf = np.full(nLS, 0)
        sup = np.full(nLS, np.inf)
        inf_eq = bdsc[:, 0]
        sup_eq = bdsc[:, 1]
        constr = [NonlinearConstraint(cons_aux, inf, sup), NonlinearConstraint(cons_aux_eq, inf_eq, sup_eq)]
    else:
        inf = np.full(nLS, 0)
        sup = np.full(nLS, np.inf)
        constr = NonlinearConstraint(cons_aux, inf, sup)

    # MPP approximate
    for i in range(nLS):
        Jxz, Jzx, mean_eq = JacobXZ(x + p, np.concatenate((mean_x, mean_p)))
        Jyx = np.matmul(Jyz, Jzx)
        Jxy = np.matmul(Jxz, Jzy)
        gradx = grad_g(d, xMPP[i,:], pMPP[i,:], ls=i)
        grady = np.matmul(np.transpose(Jxy), gradx)
        alfa[i, :] = grady / np.linalg.norm(grady) if np.linalg.norm(grady) >= 1e-12 else 0.0
        xMPP[i,:] = mean_x - np.matmul(Jxy[:nVAP, :], alfa[i,:]) * betaT[i]
        pMPP[i,:] = mean_p - np.matmul(Jxy[nVAP:, :], alfa[i,:]) * betaT[i]

    # Deterministic optimization
    def optimization(VP):
        res = minimize(faux, VP, method='slsqp', bounds=bnds, constraints=constr, options={'disp':True, 'maxiter': OPTmaxiter, 'ftol': OPTtol})
        d = res.x[:nVDt]
        mean_x = res.x[nVDt:]
        ECO2.append(f1(d, mean_x))
        cost.append(f2(d, mean_x))
        return mean_x, ECO2, cost, res

    # Update random variables
    def update(mean_x):
        for i in range(nVAP):
            x[i].mean = mean_x[i]
            x[i].calcpar()

    # Convergence history
    def history(w1, w2, ECO2, cost, res):
        print(res)
        print('w1:', w1)
        print('w2:', w2)
        print('f1:', ECO2)
        print('f2:', cost)
        print('Processing Time:', round((time.time() - execucao), 3), 's')
        print('Limit State Functions Evaluations:', Geval)
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    ECO2 = []
    cost = []

    # Pareto points
    nPar = 15
    W1 = np.linspace(1., 0., num=nPar)
    W2 = np.linspace(0., 1., num=nPar)

    for w1, w2 in zip(W1, W2):
        if w1 + w2 == 1:
            mean_x, ECO2, cost, res = optimization(VP)
            update(mean_x)
            history(w1, w2, ECO2, cost, res)

    return ECO2, cost

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

# OBJECTIVE FUNCTION 1
def f1(d, x):
    np.random.seed(seed=999)
    N = 1000
    x1 = VA('Normal', mean=x[0], sdev=0.025, size = N).random()
    x2 = VA('Normal', mean=x[1], sdev=0.025, size = N).random()
    x3 = VA('Normal', mean=x[2], sdev=0.025, size = N).random()
    x4 = VA('Normal', mean=x[3], sdev=0.025, size = N).random()
    x5 = VA('Normal', mean=x[4], sdev=0.025, size = N).random()
    x6 = VA('Normal', mean=x[5], sdev=0.025, size = N).random()
    E1 = []
    for i in range(N):
        ECO2 = es * ρs * (x5[i] * (x1[i] - 4 * c + x3[i]) + x6[i] * (x2[i] - 4 * c + x3[i])) * 10 ** (-6) +\
               ec * (((x4[i] - x3[i]) * (x1[i] * x2[i] + ac * bc + np.sqrt(x1[i] * x2[i] * ac * bc)) / 3 + x1[i] * x2[i] * x3[i] + ac * bc * (H - x4[i])) * 10 ** (-6) - ((x5[i] * (x1[i] - 4 * c + x3[i]) + x6[i] * (x2[i] - 4 * c + x3[i])) * 10 ** (-6))) +\
               ec * (hl * (x1[i] + 2 * L) * (x2[i] + 2 * L) + D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6) +\
               ef * (2 * x3[i] * (x1[i] + x2[i])) * 10 ** (-4) +\
               ee * ((H + hl) * (x1[i] + 2 * L) * (x2[i] + 2 * L) + D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6) +\
               eb * (((((H + hl) * (x1[i] + 2 * L) * (x2[i] + 2 * L) + D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6)) - (((x4[i] - x3[i]) * (x1[i] * x2[i] + ac * bc + np.sqrt(x1[i] * x2[i] * ac * bc)) / 3 + x1[i] * x2[i] * x3[i] + ac * bc * (H - x4[i])) * 10 ** (-6) - ((x5[i] * (x1[i] - 4 * c + x3[i]) + x6[i] * (x2[i] - 4 * c + x3[i])) * 10 ** (-6))) - ((hl * (x1[i] + 2 * L) * (x2[i] + 2 * L) + D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6))) + (D * (x1[i] + 2 * L) * (x2[i] + 2 * L))) * 10 ** (-6) +\
               esc * C * ρcem * (D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6)
        E1.append(ECO2)
    Emed = np.mean(E1, axis=0)
    Emin = np.min(np.abs(Emed))
    return Emin

# OBJECTIVE FUNCTION 2
def f2(d, x):
    np.random.seed(seed=999)
    N = 1000
    x1 = VA('Normal', mean=x[0], sdev=0.025, size = N).random()
    x2 = VA('Normal', mean=x[1], sdev=0.025, size = N).random()
    x3 = VA('Normal', mean=x[2], sdev=0.025, size = N).random()
    x4 = VA('Normal', mean=x[3], sdev=0.025, size = N).random()
    x5 = VA('Normal', mean=x[4], sdev=0.025, size = N).random()
    x6 = VA('Normal', mean=x[5], sdev=0.025, size = N).random()
    C1 = []
    for i in range(N):
        cost = cs * ρs * (x5[i] * (x1[i] - 4 * c + x3[i]) + x6[i] * (x2[i] - 4 * c + x3[i])) * 10 ** (-6) +\
               cc * (((x4[i] - x3[i]) * (x1[i] * x2[i] + ac * bc + np.sqrt(x1[i] * x2[i] * ac * bc)) / 3 + x1[i] * x2[i] * x3[i] + ac * bc * (H - x4[i])) * 10 ** (-6) - ((x5[i] * (x1[i] - 4 * c + x3[i]) + x6[i] * (x2[i] - 4 * c + x3[i])) * 10 ** (-6))) +\
               cc * (hl * (x1[i] + 2 * L) * (x2[i] + 2 * L) + D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6) +\
               cf * (2 * x3[i] * (x1[i] + x2[i])) * 10 ** (-4) +\
               ce * ((H + hl) * (x1[i] + 2 * L) * (x2[i] + 2 * L) + D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6) +\
               cb * (((((H + hl) * (x1[i] + 2 * L) * (x2[i] + 2 * L) + D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6)) - (((x4[i] - x3[i]) * (x1[i] * x2[i] + ac * bc + np.sqrt(x1[i] * x2[i] * ac * bc)) / 3 + x1[i] * x2[i] * x3[i] + ac * bc * (H - x4[i])) * 10 ** (-6) - ((x5[i] * (x1[i] - 4 * c + x3[i]) + x6[i] * (x2[i] - 4 * c + x3[i])) * 10 ** (-6))) - ((hl * (x1[i] + 2 * L) * (x2[i] + 2 * L) + D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6))) + (D * (x1[i] + 2 * L) * (x2[i] + 2 * L))) * 10 ** (-6) +\
               csc * C * ρcem * (D * (x1[i] + 2 * L) * (x2[i] + 2 * L)) * 10 ** (-6)
        C1.append(cost)
    Cmed = np.mean(C1, axis=0)
    Cmin = np.min(np.abs(Cmed))
    return Cmin

# ROBUST OBJECTIVE FUNCTION
def f(d, x):
    Emin = f1(d, x)
    Cmin = f2(d, x)
    return w1 * Emin / Emin0 + w2 * Cmin / Cmin0

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
betaT = [3.5] * 15

# BOUNDS
bndsX = [[60, np.inf], [60, np.inf], [15, np.inf], [37.5, np.inf], [1.35, np.inf], [1.35, np.inf]]

# INITIAL POINT
d = []  # NONE
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

# NORMALIZATION
Emin0 = f1(d, x0)
Cmin0 = f2(d, x0)

# SLA FUNCTION
ECO2, cost = SLA(f, g, betaT, x = x, p = p, bndsX = bndsX)

# RESULTS
print('Mean emissions:', ECO2)
print('Mean costs:', cost)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)