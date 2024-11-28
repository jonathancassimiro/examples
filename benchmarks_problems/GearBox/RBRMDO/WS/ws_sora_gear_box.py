from scipy.optimize import minimize, NonlinearConstraint, OptimizeResult
from distributions import VA, phi, Phi, InvPhi
import numpy as np
import time

execucao = time.time()
Geval = 0

# SEQUENTIAL OPTIMIZATION AND RELIABILITY ASSESSMENT (SORA)
def SORA(f, g, betaT, d = [], x = [], p = [], R = None, bndsd = None, bndsX = None, bnds_cons = None, cons = None, MPPmaxiter=1000, MPPtol=1e-4, OPTmaxiter=1000, OPTtol=1e-4):

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
            grad[i] = (g(d, x + dx, p)[ls] - g(d, x - dx, p)[ls]) / (2 * step)
        for i in range(nPAP):
            dp = np.zeros(nPAP)
            dp[i] = step
            grad[nVAP + i] = (g(d, x, p + dp)[ls] - g(d, x, p - dp)[ls]) / (2 * step)
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

    # Initialization
    xMPP = np.empty((nLS, nVAP))
    pMPP = np.empty((nLS, nPAP))
    LSval = np.empty(nLS)
    Jyz, Jzy = JacobZY(R)
    d = np.array(d)
    mean_x = np.array([xi.mean for xi in x])
    mean_p = np.array([pi.mean for pi in p])
    VP = np.concatenate((d, mean_x))
    for i in range(nLS):
        xMPP[i, :] = mean_x
        pMPP[i, :] = mean_p

    # Deterministic inequality constraints
    def cons_aux(var):
        constr = np.empty(nLS)
        d = var[:nVDt]
        mean_x = var[nVDt:]
        for i in range(nLS):
            constr[i] = g(d, mean_x - s[i, :], pMPP[i, :])[i]
        return constr

    # Deterministic equality constraints
    def cons_aux_eq(var):
        constr = np.empty(ncons)
        d = var[:nVDt]
        mean_x = var[nVDt:]
        for i in range(ncons):
            constr[i] = cons(d, mean_x)[i]
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

    # Calculate MPP
    def MPP(ls, betaT, d, x0, p0, Xi):
        k = 0
        xMPP = x0
        pMPP = p0

        while (k <= MPPmaxiter):
            g_old = g(d, xMPP, pMPP)[ls]
            Jxz, Jzx, mean_eq = JacobXZ(Xi + p, np.concatenate((xMPP, pMPP)))
            Jyx = np.matmul(Jyz, Jzx)
            Jxy = np.matmul(Jxz, Jzy)
            gradx = grad_g(d, xMPP, pMPP, ls)
            grady = np.matmul(np.transpose(Jxy), gradx)
            alfa = grady / np.linalg.norm(grady)

            k += 1
            xMPP = mean_eq[:nVAP] - np.matmul(Jxy[:nVAP, :], alfa * betaT)
            pMPP = mean_eq[nVAP:] - np.matmul(Jxy[nVAP:, :], alfa * betaT)
            g_new = g(d, xMPP, pMPP)[ls]

            if (abs((g_new - g_old) / g_old) <= MPPtol):
                break

        return [xMPP, pMPP]

    # Deterministic Optimization
    def optimization(VP):
        res = minimize(faux, VP, method='slsqp', bounds=bnds, constraints=constr, options={'disp':True, 'maxiter': OPTmaxiter, 'ftol': OPTtol})
        d = res.x[:nVDt]
        mean_x = res.x[nVDt:]
        VP = np.concatenate((d, mean_x))
        volume.append(f1(d, mean_x))
        stress.append(f2(d, mean_x))
        return mean_x, volume, stress, res

    # Update random variables
    def update(mean_x):
        for i in range(nVAP):
            x[i].mean = mean_x[i]
            x[i].calcpar()

    # Convergence history
    def history(w1, w2, volume, stress, res):
        print(res)
        print('w1:', w1)
        print('w2:', w2)
        print('f1:', volume)
        print('f2:', stress)
        print('Processing Time:', round((time.time() - execucao), 3), 's')
        print('Limit State Functions Evaluations:', Geval)
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    # Shift vector
    s = mean_x - xMPP

    volume = []
    stress = []

    # Pareto points
    nPar = 15
    W1 = np.linspace(1., 0., num=nPar)
    W2 = np.linspace(0., 1., num=nPar)

    for w1, w2 in zip(W1, W2):
        if w1 + w2 == 1:
            mean_x, volume, stress, res = optimization(VP)
            update(mean_x)
            history(w1, w2, volume, stress, res)

    return volume, stress

# OBJECTIVE FUNCTION 1
def f1(d, x):
    np.random.seed(seed=999)
    N = 1000
    x1 = VA('Normal', mean=x[0], sdev=0.005, size = N).random()
    x2 = VA('Normal', mean=x[1], sdev=0.005, size = N).random()
    x3 = VA('Normal', mean=x[2], sdev=0.005, size = N).random()
    x4 = VA('Normal', mean=x[3], sdev=0.005, size = N).random()
    x5 = VA('Normal', mean=x[4], sdev=0.005, size = N).random()
    x6 = VA('Normal', mean=x[5], sdev=0.005, size = N).random()
    x7 = VA('Normal', mean=x[6], sdev=0.005, size = N).random()
    V1 = []
    for i in range(N):
        V = 0.7854*x1[i]*x2[i]**2*(3.3333*x3[i]**2 + 14.9334*x3[i] - 43.0934) - 1.508*x1[i]*(x6[i]** 2 + x7[i]**2) + 7.477*(x6[i]**3 + x7[i]**3) + 0.7854*(x4[i]*x6[i]**2 + x5[i]*x7[i]**2)
        V1.append(V)
    Vmed = np.mean(V1, axis=0)
    Vmin = np.min(np.abs(Vmed))
    return Vmin

# OBJECTIVE FUNCTION 2
def f2(d, x):
    np.random.seed(seed=999123)
    N = 1000
    x1 = VA('Normal', mean=x[0], sdev=0.005, size = N).random()
    x2 = VA('Normal', mean=x[1], sdev=0.005, size = N).random()
    x3 = VA('Normal', mean=x[2], sdev=0.005, size = N).random()
    x4 = VA('Normal', mean=x[3], sdev=0.005, size = N).random()
    x5 = VA('Normal', mean=x[4], sdev=0.005, size = N).random()
    x6 = VA('Normal', mean=x[5], sdev=0.005, size = N).random()
    x7 = VA('Normal', mean=x[6], sdev=0.005, size = N).random()
    S1 = []
    for i in range(N):
        S = 1100. - np.sqrt((745*x4[i]/(x2[i]*x3[i]))**2 + 16.9 * 10e6) * (0.10*x6[i]**3)**-1
        S1.append(S)
    Smed = np.mean(S1, axis=0)
    Smax = np.max(np.abs(Smed))
    return Smax

# ROBUST OBJECTIVE FUNCTION
def f(d, x):
    Vmin = f1(d, x)
    Smax = f2(d, x)
    return w1 * Vmin / Vmin0 + w2 * Smax / Smax0

# LIMIT STATES FUNCTIONS
def g(d, x, p):
    global Geval
    Geval += 1
    g1 = 1. - 27. * (x[0] * x[1]**2 * x[2])**-1
    g2 = 1. - 397.5 * (x[0] * x[1]**2 * x[2]**2)**-1
    g3 = 1. - 1.93 * x[3]**3 * (x[1] * x[2] * x[5]**4)**-1
    g4 = 1. - 1.93 * x[4]**3 * (x[1] * x[2] * x[6]**4)**-1
    g5 = 1100. - np.sqrt((745*x[3]/(x[1]*x[2]))**2 + 16.9e6) * (0.10 * x[5]**3)**-1
    g6 = 850.  - np.sqrt((745*x[4]/(x[1]*x[2]))**2 + 157.5e6) * (0.10 * x[6]**3)**-1
    g7 = 40. - x[1] * x[2]
    g8 = x[0] * x[1]**-1 - 5.
    g9 = 12. - (x[0] * x[1]**-1)
    g10 = 1. - (1.5 * x[5] + 1.9) * x[3]**-1
    g11 = 1. - (1.1 * x[6] + 1.9) * x[4]**-1
    return g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11

# TARGET RELIABILITY INDEX
betaT = [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]

# BOUNDS
bndsX = [[2.60, 3.60], [0.70, 0.80], [17.0, 28.0], [7.30, 8.30], [7.30, 8.30], [2.90, 3.90], [5.00, 5.50]]

# INITIAL POINT
d = []  # NONE
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

# NORMALIZATION
Vmin0 = f1(d, x0)
Smax0 = f2(d, x0)

# SORA FUNCTION
volume, stress = SORA(f, g, betaT, x = x, bndsX = bndsX)

# RESULTS
print('Mean volume:', volume)
print('Mean stress:', stress)
print('Processing Time:', round((time.time() - execucao),3), 's')
print('Limit State Functions Evaluations:', Geval)