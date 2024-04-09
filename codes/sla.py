from scipy.optimize import minimize, NonlinearConstraint, OptimizeResult, root
from scipy.special import gamma, gammainc, erf, erfinv
import numpy as np

def phi(x):
    return 1. / np.sqrt(2. * np.pi) * np.exp(-x ** 2 / 2)

def Phi(x):
    return 1. / 2. * (1. + erf(x / np.sqrt(2.)))

def InvPhi(Fx):
    return np.sqrt(2.) * erfinv(2 * Fx - 1.)

# Classe de variavies aleatorias
class VA:

    def __init__(self, dist, **kwargs):
        if dist not in ['Uniform', 'Normal', 'Lognormal', 'Exponential', 'Rayleigh', 'Logistic', 'Gamma', 'Gumbel_MIN', 'Gumbel_MAX']:
            raise ValueError('Distribuicao desconhecida: {}'.format(dist))
        self.dist  = dist
        self.mean  = kwargs.get('mean')
        self.sdev  = kwargs.get('sdev')
        self.cv    = kwargs.get('cv')
        self.par1  = kwargs.get('par1')
        self.par2  = kwargs.get('par2')
        self.par3  = kwargs.get('par3')
        self.vmin  = kwargs.get('vmin')
        self.vmax  = kwargs.get('vmax')
        self.size  = kwargs.get('size')
        self.low   = kwargs.get('low')
        self.high  = kwargs.get('high')
        self.shape = kwargs.get('shape')

    def __float__(self):
        # Retorne um valor do tipo float que representa o objeto
        return float(self.mean)

    def calcpar(self):
        if (self.cv != None):
            self.sdev = self.mean * self.cv
        if (self.dist == 'Uniform'):
            self.par1 = self.mean - sqrt(3.) * self.sdev
            self.par2 = self.mean + sqrt(3.) * self.sdev
        elif (self.dist == 'Normal'):
            self.par1 = self.mean
            self.par2 = self.sdev
        elif (self.dist == 'Lognormal'):
            if self.mean == 0.:
                self.par2 = float('inf')  # Definir como infinito quando mean=0
                self.par1 = -0.5 * self.par2 ** 2
            else:
                self.par2 = sqrt(log(1. + (self.sdev / self.mean) ** 2))
                self.par1 = log(self.mean) - 0.5 * self.par2 ** 2
        elif (self.dist == 'Exponential'):
            self.par1 = 1./self.sdev
            self.par3 = self.mean - self.sdev
        elif (self.dist == 'Rayleigh'):
            self.par1 = self.sdev/sqrt(2. - pi/2.)
            self.par3 = self.mean - self.par1 * sqrt(pi / 2.)
        elif (self.dist == 'Logistic'):
            self.par1 = self.mean
            self.par2 = self.sdev * sqrt(3.) / pi
        elif (self.dist == 'Gamma'):
            self.par1 = (self.mean / self.sdev) ** 2
            self.par2 = self.mean / self.sdev ** 2
        elif (self.dist == 'Gumbel_MIN'):
            self.par2 = pi / (sqrt(6.) * self.sdev)
            self.par1 = self.mean + 0.577216 / self.par2
        elif (self.dist == 'Gumbel_MAX'):
            self.par2 = pi / (sqrt(6.) * self.sdev)
            self.par1 = self.mean - 0.577216 / self.par2

    def pdf(self, x):
        if (self.dist == 'Uniform'):
            if ((x < self.par1) or (x > self.par2)):
                return 0.
            else:
                return 1./(self.par2*self.par1)
        elif (self.dist == 'Normal'):
            if self.par2 == 0:
                return 0.
            else:
                return phi((x - self.par1)/self.par2) / self.par2
        elif (self.dist == 'Lognormal'):
            if self.mean == 0.:
                if x == 0.:
                    return 0.
                else:
                    return float('inf')  # Retornar infinito quando mean=0 e x != 0
            else:
                return phi((log(x) - self.par1) / self.par2) / (x * self.par2)
        elif (self.dist == 'Exponential'):
            if (x < self.par3):
                return 0.
            else:
                return self.par1 * exp(-self.par1 * (x - self.par3))
        elif (self.dist == 'Rayleigh'):
            if (x < self.par3):
                return 0.
            else:
                return (x - self.par3) / self.par1 ** 2 * exp(-1./2. * ((x - self.par3) / self.par1) ** 2)
        elif (self.dist == 'Logistic'):
            return exp(-(x - self.par1) / self.par2) / (self.par2 * (1. + exp(-(x - self.par1) / self.par2)) ** 2)
        elif (self.dist == 'Gamma'):
            if (x < 0.):
                return 0.
            else:
                return self.par2 * (self.par2 * x) ** (self.par1 - 1) * exp(-self.par2 * x) / gamma(self.par1)
        elif (self.dist == 'Gumbel_MIN'):
            return self.par2 * exp(self.par2 * (x - self.par1) - exp( self.par2 * (x - self.par1)))
        elif (self.dist == 'Gumbel_MAX'):
            return self.par2 * exp(self.par2 * (x - self.par1) - exp(-self.par2 * (x - self.par1)))

    def cdf(self, x):
        if (self.dist == 'Uniform'):
            if (x < self.par1):
                return 0.
            elif (x > self.par2):
                return 1.
            else:
                return (x - self.par1) / (self.par2 - self.par1)
        elif (self.dist == 'Normal'):
            if self.par2 == 0:
                return 0.
            else:
                return Phi((x - self.par1)/self.par2)
        elif (self.dist == 'Lognormal'):
            if self.mean == 0. or x < 0.:
                return 0.
            else:
                return Phi((log(x) - self.par1) / self.par2)
        elif (self.dist == 'Exponential'):
            if (x < self.par3):
                return 0.
            else:
                return 1. - exp(-self.par1 * (x - self.par3))
        elif (self.dist == 'Rayleigh'):
            if (x < self.par3):
                return 0.
            else:
                return 1. - exp(-1./2. * ((x - self.par3) / self.par1) ** 2)
        elif (self.dist == 'Logistic'):
            return 1./(1. + exp(-(x - self.par1) / self.par2))
        elif (self.dist == 'Gamma'):
            if (x < 0.):
                return 0.
            else:
                return gammainc(self.par1, x * self.par2)
        elif (self.dist == 'Gumbel_MIN'):
            return 1. - exp(-exp(self.par2 * (x - self.par1)))
        elif (self.dist == 'Gumbel_MAX'):
            return exp(-exp(-self.par2 * (x - self.par1)))

    def invcdf(self, Fx):
        if (self.dist == 'Uniform'):
            return (self.par2 - self.par1) * Fx + self.par1
        elif (self.dist == 'Normal'):
            return self.par2 * InvPhi(Fx) + self.par1
        elif (self.dist == 'Lognormal'):
            return exp(self.par2 * InvPhi(Fx) + self.par1)
        elif (self.dist == 'Exponential'):
            return (self.par3 - log(1. - Fx)) / self.par1
        elif (self.dist == 'Rayleigh'):
            return self.par3 + self.par1 * sqrt(-2. * log(1. - Fx))
        elif (self.dist == 'Logistic'):
            return self.par1 - self.par2 * log(1./Fx - 1.)
        elif (self.dist == 'Gamma'):
            return optimize.root(lambda x: gammainc(self.par1, x * self.par2) - Fx, x0 = self.mean, method='hybr').x[0]
        elif (self.dist == 'Gumbel_MIN'):
            return self.par1 + log(-log(1. - Fx)) / self.par2
        elif (self.dist == 'Gumbel_MAX'):
            return self.par1 - log(-log(Fx)) / self.par2

    def random(self):
        if (self.dist == 'Uniform'):
            return np.random.uniform(low = self.low, high = self.high, size = self.size)
        if (self.dist == 'Normal'):
            return np.random.normal(loc = self.mean, scale = self.sdev, size = self.size)
        if (self.dist == 'Lognormal'):
            return np.random.lognormal(mean = self.mean, sigma = self.sdev, size = self.size)
        if (self.dist == 'Exponential'):
            return np.random.exponential(scale = self.sdev, size = self.size)
        if (self.dist == 'Rayleigh'):
            return np.random.rayleigh(scale = self.sdev, size = self.size)
        if (self.dist == 'Logistic'):
            return np.random.logistic(loc = self.mean, scale = self.sdev, size = self.size)
        if (self.dist == 'Gamma'):
            return np.random.gamma(shape = self.shape, scale = self.sdev, size = self.size)
        if (self.dist == 'Gumbel'):
            return np.random.gumbel(loc = self.mean, scale = self.sdev, size = self.size)

# SINGLE LOOP APPROACH (SLA)
def SLA(f, df, g, betaT, d = [], x = [], p = [], R = None, bndsd = None, bndsX = None, bnds_cons = None, cons = None, maxciclo = 50, ftol = 1e-4, disp = False, OPTmaxiter = 1000, OPTtol = 1e-4):

    nVDt = len(d)
    nVAP = len(x)
    nPAP = len(p)
    nVP = nVDt + nVAP
    nVA = nVAP + nPAP
    nLS = len(betaT)

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

    # Jacobian of the Objective Function
    def dfaux(var):
        d = var[:nVDt]
        x = var[nVDt:]
        return df(d, x)

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
            grad[i + nVAP] = (g(d, x, p + dp)[ls] - g(d, x, p - dp)[ls]) / (2 * step)
        return grad

    # Bounds
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
    k = 0
    xMPP = np.empty((nLS, nVAP))
    pMPP = np.empty((nLS, nPAP))
    alfa = np.empty((nLS, nVA))
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
        x = var[nVDt:]
        for i in range(nLS):
            Xi = x - np.matmul(Jxy[:nVAP, :], alfa[i, :]) * betaT[i]
            constr[i] = g(d, Xi, pMPP[i, :])[i]
        return constr

    # Deterministic equality constraints
    def cons_aux_eq(var):
        constr = np.empty(ncons)
        d = var[:nVDt]
        x = var[nVDt:]
        for i in range(ncons):
            constr[i] = cons(d, x)[i]
        return constr

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

    # Deterministic Optimization
    def optimization(VP):
        res = minimize(fun=faux, jac = dfaux, x0=VP, method='slsqp', bounds=bnds, constraints=constr,
                       options={'disp': True, 'maxiter': OPTmaxiter, 'ftol': OPTtol})
        d = res.x[:nVDt]
        mean_x = res.x[nVDt:]
        VP = np.concatenate((d, mean_x))
        f_new = res.fun
        return d, mean_x, VP, f_new, res

    # Update random variables
    def update(mean_x):
        for i in range(nVAP):
            x[i].mean = mean_x[i]
            x[i].calcpar()

    # Convergence history
    def history(k, d, mean_x, f_new, xMPP, pMPP, VP):
        print('Cycles {}'.format(k))
        print('d: ', d)
        print('mean_x: ', mean_x)
        print('f(d, mean_x): ', f_new)
        print('xMPP: ', xMPP)
        print('pMPP: ', pMPP)
        print('g(d, mean_x, pMPP): ', cons_aux(VP))
        if ncons != 0:
            print('constr(d, mean_x): ', cons_aux_eq(VP))
        # print('Processing Time, cycle {}: '.format(k), round((time.time() - execucao), 3), 's')
        # print('Limit State Functions Evaluations:', Geval)
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    # Iterative Process
    while k <= maxciclo:

        f_old = faux(VP)
        for i in range(nLS):
            Jxz, Jzx, mean_eq = JacobXZ(x + p, np.concatenate((mean_x, mean_p)))
            Jyx = np.matmul(Jyz, Jzx)
            Jxy = np.matmul(Jxz, Jzy)
            gradx = grad_g(d, xMPP[i, :], pMPP[i, :], ls=i)
            grady = np.matmul(np.transpose(Jxy), gradx)
            alfa[i, :] = grady / np.linalg.norm(grady)
            xMPP[i, :] = mean_x - np.matmul(Jxy[:nVAP, :], alfa[i, :]) * betaT[i]
            pMPP[i, :] = mean_p - np.matmul(Jxy[nVAP:, :], alfa[i, :]) * betaT[i]

        d, mean_x, VP, f_new, res = optimization(VP)
        k += 1
        update(mean_x)

        # Convergence history
        if disp == True:
            history(k, d, mean_x, f_new, xMPP, pMPP, VP)

        # Convergence criteria
        if abs((f_new - f_old) / f_new) <= ftol:
            break

    out = {'d = ': d, 'mean_x = ': mean_x, 'fun = ': f_new, 'xMPP = ': xMPP, 'pMPP = ': pMPP, 'g': cons_aux(VP)[:nLS], 'constr': cons_aux(VP)[nLS:], 'nciclos': k}

    return OptimizeResult(out)