from scipy.optimize import root
from scipy.special import gamma, gammainc, erf, erfinv
from numpy import *
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