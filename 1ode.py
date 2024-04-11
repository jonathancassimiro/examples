import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parâmetros
T = 5               # time constant of the system
x0 = 1
t = 0 

a = -1/T
ti = 0; tf = 25
h = 1
N = 25

x = []
x = np.zeros(tf+1)
t = range(ti, tf + 1, h)
#t = np.linspace(ti, tf, N+1)

# Utilizando o comando range
for i in range(tf):
    x[i] = mt.exp(a*t[i])*x0

# Utilizando o comando linspace
#x = np.exp(a*t)*x0

print(t, x)

# Plotando o gráfico
plt.plot(t, x)
plt.title('Plotting Differential Equation Solution')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.axis([0, 25, 0, 1])
plt.show()

