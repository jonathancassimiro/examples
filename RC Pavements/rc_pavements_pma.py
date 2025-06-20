from codes import AnalysisOptions, StochasticModel, LimitState, Form, Normal, Lognormal
from scipy.optimize import minimize
import numpy as np
import pdb

# Roadway platform
L = 900.                                                             # Length of the tangent section of the highway, in m
l = 3.75                                                             # Travel lane width, in m
so = 3.5                                                             # Outer shoulder width, in m
si = 1.2                                                             # Inner shoulder width, in m
A = 7.5                                                              # Slab width, in m
B = 15.                                                              # Slab length, in m
C = 0.5                                                              # Anchorage length of the dowels and tie bars, in m
s = 0.3                                                              # Spacing for dowels bars, in m
d = 0.15                                                             # The slab thickness of the outer and inner shoulders, in m

# Vehicle of design
n1 = 0.3                                                             # Distance between wheels on dual axles, m
n2 = 0.3                                                             # Spacing between single and tandem axle sets, m
q = 0.7                                                              # Tire inflation pressure, MPa

# Data of design
v = 0.15                                                             # Poisson's ratio
fck = 25                                                             # Concrete compressive strength, in MPa
fyk = 500                                                            # Steel yield strength, in MPa
P = 170                                                              # Total load of simple axle, in kN
Ec = (0.8 + 0.2 * fck / 80) * 5600 * np.sqrt(fck)                    # Concrete longitudinal modulus of elasticity, MPa
k = 60                                                               # Modulus of Westergaard, in MPa
ρs = 7850.                                                           # Steel specific density, kg/m³
ρmin = 0.00075                                                       # Minimum steel rate
c = 0.03                                                             # Concrete cover to reinforcement, in m
γc = 1.0                                                             # Concrete safety coefficient
γs = 1.0                                                             # Steel safety coefficient
λ = 0.8                                                              # Ratio between the neutral axis depth in the parabola-rectangle diagram and the equivalent rectangular stress diagram in concrete
F = 1.7                                                              # Coefficient of friction between the slab and the sub-base layer

# Auxiliary functions
def MR(As, A, fck, fyk, h_):
    σcd = 0.85 * fck / γc                                            # MPa
    b = 1                                                            # m
    x = 1.25 * (As/1e4) * A * (fyk / γs) / (λ * b * σcd)             # The neutral axis's depth, in m
    μ = 0.5 * (1 - (1 - (x / (1.25 * h_)))**2)                       # The reduced bending moment
    MR = μ * b * (h_)**2 * σcd * 1000                                # Ultimate bending, kNm
    return MR
def MS_center(Ec, h, v, k, P, q, n1, n2):

    l = ((Ec * (h/1e2)**3) / (12 * (1 - v**2) * k))**(1/4)           # The radius of relative stiffness of the slab, in m
    n5 = P / (4 * q * 1000)                                          # The contact area, in m²
    n4 = np.sqrt(n5 / 0.523)                                         # The tire length, in m
    n3 = 0.6 * n4                                                    # The tire width, in m

    # Pickett and Ray’s charts in the center of slab
    chart_center = {0.0: {0.1: 50, 0.2: 100, 0.3: 150, 0.4: 225, 0.5: 300, 0.6: 400, 0.7: 500, 0.8: 590, 0.9: 700, 1.0: 790},
                    0.2: {0.1: 15, 0.2: 30, 0.3: 50, 0.4: 110, 0.5: 180, 0.6: 250, 0.7: 350, 0.8: 450, 0.9: 550, 1.0: 650},
                    0.4: {0.1: 15, 0.2: 30, 0.3: 40, 0.4: 60, 0.5: 95, 0.6: 140, 0.7: 190, 0.8: 250, 0.9: 325, 1.0: 390},
                    0.8: {0.1: 5, 0.2: 10, 0.3: 15, 0.4: 20, 0.5: 25, 0.6: 30, 0.7: 40, 0.8: 50, 0.9: 70, 1.0: 85}}

    tire1 = chart_center.get(np.round(0. / l[0], 1), {}).get(np.round(min(max(n4[0] / l[0], 0.1), 1.0), 1), None)

    key_n2 = min(chart_center.keys(), key=lambda k: abs(k - n2 / l[0]))
    tire2 = chart_center.get(key_n2, {}).get(np.round(min(max(n4[0] / l[0], 0.1), 1.0), 1), None)

    N_center = (tire1 if tire1 is not None else 0) + (tire2 if tire2 is not None else 0)
    MS_center = (N_center * q * l**2) / 10                           # Bending moment, in kNm/m

    return MS_center
def k3(k6, fck):
    k6_25 = [18638, 4710, 2118, 1206, 781.7, 550.4, 410.3, 319.0, 256.2, 211.2, 177.8, 152.4, 132.7, 117.1, 104.5, 94.3, 85.94, 79.01, 73.19, 68.22, 63.94, 60.2, 56.92, 54.01, 51.42, 49.14, 47.54, 46.06, 44.68, 43.39, 42.2, 41.07, 40.02, 39.03, 38.1, 37.22, 36.4, 35.61, 34.87, 34.17, 33.5, 32.87]
    k6_30 = [15531, 3925, 1726, 1005, 651.4, 458.6, 341.8, 265.8, 213.5, 176, 148.2, 127, 110.6, 97.54, 87.08, 78.58, 71.62, 65.85, 60.99, 56.85, 53.28, 50.17, 47.43, 45.01, 42.85, 40.95, 39.62, 38.38, 37.23, 36.16, 35.16, 34.23, 33.35, 32.53, 31.75, 31.02, 30.33, 29.68, 29.06, 28.47, 27.92, 27.39]
    k3_list = [0.269, 0.270, 0.271, 0.272, 0.273, 0.274, 0.275, 0.276, 0.277, 0.278, 0.279, 0.280, 0.282, 0.283, 0.284, 0.285, 0.287, 0.288, 0.290, 0.291, 0.293, 0.294, 0.296, 0.298, 0.299, 0.301, 0.302, 0.304, 0.305, 0.307, 0.308, 0.309, 0.311, 0.312, 0.314, 0.316, 0.317, 0.319, 0.320, 0.322, 0.324, 0.325]

    if fck == 25:
        index = min(range(len(k6_25)), key=lambda i: abs(k6_25[i] - k6))
    elif fck == 30:
        index = min(range(len(k6_30)), key=lambda i: abs(k6_30[i] - k6))
    else:
        raise ValueError("fck equal to 25 or 30 (in MPa)")

    return k3_list[index]

def lsf(x, g):
    PMA = True
    options = AnalysisOptions()
    options.is_PMA = PMA
    options.beta_targ = betaT
    stochastic_model = StochasticModel()
    stochastic_model.addVariable(Normal('h', x[0], 0.025*x[0]))
    stochastic_model.addVariable(Normal('As_center', x[1], 0.025*x[1]))
    stochastic_model.addVariable(Normal('As_edge', x[2], 0.025*x[2]))
    stochastic_model.addVariable(Normal('As_shrinkage', x[3], 0.025*x[3]))
    stochastic_model.addVariable(Lognormal('fck', fck, 0.100*fck))
    stochastic_model.addVariable(Lognormal('fyk', fyk, 0.050*fyk))
    stochastic_model.addVariable(Lognormal('k', k, 0.100*k))
    stochastic_model.addVariable(Normal('P', P, 0.100*P))
    limit_state = LimitState(g)
    Analysis = Form(analysis_options=options, stochastic_model=stochastic_model, limit_state=limit_state)
    Analysis.run()
    return Analysis.G

# MATERIAL COSTS
crp = np.array([236.38, 256.41, 251.17, 250.56, 232.61, 225.69, 220.24, 217.75, 224.85, 190.76, 193.55, 199.08, 207.43, 237.67, 249.30, 246.68, 250.98, 257.84, 272.67, 266.86, 269.19, 290.47, 296.86, 310.29])
csw = np.array([3.0872, 3.6903, 3.7657, 4.1411, 4.5500, 4.8664, 4.5157, 4.3521, 4.2608, 4.3179, 4.4257, 4.5529, 5.1400, 6.1700, 7.3727, 8.6210, 9.6266, 10.2702, 9.7652, 9.1645, 7.9485, 8.4367, 7.1065, 6.6856])
cs  = np.array([3.0330, 3.4241, 3.7261, 4.0796, 4.4558, 4.5107, 4.2415, 4.1210, 4.0077, 4.0607, 4.1627, 4.3469, 5.1215, 6.1600, 7.0276, 8.1269, 8.7008, 9.7496, 8.4006, 8.1208, 7.2479, 7.6891, 6.5266, 6.2372])

f_opt_list = []
x_opt_list = []

# LOOP OPTIMIZATION
for i in range(24):

    # OBJECTIVE FUNCTION
    def f(x):

        h, As_center, As_edge, As_shrinkage = x[0]/1e2, x[1]/1e4, x[2]/1e4, x[3]/1e4

        # rigid pavement volume of concrete, in m3
        Vrp = h * A * L + d*(so+si)*L

        # CA-60 weight of steel, in kg
        Wsw = (L/B) * ((B * As_center * (A - 2*c) + A * As_center * (B - 2*c)) + (B * As_shrinkage * (A - 2*c) + A * As_shrinkage * (B - 2*c)) + (1.6 * As_edge * (B - 2*c))) * ρs

        # CA-50 weight of steel, in kg
        if h <= 0.175:
            phi_tb = phi_lb = 0.020
        elif h <= 0.225:
            phi_tb = phi_lb = 0.025
        elif h <= 0.300:
            phi_tb = phi_lb = 0.032
        else:
            phi_tb = phi_lb = 0.040

        Ws = C * (25 * ((L/B) - 1) * (np.pi * (phi_tb)**2 / 4) + 16 * (L/B) * (np.pi * (phi_lb)**2 / 4)) * ρs

        cost = crp[i] * Vrp + csw[i] * Wsw + cs[i] * Ws

        return cost

    # LIMIT STATES FUNCTIONS
    def g1(h, As_center, As_edge, As_shrinkage, fck, fyk, k, P):
        h_ = h/1e2 - c                                          # m
        MR_ = MR(As_center, A, fck, fyk, h_)                    # kNm
        MS_center_ = MS_center(Ec, h, v, k, P, q, n1, n2)       # kNm/m
        return MR_ - MS_center_
    def g2(h, As_center, As_edge, As_shrinkage, fck, fyk, k, P):
        h_ = h/1e2 - c                                          # m
        MR_ = MR(As_edge, A, fck, fyk, h_)                      # kNm
        MS_center_ = MS_center(Ec, h, v, k, P, q, n1, n2)       # kNm/m
        MS_edge_ = 2 * MS_center_                               # kNm/m
        return MR_ - MS_edge_
    def g3(h, As_center, As_edge, As_shrinkage, fck, fyk, k, P):
        h_ = h/1e2 - c                                          # m
        MS_center_ = MS_center(Ec, h, v, k, P, q, n1, n2)       # kNm/m
        k6 = 100 * (h_*1e2)**2 / (10.2 * MS_center_)
        k3_ = k3(k6[0], fck=25)
        As_center_calc = k3_ * (10.2 * MS_center_) / (h_*1e2)   # cm²/m
        As_min = ρmin * B*1e2 * h / A                           # cm²/m
        return As_center - np.maximum(As_center_calc, As_min)
    def g4(h, As_center, As_edge, As_shrinkage, fck, fyk, k, P):
        h_ = h/1e2 - c                                          # m
        MS_center_ = MS_center(Ec, h, v, k, P, q, n1, n2)       # kNm/m
        MS_edge_ = 2 * MS_center_                               # Nm/m
        k6 = 100 * (h_*1e2)**2 / (10.2 * MS_edge_)
        k3_ = k3(k6[0], fck=25)
        As_edge_calc = k3_ * (10.2 * MS_edge_) / (h_*1e2)       # cm²/m
        As_min = ρmin * B*1e2 * h / A                           # cm²/m
        return As_edge - np.maximum(As_edge_calc, As_min)
    def g5(h, As_center, As_edge, As_shrinkage, fck, fyk, k, P):
        As_shrinkage_calc = F * B * h / 333.                    # cm²/m
        As_min = ρmin * B*1e2 * h / A                           # cm²/m
        return As_shrinkage - np.maximum(As_shrinkage_calc, As_min)

    cons = ({'type': 'ineq', 'fun': lambda x: lsf(x, g1)},
            {'type': 'ineq', 'fun': lambda x: lsf(x, g2)},
            {'type': 'ineq', 'fun': lambda x: lsf(x, g3)},
            {'type': 'ineq', 'fun': lambda x: lsf(x, g4)},
            {'type': 'ineq', 'fun': lambda x: lsf(x, g5)})

    # TARGET RELIABILITY INDEX
    betaT = 3.

    # BOUNDS
    bnds = [[10., np.inf], [1., np.inf], [1., np.inf], [1., np.inf]]

    # INITIAL POINTS
    x0 = [15., 2.25, 2.25, 2.25]

    # OPTIMIZATION
    res = minimize(fun=f, x0=x0, method='slsqp', bounds=bnds, constraints=cons, options={'disp': False, 'maxiter':1000, 'ftol': 1e-4})
    f_opt_list.append(res.fun)
    x_opt_list.append(res.x)

# RESULTS
for i in range(24):
    print(f"f = {f_opt_list[i]:.2f}, x = {x_opt_list[i]}")
    print(f_opt_list)
    print(x_opt_list)