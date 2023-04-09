# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:19:17 2023

@author: ksawe
"""

import numpy as np
from scipy import linalg
from scipy import signal
from numpy import log as ln
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import control
import sympy

#MACIERZE
A = np.matrix([[0,1],[0,0]])
B = np.matrix([[0],[1]])

#2
#zmienne symboliczne
a1,a2,c1,c2, t =sympy.symbols('a1 a2 c1 c2 t')
    
#2.1
x2 = (a1*t**2)/2-a2*t+c1
x1 = (a1*t**3)/6-(a2*t**2)/2+c1*t +c2
########
lam1 = a1
lam2 = -a1*t+a2
    
#2.2
T = 1 
eq2T1 = sympy.Eq((a1*T**2)/2-a2*T+c1, 0)
eq1T1 = sympy.Eq((a1*T**3)/6-(a2*T**2)/2+c1*T+c2, 0)
    
T = 0
eq2T0 = sympy.Eq((a1*T**2)/2-a2*T+c1, 1)
eq1T0 = sympy.Eq((a1*T**3)/6-(a2*T**2)/2+c1*T+c2,1)
    
solution = sympy.solve((eq2T1, eq1T1, eq2T0, eq1T0), (a1, a2, c1 ,c2))
    
# wynik z solution

a1 = 18
a2 = 10
c1 = 1
c2 = 1
    
lam1 = a1
lam2 = -a1*t+a2
print(lam1, lam2)
print(solution)

t = np.arange(0,10,0.01)
u = -a1*t+a2
    
def model2(x,t):
     x2 = (a1*t**2)/2-a2*t+c1
     x1 = (a1*t**3)/6-(a2*t**2)/2+c1*t +c2
     return [x1 , x2]
y1 = odeint(model2,[0,0],t)
plt.figure(2)
plt.plot(t,y1)
   
     
# def zadanie3():
#     H = 1 + lam1*x2+lam1*u
#     lam1_poch = 0
#     lam2_poch = -lam1


# Definiujemy funkcję Hamiltonianu

# x2e = (a1*t**2)/2-a2*t+c1
# x1e = (a1*t**3)/6-(a2*t**2)/2+c1*t +c2
x1e, x2e, lam1e, lam2e, t = sympy.symbols('x1e x2e lam1e lam2e te')
H = 1 + lam1e*x1e**2 + lam2e*x2e**2

# Definiujemy równania zmiennych sprzężonych
# dx1dt = -sympy.diff(H, x1e)
# dx2dt = -sympy.diff(H, x2e)
# dlam1dt = sympy.diff(H, lam1e)
# dlam2dt = sympy.diff(H, lam2e)

# dx1dt = lambda x1e, x2e, lam1e, lam2e, te: -sympy.diff(H, x1e)
# dx2dt = lambda x1e, x2e, lam1e, lam2e, te: -sympy.diff(H, x2e)
# dlam1dt = lambda x1e, x2e, lam1e, lam2e, te: sympy.diff(H, lam1e)
# dlam2dt = lambda x1e, x2e, lam1e, lam2e, te: sympy.diff(H, lam2e)


def dx_dt(x, t, a1, a2, c1, c2):
    x1, x2, lam1, lam2 = x
    dx1dt = a1*t**2 - a2*t + c1
    dx2dt = a1*t**3 - a2*t**2 + c1*t + c2
    dlam1dt = 0
    dlam2dt = -lam1
    return [dx1dt, dx2dt, dlam1dt, dlam2dt]

x0 = [0, 1, 0, -1]
y2 = odeint(dx_dt, x0, t, args=(a1, a2, c1, c2))
# Definiujemy warunki początkowe
x1_0 = 1
x2_0 = 1
lam1_0 = 0
lam2_0 = 0


# Definiujemy czas symulacji
te = np.linspace(0, 10, 100)

# Rozwiązujemy równania różniczkowe
# x1_sol = odeint(dx1dt, x1_0, te)
# x2_sol = odeint(dx2dt, x2_0, te)
# lam1_sol = odeint(dlam1dt, lam1_0, te)
# lam2_sol = odeint(dlam2dt, lam2_0, te)

# Wykres rozwiązań
# plt.plot(te, x1_sol, 'b', label='x1')
# plt.plot(te, x2_sol, 'g', label='x2')
# plt.plot(te, lam1_sol, label='lam1')
# plt.plot(te, lam2_sol, label='lam2')
# #zadanie2()
