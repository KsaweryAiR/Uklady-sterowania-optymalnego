# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:52:04 2022

@author: ksawe
"""
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import scipy.interpolate as interp
from scipy.integrate import odeint
m = 9
J = 1
d = 0.5
g = 9.81
l = 1
tau = 1
# A = np.array([[0,1],[-m*g*l*sin(x1),-d/J]])
# B = np.array([[0],[tau/J]])
Q = np.eye(2)
R = 1
A = np.array([[0, 1], [-m * g * l / J * np.cos(np.pi), -d / J]])
B = np.array([[0], [tau / J]])
P = solve_continuous_are(A, B, Q, R)
K = 1 / R * B.T @ P

#2.2
def model(x,t):
    u = 1
    dx1 = x[1]
    dx2 = u*tau/J -d/J*x[1] -m*g*l*np.sin(x[0])
    return dx1,dx2
#2.3
def model_lin(x,t):
    u = -K@(x-[np.pi,0]) + 0
    dx2 = A@x +B@u
    return dx2
#2.4
print(K)
#2.5
def układ_1():
    t = np.linspace(0, 5, 100)
    sol = odeint(model, [np.pi/4,0], t)
    sol1 = odeint(model_lin, [np.pi-0.1, 0], t)
    #2.5 tak układ jest stabilny
    #2.6 chyba tak
    #2.7 Nie ale nie wiem dlaczego
    # print(sol)
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(t, sol, label="obiekt")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(t, sol1-[np.pi-0.1],label="lioniowa aproksymacja")
    plt.legend()
  
   

def riccati(p, t):
    p = np.reshape(p,(2,2))
    pdot = -(p@A + A.T@p -p@B*(1/R)@B.T@p +Q)
    return pdot.flatten()

def model2(x,t,f1,f2,f3,f4):
    x_temp = np.array([[x[0]],[x[1]]])
    p = np.array([[f1(t),f2(t)],[f3(t),f4(t)]])
    K = 1 / R * B.T @ p

    u = -K@(x-[np.pi,0]) + 0
    dx = A@x + B@u

    return dx.flatten()
def układ_2():
    p0 = np.ones((2, 2))

    # Oblicz przebieg wartości macierzy P w czasie
    t = np.linspace(2, 0, 500)
    p = sci.integrate.odeint(riccati, p0.flatten(), t)

    s = interp.interp1d(t,p[:,0],fill_value="extrapolate")
    s1 = interp.interp1d(t, p[:, 1], fill_value="extrapolate")
    s2 = interp.interp1d(t, p[:, 2], fill_value="extrapolate")
    s3 = interp.interp1d(t, p[:, 3], fill_value="extrapolate")
    t = np.linspace(0, 1, 500)
    f = odeint(model2, [np.pi,0], t,args=(s,s1,s2,s3))
    plt.figure(2)
    plt.plot(t, f)

    

if __name__ == '__main__':
    układ_1()
    układ_2()