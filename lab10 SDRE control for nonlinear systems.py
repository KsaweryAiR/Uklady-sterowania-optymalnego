# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:04:59 2023

@author: ksawe
"""

import numpy as np
from scipy import linalg
from scipy import signal
from numpy import log as ln
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import control
from scipy.linalg import solve_continuous_are

#zad 1 
m = 9
J = 1
l = 1
d = 0.5
g = 9.81
t = np.arange(0,10,0.1)
x0 = [np.pi/4, 0]
u0= 0

#1.2
def A(x) :
    return np.matrix([[0 , 1] , [m* g*l* np . sin (x [0]) /x [0] , -d/J ]])
def B(x) :
    return np.matrix([[0] ,[1/ J ]])
def Q(x) :
    return np.matrix([[ x [0]**2 ,0] ,[0 , x [1]**2]])
def R(x) :
    return np.matrix([[1]])

#1.1
def model(x,t):
    u = u0
    x1 = np.array([x]).T
    dx3 = A(x)@x1 + B(x)*u
    dx1 = np.ndarray.tolist(dx3.T[0])
    return dx1[0]
y = odeint(model,x0,t)
plt.figure(1)
plt.plot(t,y)


tf = 10
t= np . arange (0 , tf ,0.01)
#LQR
def control (t ,x):
    P = solve_continuous_are(A(x),B(x),Q(x),R(x))
    K = R(x).I @ B(x).T@P
    u = -K@np.matrix([x]).T
    return u

def model1 (t , x):
    u = control(t , x)
    #u = 0
    dx = x.copy()
    dx [0] = x [1]
    dx [1] = 1/ J*u - d/J* x [1] + m*g* l* np . sin (x [0])
    return dx

res = odeint(model1 ,[2*np.pi ,0] , t , tfirst = True , rtol =1e-10)

plt.figure(2)
plt.plot(t , res [: ,0])
plt.plot(t , res [: ,1])
plt.grid() ;

