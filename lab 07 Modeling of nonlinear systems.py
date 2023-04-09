# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:06:29 2022

@author: ksawe
"""

import numpy as np
from scipy import linalg
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, figure, show, title, legend
from numpy import linspace, power, sqrt, clip, cos, sin, pi
from scipy.constants import g as g_acc

def zadanie2():
    c=0
    time = np.arange(0,10,0.1)
    #numerycznie
    def model(x, t):
         dx = t**2
         return dx
    y = odeint(model,0,time)
    plt.figure(1)
    plt.plot(time,y, label="odeint")
    
    #analitycznie
    y1 = 1/3*time**3+c
    plt.figure(1)
    plt.plot(time,y1, label="analityczna")
    legend()
    show()

#zadanie 3
def zadanie3():
    #parameters
    kp = 2
    omega = 4
    ksi = 0.25
    u = 1 
    t2 = np.arange(0,100,0.1)
    def model2(x, t):
         y = x[0]
         dydt = x[1]
         dy2dt2 = (-2*ksi/omega)*x[1]-(1/omega*np.sqrt(x[0]))+(kp/omega**2*u)
         return [dydt, dy2dt2]
    y2 = odeint(model2,[0,0],t2)
    plt.figure(2)
    plt.plot(t2,y2)

#zadanie 4
def zadanie4():
    # parametry obiektu
    kp = 2
    T = 2
    k_ob = 4

    def feedback(x, t, z):
        # model
        # uchyb
        yt = x[-1]
        et = z - yt
        #ut = kp*et #bez granicy
        ut = clip(kp*et, -0.1, 0.1)  # z granicą
        # stan
        dxdt = -1/k_ob*x + 1*ut

        return dxdt

    #wykresy
    
    t = linspace(0, 15, 200)
    x0 = 0
    xz1 = odeint(feedback, x0, t, args=(1,))
    xz2 = odeint(feedback, x0, t, args=(2,))
    xz3 = odeint(feedback, x0, t, args=(3,))
    
    yz1 = k_ob/T * xz1
    yz2 = k_ob/T * xz2
    yz3 = k_ob/T * xz3
    
    # wyswietlanie
    figure()
    plot(t, yz1, label='z = 1')
    plot(t, yz2, label='z = 2')
    plot(t, yz3, label='z = 3')
    legend()
    show()

def zadanie5():
    def odeint_model(x, t):
    # parametry modelu
        A = 1.5
        w = 0.65
        dump=0.5
        m = 0.01
        R = 10
        J = m*R**2

        # wymuszenie
        tau_m = A*cos(w*t)
        
        # stan
        x1, x2 = x

        dx1dt = x2
        dx2dt = tau_m/J - dump/J*x2 - m*g_acc/J*R*sin(x1)
        return [dx1dt, dx2dt]
    
    t = linspace(0, 150, 500)
    x0 = [pi, 0]
    x = odeint(odeint_model, x0, t)
    theta = x[:, 0] * 180/pi
    figure()
    plot(t, theta, label=r'$\theta (t)$')
    legend()
    show()
    
zadanie2()
zadanie3()
zadanie4()
zadanie5()    
