# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:57:22 2022

@author: ksawe
"""

import numpy as np
import matplotlib .pyplot as plt
from scipy.integrate import odeint

R1 = 2
R2 = 5 
C1 = 0.5
L1 = 2
L2 = 0.5
A = np.matrix([[-R2/L2, 0, -1/L2],[0, -R1/L1, 1/L1], [1/C1, -1/C1, 0]])
B = np.matrix([[0],[1/L1],[0]])
C = np.matrix([[1 ,0 , 0]])
D = 0
#t = np.linspace(0,6, 20)
t = np.arange(0.1,15,0.1)
u = 1
def zadanie2():
    #zad2()
   
    def model(x, t):
         x1 = np.array([x]).T
         u1 = np.array([u])
         dx = A@x1 + B@u1
         dx1 = np.ndarray.tolist(dx.T[0])
         return dx1[0]
    
    y = odeint(model,[1,1,1],t)
    plt.figure(0)
    plt.plot(t,y)
    
    #2.3/2.4 PID
    # nastawy PID
    kp = 1
    ki = 1
    kd = 8
    yd = 3 #wartoć zadana
    
    # nastawy PID dla zigera zad3
    # ku = 75.5
    # Tu = 1.85

    # kp = 0.6*ku
    # ki = 1.2*ku*Tu**-1
    # kd = 0.075*ku*Tu
    
    def model2(x, t, Kp, Ki, Kd, setpoint):
        # x[0] = pozycja, x[1] = prędkość, x[2] = całka błędu
        x1 = np.array([x]).T
        error = setpoint - x1[0]
        integral = x1[1] + error * t
        derivative = (error - x1[2])/t
        u = Kp * error + Ki * integral + Kd * derivative
        u1 = np.array([u])
        dx = A@x1 + B@u1
        dx1 = np.ndarray.tolist(dx.T[0])
        return dx1[0]#, error, integral, derivative
    
    y1 = odeint(model2,[1,1,1],t, args=(kp, ki, kd, yd))
    plt.figure(1)
    plt.plot(t,y1)
    
    #zad 3
    I_ISE = np.mean((y1 - yd) ** 2)
    I_ITSE = np.mean(t*I_ISE)
    I_IAE = np.mean(np.abs(y1 - yd))
    I_ITAE = np.mean(t*I_IAE)
    I_OPT = np.mean(I_ISE + u**2)
    print('I_ISE ='+str(I_ISE))
    print('I_ITSE ='+str(I_ITSE))
    print('I_IAE ='+str(I_IAE))
    print('I_ITAE ='+str(I_ITAE))
    print('I_OPT ='+str(I_OPT))
    

zadanie2()