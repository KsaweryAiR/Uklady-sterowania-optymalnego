# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:52:04 2022

@author: ksawe
"""

import numpy as np
from scipy import linalg
from scipy import signal
from numpy import log as ln
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import control as ctrl

def zadanie2():

    u = 1;
    t = np.arange(0,10,0.1)
    def model1(z, t):
        z1 = z[0]
        z2 = z[1]
        dz1 = z1*ln(z2)
        dz2 = -z2*ln(z1)+z2*u
        return [dz1, dz2]
    y1 = odeint(model1,[1,1],t)
   
    def model2(x ,t):
        x1 = x[0]
        x2 = x[1]
        dx1 = x2
        dx2 = -x1+u
        return [dx1, dx2]
    y2 = odeint(model2,[ln(1),ln(1)],t)
  
    #2.1
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,y1[:, 0], label="z1")
    plt.plot(t,y1[:, 1], label="z2")
    plt.legend()
    
    #2.2 #inny układ współrzędnych (inny dobór zmiennych stanu) 
    plt.subplot(3,1,2)
    plt.plot(t, y2[:, 0], label="x1")
    plt.plot(t,y2[:, 1], label="x2")
    plt.legend()
    
    #2.5
    y3 = np.exp(y2)
    plt.subplot(3,1,3)
    plt.plot(t, y3[:, 0], label="exp(x1)")
    plt.plot(t, y3[:, 1], label="exp(x2)")
    plt.legend()
    plt.show()

    R = 1
    m = 9
    J = 1
    g = 10
    d = 0.5
    #liniowy model wahadła
    x0 = [0,0]
    #u0 = [0, 5, 20, 45*np.sqrt(2), 70]
    u0 = 70
    A = np.matrix([[0,1],[-(m*g*R)/J, -d/J]])
    B = np.matrix([[0],[1/J]])
    C = np.matrix([0, 0])
    D = 0
    t = np.arange(0,10,0.01)
    
    def model2(x,t):
        u  = u0
        x1 = x[0]
        x2 = x[1]
        dx1 = x2
        dx2 = 1/J*u - d/J*x2 -(m*g)/J*R*np.sin(x1)
        return [dx1 , dx2]
    y1 = odeint(model2,[0,0],t)
    plt.figure(2)
    # 5.1
    plt.subplot(2,1,1)
    plt.plot(t, y1[:, 0], label="x1")
    plt.plot(t, y1[:, 1], label="x2")
    plt.legend()
    
    def model3(x,t):
        u = u0
        x1 = np.array([x]).T
        dx3 = A@x1 + B*u
        dx1 = np.ndarray.tolist(dx3.T[0])
        return dx1[0]
    y3 = odeint(model3,[0,0],t)
    plt.figure(2)
    plt.plot(t,y3)
#zadanie2()
#zadanie5()


def zadanie5():
    # parametry obiektu
    J = 1 # kg*m^2
    g = 10 # m/s^2
    R = 1 # m
    m = 9 # kg
    d = 0.5 # Nm*s^2/rad^2

    # wymuszenie
    u_idx = 2
    u = [0, 5, 20, 45*np.sqrt(2), 70][u_idx]
    # u = 0
    
    def ex5_1_odeint(x, t):
        x1, x2 = x
        dx1dt = x2
        dx2dt = 1/J*u-d/J*x2-m*g/J*R*np.sin(x1)
        return [dx1dt, dx2dt]

    t = np.linspace(0, 15, num=200)

    # 5.1
    x51 = odeint(ex5_1_odeint, [0, 0], t) # u = 0 tutaj
    plt.subplot(2,1,1)
    plt.plot(t, x51[:, 0], label="x1")
    plt.plot(t, x51[:, 1], label="x2")
    plt.legend()

    # 5.2
    x0 = [np.pi/4, 0]
    u0 = np.sqrt(2)*45

    A = np.array([[0, 1], [-m*g*R/J*np.cos(x0[0]), -d/J]])
    B = np.array([[0], [1/J]])
    #C = np.array([[1, 0]])
    #D = 0
    
    # 5.4 #kalman
    print(f"sterowalny={np.linalg.matrix_rank(ctrl.ctrb(A,B)) == np.linalg.matrix_rank(A)}")

    def ex5_2_odeint(x,t):
        x = x.reshape((2,1))
        dxdt = A@x+B*(u+u0)
        return dxdt.flatten()

    #sys = ss.StateSpace(A,B,C,D)
    #_, _, x52 = ss.lsim2(sys, np.ones_like(t)*u+u0, t, x0)
    x52 = odeint(ex5_2_odeint, [0,0], t)

    plt.subplot(2,1,2)
    plt.plot(t, x52[:, 0] + x0[0], label="x1 (lin)")
    plt.plot(t, x52[:, 1] + x0[1], label="x2 (lin)")
    plt.legend()
    plt.show()

    # 5.7
    u = 0
    x0_5_7 = [np.pi/4, 0] # [0, 0]
    x57 = odeint(ex5_1_odeint, x0_5_7, t)
    plt.subplot(2, 1, 1)
    plt.plot(t, x57[:, 0], label="x1")
    plt.plot(t, x57[:, 1], label="x2")
    plt.legend()

    # 5.8
    def x_to_SDC_AB(x):
        x1, _ = x
        A = np.array([[0, 1], [-m*g*R/J*np.sin(x1)/x1, -d/J]])
        B = np.array([[0], [1/J]])
        return A, B

    # 5.9
    def ex5_9_odeint(x, t):
        A, B = x_to_SDC_AB(x)
        x = x.reshape((2,1))
        return (A@x+B*u).flatten()

    x59 = odeint(ex5_9_odeint, x0_5_7, t)
    plt.subplot(2, 1, 2)
    plt.plot(t, x59[:, 0], label="x1 SDC")
    plt.plot(t, x59[:, 1], label="x2 SDC")
    plt.legend()
    plt.show()

    # Uzyskane wykresy pokrywają się
    # Na przykład problemem może być dzielenie przez 0 przy liczeniu macierzy jak x0=0 (jak w 5.10)
    # Linearyzacja to dalej przyblizenie wiec moze blad wystepowac

#zadanie2()
#zadanie5()
