# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:11:57 2022

@author: ksawe
"""
import numpy as np
import matplotlib .pyplot as plt
from scipy import signal
from scipy.integrate import odeint
from scipy.signal import tf2ss
from scipy.signal import ss2tf
from scipy.signal import bessel, lsim2

def zadanie2():
    #2.1
    #2.2 zmienne
    k_p = 3
    T = 2
    A = - 1/T
    B = k_p/T
    C = 1
    D = 0
    #2.3
    num = [k_p]
    den = [T, 1]
    sys= signal.TransferFunction(num,den)
    #2.4
    t, y =signal.step(sys)
    plt.figure(0)
    plt1 = plt.plot(t,y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('sys step')
    plt.grid()
    #Czy odpowiedź skokowa odpowiada teoretycznym założeniom?
    text = print('zad 2.4: Tak, odpowiada')
    #2.5
    sys2 = signal.StateSpace(A,B,C,D)
    t1, y2 =signal.step(sys2)
    plt.figure(1)
    plt2 = plt.plot(t1,y2,'r')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('sys step')
    plt.grid()
    #2.6/2.7/2.8
    # initial condition
    t3 = np.linspace(0,15)
    y_0 = 0
    def model(y,t3):    
        k_p = 3
        T = 2
        u = 1
        dydt = (k_p * u - y)/T
        return dydt

    y3 = odeint(model,y_0,t3)
    #graph
    plt.figure(2)
    plt3 = plt.plot(t3,y3,'r-',label='Output (y(t))') 
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.title( '2.6 odeint')
    plt.grid()
    return (plt1, plt2, plt3,text)

def zadanie3():
    #3.1
    R = 12
    L = 1
    C = 0.0001
    
    num=[1,0]
    den=[L,R,1/C]
    obj1 = signal.TransferFunction(num,den)
    t, y =signal.step(obj1)
    t1, y1 = signal.impulse(obj1)
    #step
    plt.figure(3)
    plt1 = plt.plot(t,y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude') 
    plt.title('obj1 step')
    plt.grid()
    #impulse
    plt.figure(4)
    plt2 = plt.plot(t1,y1)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude') 
    plt.title('obj1 impulse')
    plt.grid()

    #3.2
    A = np.matrix([[0,1],[-1/(L*C),-R/L]])
    B = np.matrix([[0],[1/L]])
    C = np.matrix([[0,1]])
    D = 0

    obj2 = signal.StateSpace(A,B,C,D)
    t2, y2 = signal.step(obj2)
    t3, y3 = signal.impulse(obj2)
    #step2
    plt.figure(5)
    plt3 = plt.plot(t2,y2)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude') 
    plt.title('obj2 step')
    plt.grid()
    #impulse2
    plt.figure(6)
    plt4 = plt.plot(t3,y3)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude') 
    plt.title('obj2 impulse')
    plt.grid()
    #Czy wykresy się pokrywają?
    text1 = print('zad 3.2: Wykresy jak najbardziej się pokrywają')
    #3.3 
    A1,B1,C1,D1 = tf2ss(num,den)
    ss = print(A1,B1,C1,D1)
    num1, den1 = ss2tf(A,B,C,D)
    tf = print(num1,den1)
    # Czy wyprowadzone postaci modeli pokrywaj¡ si¦ z tymi wyznaczonymi w Pythonie?
    text2 = print('zad 3.3/3.4: Niestety nie. Macierze mają takie same wartoci ale w innych komórkach, natomiast den1 się zgadza ale num1 już nie (wartoci są w wierszach a powinny być w kolumnach)')
    return(plt1,plt2,plt3,plt4,text1,ss,tf,text2)

#zadanie2()
#zadanie3()

#parameters
m = 1
L = 0.5
d = 0.1
J = 1/3*m*L**2
#4.1 State-space model
A3 = np.matrix([[0,1],[0,-d/J]])
B3 = np.matrix([[0],[1/J]])
C3 = np.matrix([[1,0]])
D3 = 0

#4.2
sys2 = signal.StateSpace(A3,B3,C3,D3)
t4, y4 = signal.step(sys2)
plt.figure(7)
plt5 = plt.plot(t4, y4)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title( '4.2 manipulator')
plt.grid()
# Jaki jest charakter odpowiedzi skokowej obiektu?
text3 = print('4.2: Jest to charakter całkujący rzeczywisty')

#4.3 

#4.4
w, mag, phase = signal.bode(sys2)
plt.figure(8)
plt.title('4.4 Bode magnitude plot')
plt.semilogx(w, mag)
plt.grid()    # Bode magnitude plot
plt.figure(9)
plt.title('4.4 Bode phase plot')
plt.semilogx(w, phase)  # Bode phase plot
plt.grid()

