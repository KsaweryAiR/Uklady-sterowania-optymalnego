# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:55:01 2022

@author: ksawe
"""
import numpy as np
import matplotlib .pyplot as plt
from scipy.optimize import minimize, LinearConstraint, minimize_scalar
from scipy.integrate import odeint
from numpy import linspace, eye
def przyklad():  
    #ograniczenia
    plt.figure(0)
    point1 = -5
    point2 = 5
    scal = 0.08
    interval = np.arange(point1 ,point2+scal ,scal)
    b = np.arange(point1 ,point2+scal ,scal)
    wait = 0
    for bs in b:
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        ymain = []
        y_plot =[]
        x_plot =[]
        i=0
        for x in interval:
           y1.append(-1/2*x+2)  
           y2.append(1/2*x+4) 
           y3.append(4*x -2)
           y4.append(-2*x -2)
           ymain.append(2/3*x+bs) 
           if (ymain[i] < y1[i]) and (ymain[i] <= y2[i]) and (ymain[i] >= y3[i]) and (ymain[i] >= y4[i]):
               result = 2*x-3*ymain[i]    
               y_plot.append(ymain[i])
               x_plot.append(x)
               if wait == 0:
                   resultMAX = result
                   y_res = ymain[i]
                   x_res = x 
                   wait = 1
               if result > resultMAX:
                   resultMAX = result
                   y_res = ymain[i]
                   x_res = x  
           i=i+1   
        #plt.plot(interval,ymain,'--', color ='grey') #full space
        plt.plot(x_plot,y_plot, color ='grey') 
      
    #correct
    resultMAX = np.around(resultMAX)   
    x_res = np.around(x_res)
    y_res = np.around(y_res)
    print('Max = ',resultMAX, ' dla x =',x_res, ' y = ', y_res)
    #graph
    plt.xlabel('arguments')
    plt.ylabel('value')
    plt.grid('both')
    plt.plot(interval,y1,linewidth=3)
    plt.plot(interval,y2,linewidth=3)
    plt.plot(interval,y3,linewidth=3)
    plt.plot(interval,y4,linewidth=3)
    plt.plot(x_res , y_res, '.', markersize = 15, color = 'black')
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
def zadanie2():   
    #ograniczenia
    plt.figure(1)
    point1 = -5
    point2 = 5
    scal = 0.08
    interval = np.arange(point1 ,point2+scal ,scal)
    b = np.arange(point1 ,point2+scal ,scal)
    wait = 0
    for bs in b:
        y1 = []
        y2 = []
        y3 = []
        ymain = []
        y_plot =[]
        x_plot =[]
        i=0
        for x in interval:
           y1.append(2*x-4)  
           y2.append(-x+3) 
           y3.append(-4*x -2)
           ymain.append(-x+bs) 
           if (ymain[i] >= y1[i]) and (ymain[i] > y2[i]) and (ymain[i] >= y3[i]):
               result = 2*x-3*ymain[i]    
               y_plot.append(ymain[i])
               x_plot.append(x)
               if wait == 0:
                   resultMAX = result
                   y_res = ymain[i]
                   x_res = x 
                   wait = 1
               if result > resultMAX:
                   resultMAX = result
                   y_res = ymain[i]
                   x_res = x  
           i=i+1   
        #plt.plot(interval,ymain,'--', color ='grey') #full space
        plt.plot(x_plot,y_plot, color ='grey') 
      
    #correct
    correct = 2
    resultMAX = np.around(resultMAX, decimals = correct)   
    x_res = np.around(x_res, decimals = correct)
    y_res = np.around(y_res, decimals = correct)
    print('f(x)_Max = ',resultMAX, ' dla x =',x_res, ' y = ', y_res)
    #graph
    plt.xlabel('arguments')
    plt.ylabel('value')
    plt.grid('both')
    plt.plot(interval,y1,linewidth=3)
    plt.plot(interval,y2,linewidth=3)
    plt.plot(interval,y3,linewidth=3)
    plt.plot(x_res , y_res, '.', markersize = 15, color = 'black')
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    
    #2.3
    def f2(xy):
        return xy[1]  # bez minusa bo uzywamy minimize [-y -> max = y -> min]

    # start
    xy_start = [0, 0]

    # OGRANICZENIA
    bnds = [
        {
            'type': 'ineq',
            'fun': lambda xy: (xy[0] + xy[1] - 3)
        },
        {
            'type': 'ineq',
            'fun': lambda xy: (-2*xy[0] + xy[1] + 4)
        },
        {
            'type': 'ineq',
            'fun': lambda xy: (xy[1] + 4*xy[0] + 2)
        }
    ]

    result = minimize(f2, xy_start, options={"disp": True}, constraints=bnds)
    if result.success:
        print(f"rozw={result.x}")
    else:
        print("nie udało się")
def zadanie3():
    def f3(x):
        return x**4 - 4*x**3 - 2*x**2 + 12*x + 9
    # init. cond.
    x0 = [0]
    # bounds
    bds = [(0, None)]
    # method
    mth = 'Powell'
    result = minimize(f3, x0, bounds=bds,  method=mth, options={'disp': True})
    if result.success:
        print(f"rozw={result.x}")
    else:
        print("nie udało się")
def zadanie4():
    t_start = 0
    t_end = 1
    x_t_start = 1
    x_t_end = 3
    
    def model(_, t, a0, a1, a2, a3):
        xt = a0 + a1*t + a2*t**2 + a3*t**3
        dxt_dt = a1 + 2*a2*t + 3*a3*t**2
        dJ_dt = 24*xt + 2*dxt_dt**2 - 4*t
        return dJ_dt

    def problem_dyn(a):
        t = linspace(t_start, t_end)
        int_0t_J = odeint(model, [0], t, args=a)
        return int_0t_J[-1][0]
        
    def min_a(a):
        return problem_dyn((a[0], a[1], a[2], a[3]))
    
    a_cstr = [
        {
            "type": "eq",
            "fun": lambda a: a[0] + a[1]*t_start + a[2]*t_start**2 + a[3]*t_start**3 - x_t_start # = 0
        },
        {
            "type": "eq",
            "fun": lambda a: a[0] + a[1]*t_end + a[2]*t_end**2 + a[3]*t_end**3 - x_t_end # = 0
        }
    ]

    a0 = [0, 0, 0, 0]
    res = minimize(min_a, a0, constraints=a_cstr, options={"disp":True})
    if res.success:
        print(f"rozw={res.x}")
    else:
        print("nie tym razem")
#zadanie2()
#przyklad()
#zadanie3()
#zadanie4()