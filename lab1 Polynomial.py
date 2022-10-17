# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib .pyplot as plt

def polynomial(coeffcients, x1, x2, precision):
    interval = np.arange(x1 ,x2+precision ,precision)
    y = []
    for j in range(0,interval.size):
        s = coeffcients.size -1
        result = 0
        for i in range(0,coeffcients.size):    
            c = coeffcients[0][i]*interval[j]**s
            result = c + result
            s = s-1
        #graph value
        y.append(result)
        x = interval
        #check max and min
        if j == 0:
            checkmax = result
            checkmin = result
            pointmax = interval[j]
            pointmin = interval[j]
        else:
            if result > checkmax:
                checkmax = result
                pointmax = interval[j]
            if result < checkmin:
                checkmin = result
                pointmin = interval[j]

    print('extremum = ' ,checkmax,'w punkcie' ,pointmax)
    print('minimum = ',checkmin,' w punkcie', pointmin)
    
    #graph
    plt.xlabel('arguments')
    plt.ylabel('value')
    plt.title('Polynomial')
    plt.grid('both')
   # plt.ylim(checkmin,checkmax)
    return(plt.plot(x,y))

    
   
x1 = -46
x2 = 14
precision= 1
array = np.array([[1,1,-129,171,1620]])

hi = polynomial(array, x1, x2 ,precision)    
