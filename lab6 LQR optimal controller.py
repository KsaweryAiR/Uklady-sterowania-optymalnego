from numpy import array, linspace, asarray, eye, append
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are, inv
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot, figure, show, legend
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
def main():
    # parametry modelu
    params = dict(R=0.5, L=0.2, C=0.5)

    # model
    A = array([[0, 1], [-1/params['L']/params['C'], -params['R']/params['L']]])
    B = array([[0],[1/params['L']]])

    def zadanie2():
        # odeint fun (2.2)
        t = np.arange(0,5,0.01)
        Q = np.matrix([[1,0],[0,1]])
        R = np.matrix([1])
        P = linalg.solve_continuous_are(A, B, Q, R)   
        K = R**(-1)@(B.T)@P
        #2.1-5
        def model(x, t):
            x1 = np.array([x]).T
            u = -K@x1
            dx = A@x1 + B@u
            dJdt = x1.T @ Q @ x +u.T @ R @ u #wskaźnik
            dx1 = np.ndarray.tolist(dx.T[0])
            return dx1[0]
        y = odeint(model,[1,1],t)
        plt.figure(1)
        plt.plot(t,y)

        # odeint fun (2.6)
        def model_odeint_2_6(x, t, K, R):
            x = (asarray(x)[0:-1]).reshape((2,1))
            u = -K@x
            dxdt = A@x+B@u
            dJdt = x.T@Q@x+u.T@R@u
            return append(dxdt, dJdt)

        K = inv(R)*B.T@P # R^-1 * B^T * P
        print(f'K={K}')

           

        # 2.6
        x0 = [0, 1, 0]
        x_2_6 = odeint(model_odeint_2_6, x0, t, args=(K,R))
        J = x_2_6[-1, 2]
        print(f'J={J}')
        figure()
        plot(t, x_2_6[:, 0], label='x1')
        plot(t, x_2_6[:, 2], label='J')
        legend()
        show()

    def zadanie3():

        # rikati
        def ric_odeint(P, t, Q, R):
            P = asarray(P).reshape((2,2))
            return (-(P@A-P@B@inv(R)@B.T+A.T@P+Q)).flatten()

        # parametry regulatora
        Q = eye(2,2)
        R = array([[1]])
        
        # sym
        t1 = 5 # s 
        t = linspace(t1, 0, num=500)
        Pt1 = [1, 0, 0, 1] # S
        P = odeint(ric_odeint, Pt1, t, args=(Q,R))
            
        # plotowanie macierzy P (3.2) #Wykreślić przebieg elementów macierzy P(t) w czasie
        figure()
        plot(t, P[:,0], label="P00")
        plot(t, P[:,1], label="P01")
        plot(t, P[:,2], label="P10")
        plot(t, P[:,3], label="P11")
        legend()
        show()

        # funkcja macierzy P (interpolacja)
        Pfun = array([interp1d(t, P[:, i], fill_value='extrapolate') for i in range(P.shape[1])]).reshape((2,2))
        
        # odeint fun 3.4 
        def model_odeint_3_4(x, t, Pfun, R):
            x = asarray(x).reshape((2, 1)) # jest używana do konwertowania zmiennej x na tablicę numpy i zmieniania jej kształtu na (2, 1).
            P = array([[Pfun[i, j](t) for j in range(Pfun.shape[0])] for i in range(Pfun.shape[1])])
            K = inv(R)*B.T@P
            ud = 1 # skok
            u = -K@x + ud
            return (A@x+B*u).flatten()

        x0 = [10, 0]
        t = linspace(0, 15, num=200)
        x = odeint(model_odeint_3_4, x0, t, args=(Pfun, R))

        #wykres odp skokowej (3.6)
        figure()
        plot(t, x[:, 0], label="x1")
        plot(t, x[:, 1], label="x2")
        legend()
        show()

        # 3.7
        def model_odeint_3_7(x, t, Pfun, R):
            x = (asarray(x)[0:-1]).reshape((2,1))

            P = array([[Pfun[i, j](t) for j in range(Pfun.shape[0])] for i in range(Pfun.shape[1])])
            K = inv(R)*B.T@P
    
            ud = 1
            u = -K@x + ud
            
            dxdt = A@x+B@u
            dJdt = x.T@Q@x+u.T@R@u
            return append(dxdt.flatten(), dJdt)

        # obliczanie wskaznika jakosci
        x0 = append(x0, 0)
        J = odeint(model_odeint_3_7, x0, t, args=(Pfun, R))[:, -1][-1]
        print(f"J={J}")
        

    def zadanie4():
        def model_odeint_4_1(x, t, K):
            x = asarray(x).reshape((2,1))
            qd = 9 #pkt do stabilizacji
            xd = array([[qd], [0]])            
            e = xd - x
            u = -(-K@e) +1/params['C']*qd #- A@xd
            dxdt = A@x + B*u
            return dxdt.flatten()

        # wyznaczanie K, R
        Q = eye(2,2)
        R = array([[1]])
        P = solve_continuous_are(A,B,Q,R)
        K = inv(R)*B.T@P # R^-1 * B^T * P
        print(f'K={K}')

        # 4.5
        t = linspace(0, 15, num=200)
        x0 = [0, 0]
        x_4_5 = odeint(model_odeint_4_1, x0, t, args=(K,))

        figure()
        plot(t, x_4_5[:, 0], label='x1')
        plot(t, x_4_5[:, 1], label='x2')
        legend()
        show()
    
    # wykresy sa zakomentowane
    zadanie2()
    #zadanie3()
    #zadanie4()

if __name__ == '__main__':
    main()