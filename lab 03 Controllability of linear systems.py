import numpy as np
import pprint
from matplotlib import pyplot as plt
from scipy.signal import lsim, lsim2
import control
import sympy as sym
from scipy.integrate import solve_ivp

pp = pprint.PrettyPrinter()

def ex1():
    t = np.linspace(0, 50, num=1000)
    # u = 5 * np.sin(t)
    u =  np.zeros_like(t) #np.ones(len(t))
    
    #1.3 wymuszania wykresy
    def plt_sys_response_lsim(ax, A,B,C,D, x0):
        tout, yout, _ = lsim((A,B,C,D), U=u, T=t, X0=x0)
        ax.plot(tout, yout)

    def plt_sys_response_lsim2(ax, A,B,C,D, x0):
        tout, yout, _ = lsim2((A,B,C,D), U=u, T=t, X0=x0)
        ax.plot(tout, yout)

    def plt_sys_response_lsim_and_lsim2(A,B,C,D, x0 = None):
        _, ax = plt.subplots(2, 1)
        ax[0].set_title("lsim")
        plt_sys_response_lsim(ax[0], A,B,C,D, x0)
        
        ax[1].set_title("lsim2")
        plt_sys_response_lsim2(ax[1], A,B,C,D, x0)

        plt.show()
    #1.2 
    def is_ctrb(A,B):
        K = control.ctrb(A, B)
        print(f'sterowalna = {np.linalg.matrix_rank(K) == np.size(A, 0)}\nmacierz S = ')
        pp.pprint(K)
        print()
        #macierz kalmana 

    def ex1_X(A,B,C,D):
        is_ctrb(A,B)
        plt_sys_response_lsim_and_lsim2(A,B,C,D)
    #2.1
    def can_ctrb(A):
        coeffs = np.poly(A)
        _, cols = np.shape(A)
        new_A = np.zeros((cols, cols))
        for i in range(cols - 1):
            new_A[i, i+1] = 1
        new_A[cols-1, :] = -coeffs[::-1][0:cols]
        new_B = np.zeros((cols, 1))
        new_B[-1, 0] = 1
        return new_A, new_B
    #tworzy macierze sterowalne dla zadania 2
    
    #2.1
    def ex2_X(A, B, C, D):
        As, Bs = can_ctrb(A)
        pp.pprint(As)
        K = control.ctrb(A,B)
        Ks = control.ctrb(As, Bs)
        Pinv = K @ np.linalg.inv(Ks)
        Cs = C @ Pinv
        Ds = D

        print("A* =")
        pp.pprint(As)

        print("B* =")
        pp.pprint(Bs)

        print("C* =")
        pp.pprint(Cs)

        print("D* =")
        pp.pprint(Ds)

        plt_sys_response_lsim_and_lsim2(As,Bs,Cs,Ds) # nie zmienia sie bo przeksztalcenie liniowe nie zmienia dynamiki
        # przebiegi tez sa takie same ale w nowym ukladzie wspolrzednych (przebieg sie nie zmienia ale osie tak)
        return As, Bs, Cs, Ds

    def ex3_X(A,B,C,D, poles):
        # 3.1
        # macierz z 2 zad jest 3x3 wiec tylko taka rozwazam
        s = sym.Symbol('s')
        _, cols = np.shape(A) #okrela rozmiary macierzy (A)
        K_sym = sym.Matrix([[-1 for i in range(cols)]])
        for ki in range(cols):
            K_sym[0, ki]= (sym.Symbol(f"k{ki}"))
        BK = sym.Matrix(B) @ K_sym
        sI = s * sym.Matrix.eye(cols)
        det = sym.Matrix.det(sI - (A - BK))
        print(det)
# "_," jest używany do przypisania zmiennej do wartości zwracanej przez funkcję "np.shape(A)" , ale ignorując pierwszy element zwracanej pary (w tym przypadku liczbę wierszy macierzy A).
        # 3.2
        K = control.place(A,B, poles)
        print(K)
        print(f"dla biegunow P = {poles}, nalezy wybrac K = {K}")

        #3.3
        A1 = A - B@K
        plt_sys_response_lsim_and_lsim2(A1,B,C,D, [1, 2, 3])
        # odpowiedz ma charakter obiektu oscylacyjnego
        # (ogolnie obiekt nie powinien byc oscylacyjny bo bieguny ustawilismy na ujemne [i bez Im])
        # ale wczesniej odpowiedz miala charakter jak dla inercyjnego 2 rzędu mimo, że mamy 3 bieguny.
        # Z tego wynika, że jeden biegun skrócił się z zerem, a to nie udało się przy
        # lokowaniu biegunów. To fajnie widać jak się zrobi rlocus() (w matlabie) dla układu otwartego
        # i zamkniętego
        # Poszczególne wartości wektora K wpływają na pozycje biegunów różnie i to jest zależne od macierzy A oraz B
        # (przez to, że z nich mamy to rownanie |sI-(A-BK)| = 0) 
        # no i biegunów które chcemy osiągnąć (wtedy mamy układ równań i tam można coś powiedzieć)
        # tu trzeba tez pamietac zeby wzmocnienie statyczne wyrownac bo moze nam nie dojsc do wartosci zadanej

    def ex1_2_1():
        # const.
        R = 2  # Ohm
        C = 1  # F

        # x' = Ax + Bu
        A = np.array([[-1/2/C/R, 0], [0, -1/2/C/R]])
        B = np.array([[1/2/C/R], [1/2/C/R]])
        C = np.array([[1, 0]])
        D = 0

        ex1_X(A,B,C,D)

    def ex1_2_2():
        # const.
        R = 1  # Ohm
        C = 1  # F

        # x' = Ax + Bu
        A = np.array([[-1/R/C, 0, 0], [0, -1/2/R/C, 0], [0, 0, -1/3/R/C]])
        B = np.array([[1/R/C], [1/2/R/C], [1/3/R/C]])
        C = np.array([[1, 0, 0]])
        D = 0

        ex1_X(A,B,C,D)
        As, Bs, Cs, Ds = ex2_X(A,B,C,D)
        ex3_X(As,Bs,Cs,Ds, [-1, -2, -5]) # charakter oscylacyjny, zmiana biegunow wplywa na wsp. tlumienia i pulsacje.

    def ex1_2_3():
        # const.
        C = 1 # L
        R = 1 # Ohm

        # x' = Ax + Bu
        A = np.array([[1/C/R]])
        B = np.array([[0]])
        C = 1
        D = 0

        ex1_X(A,B,C,D)

    def ex1_2_4():
        # const.
        R1 = 2  # Ohm,
        R2 = 1  # Ohm,
        L1 = 0.5  # H,
        L2 = 1  # H,
        C = 2  # F

        # x' = Ax + Bu
        A = np.array([[-R1/L1, 0, -1/L1], [0, 0, 1/L1], [1/C, -1/C, -1/C/R2]])
        B = np.array([[1/L1], [0], [0]])
        C = np.array([[1, 0, 0]])
        D = 0

        ex1_X(A,B,C,D)
        ex2_X(A,B,C,D)
    
    # ============ tu trzeba wybrac ktory uklad teraz analizujemy i po kolei wszystkie zadania do podpunktu leca ============= #
    ex1_2_1()
    ex1_2_2()
    ex1_2_3()
    ex1_2_4()




if __name__ == "__main__":
    ex1()
