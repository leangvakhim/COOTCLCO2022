import numpy as np

class benchmark:
    def get_function(func_num):
        if func_num == 1: return benchmark.F1, -100, 100, 30
        elif func_num == 2: return benchmark.F2, -10, 10, 30
        elif func_num == 3: return benchmark.F3, -100, 100, 30
        elif func_num == 4: return benchmark.F4, -100, 100, 30
        elif func_num == 5: return benchmark.F5, -30, 30, 30
        elif func_num == 6: return benchmark.F6, -100, 100, 30
        elif func_num == 7: return benchmark.F7, -1.28, 1.28, 30

        # Multimodal Functions (F8-F13)
        elif func_num == 8: return benchmark.F8, -500, 500, 30
        elif func_num == 9: return benchmark.F9, -5.12, 5.12, 30
        elif func_num == 10: return benchmark.F10, -32, 32, 30
        elif func_num == 11: return benchmark.F11, -600, 600, 30
        elif func_num == 12: return benchmark.F12, -50, 50, 30
        elif func_num == 13: return benchmark.F13, -50, 50, 30

        # Fixed-dimension Multimodal Functions (F14-F23)
        elif func_num == 14: return benchmark.F14, -65, 65, 2
        elif func_num == 15: return benchmark.F15, -5, 5, 4
        elif func_num == 16: return benchmark.F16, -5, 5, 2
        elif func_num == 17: return benchmark.F17, -5, 5, 2
        elif func_num == 18: return benchmark.F18, -2, 2, 2
        elif func_num == 19: return benchmark.F19, 0, 1, 3  # Range usually [0,1] or [1,3] for Hartman
        elif func_num == 20: return benchmark.F20, 0, 1, 6
        elif func_num == 21: return benchmark.F21, 0, 10, 4
        elif func_num == 22: return benchmark.F22, 0, 10, 4
        elif func_num == 23: return benchmark.F23, 0, 10, 4
        else: return None, 0, 0, 0

    @staticmethod
    def F1(x):
        # Sphere
        return np.sum(x**2)

    @staticmethod
    def F2(x):
        # Schwefel 2.22
        return np.sum(np.abs(x)) + np.prod(np.abs(x))

    @staticmethod
    def F3(x):
        # Schwefel 1.2
        dim = len(x)
        o = 0
        for i in range(dim):
            o += np.sum(x[:i+1])**2
        return o

    @staticmethod
    def F4(x):
        # Schwefel 2.21
        return np.max(np.abs(x))

    @staticmethod
    def F5(x):
        # Rosenbrock
        dim = len(x)
        o = 0
        for i in range(dim - 1):
            o += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        return o

    @staticmethod
    def F6(x):
        # Step
        return np.sum((x + 0.5)**2)

    @staticmethod
    def F7(x):
        # Quartic with noise
        dim = len(x)
        return np.sum(np.arange(1, dim + 1) * (x**4)) + np.random.rand()

    # --- Multimodal Functions ---
    @staticmethod
    def F8(x):
        # Schwefel
        return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

    @staticmethod
    def F9(x):
        # Rastrigin
        dim = len(x)
        return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

    @staticmethod
    def F10(x):
        # Ackley
        dim = len(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - \
               np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.e

    @staticmethod
    def F11(x):
        # Griewank
        dim = len(x)
        prod_val = 1
        for i in range(dim):
            prod_val *= np.cos(x[i] / np.sqrt(i + 1))
        return np.sum(x**2) / 4000 - prod_val + 1

    @staticmethod
    def F12(x):
        # Penalized 1
        dim = len(x)
        def u(xi, a, k, m):
            if xi > a: return k * (xi - a)**m
            if xi < -a: return k * (-xi - a)**m
            return 0

        y = 1 + (x + 1) / 4
        term1 = 10 * np.sin(np.pi * y[0])**2
        term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
        term3 = (y[-1] - 1)**2

        sum_u = np.sum([u(xi, 10, 100, 4) for xi in x])

        return (np.pi / dim) * (term1 + term2 + term3) + sum_u

    @staticmethod
    def F13(x):
        # Penalized 2
        dim = len(x)
        def u(xi, a, k, m):
            if xi > a: return k * (xi - a)**m
            if xi < -a: return k * (-xi - a)**m
            return 0

        term1 = np.sin(3 * np.pi * x[0])**2
        term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:] + 1)**2))
        term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)

        sum_u = np.sum([u(xi, 5, 100, 4) for xi in x])

        return 0.1 * (term1 + term2 + term3) + sum_u

    # --- Fixed-Dimension Multimodal Functions ---
    @staticmethod
    def F14(x):
        # Shekel's Foxholes
        aS = np.array([[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],
                       [-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]])
        s = 0
        for j in range(25):
            s += 1 / (j + 1 + (x[0] - aS[0, j])**6 + (x[1] - aS[1, j])**6)
        return (1.0 / 500 + s)**(-1)

    @staticmethod
    def F15(x):
        # Kowalik
        aK = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
        bK = [1/0.25, 1/0.5, 1/1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16]
        s = 0
        for i in range(11):
            term = aK[i] - (x[0] * (bK[i]**2 + x[1] * bK[i])) / (bK[i]**2 + x[2] * bK[i] + x[3])
            s += term**2
        return s

    @staticmethod
    def F16(x):
        # Six Hump Camel
        return 4*x[0]**2 - 2.1*x[0]**4 + (x[0]**6)/3 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4

    @staticmethod
    def F17(x):
        # Branin
        return (x[1] - (5.1 / (4 * np.pi**2)) * x[0]**2 + (5 / np.pi) * x[0] - 6)**2 + \
               10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

    @staticmethod
    def F18(x):
        # Goldstein-Price
        term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
        return term1 * term2

    @staticmethod
    def F19(x):
        # Hartman 3
        aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        cH = np.array([1, 1.2, 3, 3.2])
        pH = np.array([[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470],
                       [0.1091, 0.8732, 0.5547], [0.0381, 0.5743, 0.8828]])
        s = 0
        for i in range(4):
            exponent = -np.sum(aH[i] * (x - pH[i])**2)
            s += cH[i] * np.exp(exponent)
        return -s

    @staticmethod
    def F20(x):
        # Hartman 6
        aH = np.array([
            [10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]
        ])
        cH = np.array([1, 1.2, 3, 3.2])
        pH = np.array([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
        ])
        s = 0
        for i in range(4):
            exponent = -np.sum(aH[i] * (x - pH[i])**2)
            s += cH[i] * np.exp(exponent)
        return -s

    @staticmethod
    def F21(x):
        # Shekel 5
        return benchmark.shekel(x, 5)

    @staticmethod
    def F22(x):
        # Shekel 7
        return benchmark.shekel(x, 7)

    @staticmethod
    def F23(x):
        # Shekel 10
        return benchmark.shekel(x, 10)

    @staticmethod
    def shekel(x, m):
        aS = np.array([
            [4,4,4,4], [1,1,1,1], [8,8,8,8], [6,6,6,6], [3,7,3,7],
            [2,9,2,9], [5,5,3,3], [8,1,8,1], [6,2,6,2], [7,3.6,7,3.6]
        ])
        cS = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

        s = 0
        for i in range(m):
            s += 1 / (np.dot((x - aS[i]), (x - aS[i])) + cS[i])
        return -s