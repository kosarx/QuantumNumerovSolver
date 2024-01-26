import time
import numpy as np
import matplotlib.pyplot as plt

class NumerovSolverPIB:
                    #h_bar, m, L, quant_num, x_m
    def __init__(self, phys_attrib_list, Energy, Potential_function, npoints=1000):
        self.reset_parameters(phys_attrib_list, Energy, Potential_function, npoints)

    def set_phys_parameters(self, phys_attrib_list):
        self.h_bar = phys_attrib_list[0]
        self.mass = phys_attrib_list[1]
        self.L = phys_attrib_list[2]
        self.quant_num = phys_attrib_list[3]

    def set_x_grid(self, L_down, L_up, npoints):
        self.xlower = L_down #0 or -L/2
        self.xupper = L_up # 1 or L/2
        self.npoints = npoints
        self.x = np.linspace(self.xlower, self.xupper, self.npoints)
        self.delta = self.x[1] - self.x[0]

    def set_potential(self, x, Potential_function):
        self.Potential_function = Potential_function
        self.Potential = self.Potential_function(x)

    def reset_psi_wavefunctions(self, s):
        self.s = s
        self.psi_left = None
        self.psi_right = None
        self.prob_left = None
        self.prob_right = None

    def reset_parameters(self, phys_attrib_list, Energy, Potential_function, npoints=1000):

        self.set_phys_parameters(phys_attrib_list)

        self.Energy = Energy

        self.set_x_grid(0, self.L, npoints)
        if phys_attrib_list[4]:
            self.x_matching_point, self.x_matching_point_index = self.set_x_matching_point(phys_attrib_list[4])

        self.set_potential(self.x, Potential_function)

        s = 1e-5
        self.reset_psi_wavefunctions(s)

    def index_is_close_to(self, V, E):
        indices = np.where(np.isclose(V(self.x), E, atol=1e-2))
        return indices

    def set_x_matching_point(self, x_m, tol=1e-8):
        limit = 0
        x_matching_point_index = np.where(np.isclose(self.x, x_m, atol=tol))[0]
        while not x_matching_point_index.any():
            x_matching_point_index = np.where(np.isclose(self.x, x_m, atol=tol))[0]
            if x_matching_point_index.any():
                x_matching_point_index = x_matching_point_index[0]
                x_matching_point = self.x[x_matching_point_index]
                self.x_matching_point = x_matching_point
                return x_matching_point, x_matching_point_index
            else:
                limit +=1
                if limit >= 8:
                    raise ValueError(f"No matching point with tolerance {tol} for x_m: {x_m}")
                tol *=10
    
    def get_P(self, V_i):
        m = self.mass
        h_bar = self.h_bar
        E_g = self.Energy
        return -2 * m / (h_bar ** 2) * (V_i - E_g)

    def calculateNumerov(self, psi0, psi1, P_list):
        P0 = P_list[0]
        P1 = P_list[1]
        P2 = P_list[2]
        factor = 1 / (1 + (self.delta**2) / 12 * P2)
        psi2 = factor * (2 * psi1 * (1 - (5 * self.delta**2) / 12 * P1) - psi0 * (1 + (self.delta**2) / 12 * P0))
        return psi2

    def Numerov_left(self):
        self.psi_left = np.zeros(len(self.x))
        if self.quant_num % 2 == 0: #is even
            self.psi_left[0] = 0
            self.psi_left[1] = (self.get_P(self.Potential[0]) * self.delta**2 + 1) * 1 # derived from discretised SE, with symmetrical potential
        else: #is odd
            self.psi_left[0] = 0 #not necessary
            self.psi_left[1] = self.s
        for i in range(1, len(self.x) - 1):
            P_previous = self.get_P(self.Potential[i-1])
            P_current = self.get_P(self.Potential[i])
            P_next = self.get_P(self.Potential[i+1])
            self.psi_left[i+1] = self.calculateNumerov(self.psi_left[i-1], self.psi_left[i], [P_previous, P_current, P_next])
        self.prob_left = np.trapz(np.power(self.psi_left, 2), self.x)
        self.psi_left = self.psi_left / np.sqrt(self.prob_left)

    def Numerov_right(self):
        self.psi_right = np.zeros(len(self.x))
        if self.quant_num % 2 == 0: #is even
            self.psi_right[-1] = 0
            self.psi_right[-2] = (self.get_P(self.Potential[-1]) * self.delta**2 + 1) * -1 # derived from discretised SE, with symmetrical potential
        else: #is odd
            self.psi_right[-1] = 0 #not necessary
            self.psi_right[-2] = self.s
        for i in range(len(self.x) - 2, 0, -1):
            P_previous = self.get_P(self.Potential[i+1])
            P_current = self.get_P(self.Potential[i])
            P_next = self.get_P(self.Potential[i-1])
            self.psi_right[i-1] = self.calculateNumerov(self.psi_right[i+1], self.psi_right[i], [P_previous, P_current, P_next])
        self.prob_right = np.trapz(np.power(self.psi_right, 2), self.x)
        self.psi_right = self.psi_right / np.sqrt(self.prob_right)

    def run_solver(self):
        start = time.time()
        self.Numerov_left()
        self.Numerov_right()
        end = time.time()
        duration = end - start
        self.timetaken = duration

        self.stitched_psi = np.concatenate((self.psi_left[:self.x_matching_point_index], self.psi_right[self.x_matching_point_index:]))

        return duration

    def plot_show(self, matching=False):
        prob_left = np.trapz(np.power(self.psi_left, 2), self.x)
        prob_right = np.trapz(np.power(self.psi_right, 2), self.x)

        inner_product_value = np.trapz(np.conj(self.psi_left) * self.psi_right, self.x)
        print(f"Inner Product between psi_left and psi_right: {inner_product_value}")

        plt.figure()

        plt.xlim(-0.5, self.L + 0.5)
        plt.ylim(-1.5, 1.5)

        plt.title("Numerov solution to PIB, "
                + r"$\int\psi_\mathrm{left}$=" + "{:.2g}".format(prob_left)
                + r", $\int\psi_\mathrm{right}=$" + "{:.2g}".format(prob_right)
                + "\nTime taken: {:.2f} ms".format(self.timetaken*1000))

        if matching:
            plt.plot(self.x[:self.x_matching_point_index+1], self.psi_left[:self.x_matching_point_index+1], c='b', label=r'$\psi_\mathrm{left}$', linewidth= 3.0)
            plt.scatter(self.x[:self.x_matching_point_index+1], self.psi_left[:self.x_matching_point_index+1], c='b')

            plt.plot(self.x[:self.x_matching_point_index-1:-1], self.psi_right[:self.x_matching_point_index-1:-1], c='r', label=r'$\psi_\mathrm{right}$', linewidth= 3.0)
            plt.scatter(self.x[:self.x_matching_point_index:-1], self.psi_right[:self.x_matching_point_index:-1], c='r')

        else:
            plt.plot(self.x, self.psi_left, c='b', label=r'$\psi_\mathrm{left}$')
            plt.scatter(self.x, self.psi_left, c='b')

            plt.plot(self.x, self.psi_right, c='r', label=r'$\psi_\mathrm{right}$')
            plt.scatter(self.x, self.psi_right, c='r')


        x = self.x#np.linspace(0, 1, len(self.x))
        psi = np.sqrt(2/self.L) * np.sin(self.quant_num*np.pi * x/self.L) 

        inner_product_value = np.trapz(np.conj(psi) * self.psi_left, self.x)
        print(f"Inner Product between analytical psi and psi_left: {inner_product_value}")

        plt.plot(x, psi, c='k', label="Exact solution")

        # Plot the potential depth V0
        negative_x = np.linspace(-1, 0, 1000)
        yn = 2*np.ones(len(negative_x))

        plt.fill_between(negative_x, yn, color='gray', alpha=0.3, label='Classically Forbidden Area')
        plt.fill_between(negative_x, -yn, color='gray', alpha=0.3)

        positive_x = np.linspace(self.L, self.L+1, 1000)
        yp = 2*np.ones(len(positive_x))

        plt.fill_between(positive_x, yp, color='gray', alpha=0.3)
        plt.fill_between(positive_x, -yp, color='gray', alpha=0.3)

        plt.axhline(0, color='black', linestyle='-', linewidth=2)  # x-axis reference line
        plt.xlabel(r'$x$ (bohr)')
        plt.ylabel(r'$\psi(x)$')
        plt.grid(True)
        plt.legend()

        plt.show()
    
if __name__ == '__main__':
    h_bar = 1
    m = 1
    L = 1
    quant_num = 2
    x_matching_point = 0.5

    def potential(x):
        return np.zeros(len(x))

    solver = NumerovSolverPIB([h_bar, m, L, quant_num, x_matching_point], 4*np.pi**2/2, potential, 100)
    timetaken = solver.run_solver()
    solver.plot_show(True)