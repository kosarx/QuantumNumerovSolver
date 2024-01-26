import numpy as np
import matplotlib.pyplot as plt

from NumerovMethodPIBClass import NumerovSolverPIB

class NumerovSolverQHO(NumerovSolverPIB):
    def __init__(self, phys_attrib_list, Energy, Potential_function, npoints=1000):
        self.reset_parameters(phys_attrib_list, Energy, Potential_function, npoints)

    def set_phys_parameters(self, phys_attrib_list):
        self.h_bar = phys_attrib_list[0]
        self.mass = phys_attrib_list[1]
        self.omega = phys_attrib_list[2]
        self.quant_num = phys_attrib_list[3]

    def set_x_grid(self, x_max, npoints):
        self.xlower = -x_max
        self.xupper = x_max
        self.npoints = npoints
        self.x = np.linspace(self.xlower, self.xupper, self.npoints)
        self.delta = self.x[1] - self.x[0]
    
    def set_potential(self, x, Potential_function):
        self.Potential_function = Potential_function
        self.Potential = self.Potential_function(x, self.mass, self.omega)

    def reset_parameters(self, phys_attrib_list, Energy, Potential_function, npoints=1000):
        self.set_phys_parameters(phys_attrib_list)

        self.Energy = Energy

        self.x_max = 3 if self.quant_num<=3 else 2+np.sqrt(1.8*self.quant_num+1)-1/(self.quant_num+1) # scale x range as quant_num expands
        self.set_x_grid(self.x_max, npoints)
        if phys_attrib_list[4] is not None:
            self.x_matching_point, self.x_matching_point_index = self.set_x_matching_point(phys_attrib_list[4])

        self.set_potential(self.x, Potential_function)

        s = 1e-5
        self.reset_psi_wavefunctions(s)
    
    def plot_show(self, matching=False):
        prob_left = np.trapz(np.power(self.psi_left, 2), self.x)
        prob_right = np.trapz(np.power(self.psi_right, 2), self.x)

        inner_product_value = np.trapz(np.conj(self.psi_left) * self.psi_right, self.x)
        print(f"Inner Product between psi_left and psi_right: {inner_product_value}")

        plt.figure()

        plt.title("Numerov solution to QHO, "
                + r"$\int\psi_\mathrm{left}$=" + "{:.2g}".format(prob_left)
                + r", $\int\psi_\mathrm{right}=$" + "{:.2g}".format(prob_right)
                + "\nTime taken: {:.2f} ms".format(self.timetaken*1000))

        if matching:
            plt.plot(self.x[:self.x_matching_point_index+1], self.psi_left[:self.x_matching_point_index+1] + self.Energy, c='b', label=r'$\psi_\mathrm{left}$', linewidth= 1.5)
            #plt.scatter(self.x[:self.x_matching_point_index+1], self.psi_left[:self.x_matching_point_index+1], c='b')

            plt.plot(self.x[:self.x_matching_point_index-1:-1], self.psi_right[:self.x_matching_point_index-1:-1]+ self.Energy, c='r', label=r'$\psi_\mathrm{right}$', linewidth= 1.5)
            #plt.scatter(self.x[:self.x_matching_point_index:-1], self.psi_right[:self.x_matching_point_index:-1], c='r')

        else:
            plt.plot(self.x, self.psi_left + self.Energy, c='b', label=r'$\psi_\mathrm{left}$', linewidth= 1.5)
            #plt.scatter(self.x, self.psi_left, c='b')

            plt.plot(self.x, self.psi_right + self.Energy, c='r', label=r'$\psi_\mathrm{right}$', linewidth= 1.5)
            #plt.scatter(self.x, self.psi_right, c='r')

        # Plot the potential depth V0
        plt.plot(self.x, self.Potential, color = 'g', linestyle='-', label=r'$V_0$', linewidth=2)

        # plt.fill_between(self.x, self.Potential/self.Energy, color='gray', alpha=0.3, label='Shaded Area')

        plt.axhline(0, color='black', linestyle='-', linewidth=2)  # x-axis reference line
        plt.xlabel(r'$x$ (bohr)')
        plt.ylabel(r'$\psi(x)$')
        plt.grid(True)
        plt.legend()

        plt.show()
