import numpy as np
import matplotlib.pyplot as plt

from NumerovMethodPIBClass import NumerovSolverPIB

class NumerovSolverDWP(NumerovSolverPIB):
    def __init__(self, phys_attrib_list, Energy, Potential_function, npoints=1000):
        self.reset_parameters(phys_attrib_list, Energy, Potential_function, npoints)
    
    def set_phys_parameters(self, phys_attrib_list):
        super().set_phys_parameters(phys_attrib_list)
        self.V0 = phys_attrib_list[5]
        self.V0_barrier = phys_attrib_list[6]
        self.well1_start = phys_attrib_list[7]
        self.well1_end = phys_attrib_list[8]
        self.well2_start = phys_attrib_list[9]
        self.well2_end = phys_attrib_list[10]
    
    def set_x_grid(self, L_down, L_up, npoints):
        return super().set_x_grid(L_down, L_up, npoints)
    
    def set_potential(self, x, Potential_function):
        self.Potential_function = Potential_function
        self.Potential = self.Potential_function(x, self.L, self.well1_start, self.well1_end, self.well2_start, self.well2_end, self.V0, self.V0_barrier)
    
    def reset_parameters(self, phys_attrib_list, Energy, Potential_function, npoints=1000):
        self.set_phys_parameters(phys_attrib_list)

        self.Energy = Energy

        self.set_x_grid(0, self.L, npoints)
        if phys_attrib_list[4]:
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

        plt.title("Numerov solution to FPW, "
                + r"$\int\psi_\mathrm{left}$=" + "{:.2g}".format(prob_left)
                + r", $\int\psi_\mathrm{right}=$" + "{:.2g}".format(prob_right)
                + "\nTime taken: {:.2f} ms".format(self.timetaken*1000))
        
        V0_over_E = self.Potential/self.Energy

        if matching:
            plt.plot(self.x[:self.x_matching_point_index+1], self.psi_left[:self.x_matching_point_index+1] + 1, c='b', label=r'$\psi_\mathrm{left}$', linewidth= 3.0)
            #plt.scatter(self.x[:self.x_matching_point_index+1], self.psi_left[:self.x_matching_point_index+1], c='b')

            plt.plot(self.x[:self.x_matching_point_index-1:-1], self.psi_right[:self.x_matching_point_index-1:-1] + 1, c='r', label=r'$\psi_\mathrm{right}$', linewidth= 3.0)
            #plt.scatter(self.x[:self.x_matching_point_index:-1], self.psi_right[:self.x_matching_point_index:-1], c='r')

        else:
            plt.plot(self.x, self.psi_left + 1, c='b', label=r'$\psi_\mathrm{left}$')
            #plt.scatter(self.x, self.psi_left, c='b')

            plt.plot(self.x, self.psi_right + 1, c='r', label=r'$\psi_\mathrm{right}$')
            #plt.scatter(self.x, self.psi_right, c='r')

        # Plot the potential depth V0
        plt.plot(self.x, V0_over_E, color = 'g', linestyle='-', label=r'$V_0/E$', linewidth=2)

        plt.fill_between(self.x, V0_over_E, color='gray', alpha=0.3, label='Classically Forbidden Area')

        plt.axhline(0, color='black', linestyle='-', linewidth=2)  # x-axis reference line
        plt.xlabel(r'$x$ (bohr)')
        plt.ylabel(r'$\psi(x)$')
        plt.grid(True)
        plt.legend()

        plt.show()

