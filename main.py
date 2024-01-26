#standard libraries
import numpy as np
import scipy.optimize as sp_op

#program specific
from NumerovMethodPIBClass import NumerovSolverPIB # particle in a box
from NumerovMethodFPPClass import NumerovSolverFPP # finite potential well
from NumerovMethodQHOClass import NumerovSolverQHO # quantum harmonic oscillator
from NumerovMethodPTWClass import NumerovSolverPTW # poschl-teller well
from NumerovMethodDWPClass import NumerovSolverDWP # double potential well

class MainClass:
    def __init__(self, phys_attrib_list=None):
        # Poschl-Teller Well
        self.ptw_well_width = 1
        self.ptw_V0 = 5

        #Double Potential Well
        self.dwp_V0 = 20
        self.dwp_V0_barrier = 20
    
    def get_value_from_user(self, what, problem):
        resp = None
        while not resp:
            try:
                resp = int(input(f"{what} for {problem}? -> "))
                if "quantum number" in what and resp <1:
                    continue
                if resp == 0:
                    continue
            except:
                continue
        return resp

    def pib_problem_parameters(self):
        '''Parameters for the Particle In a Box Problem, or the Infinite Potential Square Well'''
        h_bar = 1
        m = 1
        L = 1
        quant_num = self.get_value_from_user("Quantum number", "Particle in a Box Problem")
        x_matching_point = L-0.02 # for the PIB problem, we can't pick the classical turning point V(x) - E for x_m
        parameter_list = [h_bar, m, L, quant_num, x_matching_point]
        return parameter_list
    
    def fpw_problem_parameters(self):
        '''Parameters for the Finite Potential Square Well'''
        h_bar = 1
        m = 1
        L = 2
        quant_num = self.get_value_from_user("Quantum number", "Finite Potential Well")
        V0 = 36
        well_width = 1
        x_matching_point = None # set later
        parameter_list = [h_bar, m, L, quant_num, x_matching_point, V0, well_width]
        return parameter_list
    
    def qho_problem_parameters(self):
        '''Parameters for the Quantum Harmonic Oscillator'''
        h_bar = 1
        m = 1
        omega = 1
        quant_num = self.get_value_from_user("Quantum number", "Quantum Harmonic Oscillator")
        x_matching_point = 0.0
        parameter_list = [h_bar, m, omega, quant_num, x_matching_point]
        return parameter_list

    def ptw_problem_parameters(self):
        '''Parameters for the Poschl-Teller Well'''
        h_bar = 1
        m = 1
        well_width = self.ptw_well_width # parameter 'a' in potential
        quant_num = self.get_value_from_user("Quantum number", "Poschl-Teller Well")
        V0 = self.ptw_V0 # part of well depth
        x_matching_point = 0.0 # set later
        parameter_list = [h_bar, m, well_width, quant_num, x_matching_point, V0]
        return parameter_list
    
    def dwp_problem_parameters(self):
        '''Parameters for the Double Well'''
        h_bar = 1
        m = 1
        L= 5
        quant_num = self.get_value_from_user("Quantum number", "Double Well")
        V0 = self.dwp_V0
        well1_start=1.0
        well1_end=2.0
        well2_start=3
        well2_end=4
        V0_barrier= self.dwp_V0_barrier
        x_matching_point = L/2
        parameter_list = [h_bar, m, L, quant_num, x_matching_point, V0, V0_barrier, well1_start, well1_end, well2_start, well2_end]
        return parameter_list

    def plot_show(self, probObj, matching):
        probObj.plot_show(matching)

    def get_Y(self, solver, Vi, psi_i):
        P_i = solver.get_P(Vi)
        dx = solver.delta
        Y_i = (1 + 1/12*dx**2*P_i)* psi_i
        return Y_i
    
    def cooley_energy_correction(self, solver, m, first_psi, potential, E_0, E_guess):
        Y_previous = self.get_Y(solver, potential[m-1], first_psi[m-1])
        Y_current = self.get_Y(solver, potential[m], first_psi[m])
        Y_next = self.get_Y(solver, potential[m+1], first_psi[m+1])

        first_term = np.conj(first_psi[m])/np.sum(np.abs(first_psi)**2)
        second_term = solver.h_bar**2/(2*solver.mass) * (Y_next -2*Y_current + Y_previous)/solver.delta**2 + (potential[m] - E_0)*first_psi[m]

        delta_E = first_term * second_term
        E_new_guess = E_guess + delta_E
        return E_new_guess

    def count_nodes(self, psi):
        # Find indices where the sign of psi changes
        sign_changes = np.where(np.diff(np.sign(psi)))[0]

        # Count the number of crossings
        num_crossings = len(sign_changes)
                                #indices
        return num_crossings, sign_changes
    
    def numerov_cooley_wavefunc_and_energy(self, solver, iteration, param_list, E_first_guess, Energy_guess, potential_func, Npoints):
        # Use Numerov Method to solve
        solver.reset_parameters(param_list, Energy_guess, potential_func, Npoints)
        solver.run_solver()
        psi = solver.psi_left
        # Use Cooley Energy Correction formula for a better approximation
        Energy_guess = self.cooley_energy_correction(solver, solver.x_matching_point_index, \
                                                    psi, solver.Potential, E_first_guess, Energy_guess)
        print(f"{iteration}. Energy guess:", Energy_guess)
        return psi

    def solve_particle_in_a_box(self, solver, solver_secondary, param_list, E_first_guess, Emin, Emax, Energy_guess, potential_func, Npoints, tol=1e-2):
        quant_num  = param_list[3]
        for i in range(1,Npoints):
            Energy_guess = 1/2 * (Emax + Emin)
            # # Use Numerov Method to solve
            psi = self.numerov_cooley_wavefunc_and_energy(solver, i, param_list, E_first_guess, Energy_guess, potential_func, Npoints)
            if i == 0:
                first_psi = solver.psi_left
                psi = first_psi

            # count nodes of wavefunction
            n_prime, _ = self.count_nodes(psi)
            if n_prime < quant_num:
                Emin = Energy_guess
                continue
            elif n_prime > quant_num:
                Emax = Energy_guess
                continue
            elif n_prime == quant_num:
                pass

            # Bisect the energy eigenvalue bounds to 
            # better approximate the boundary condition
            
            #get wavefunction for Emin
            solver_secondary.reset_parameters(param_list, Emin, potential_func, Npoints)
            solver_secondary.run_solver()
            psi_Emin = solver.psi_left
            if psi_Emin[-1] * psi[-1] > 0:
                Emin = Energy_guess
            elif psi_Emin[-1] * psi[-1] < 0:
                Emax = Energy_guess
            
            # Check energy eigenvalue convergance
            if np.abs(Emax - Emin) <= tol:
                break

    def particle_in_a_box(self, param_list, potential_func, tol=1e-5):
        Npoints = self.get_value_from_user("Number of Points N", "Particle In a Box Problem")
        h_bar = param_list[0]
        m = param_list[1]
        L = param_list[2]
        quant_num = param_list[3]
        x_matching_point = param_list[4]
        # To numerically solve for the nth eigenstate and the eigenvalue of the
        # time dependent Schrodinger equation:
        # Estimate Emin and Emax
        Emin = 0.1
        Emax = 10 *quant_num**2
        # Make an estimation for the energy
        E_first_guess = 1/2 * (Emax + Emin)

        #main solver for wavefunction and eigen energies
        numerov_solver = NumerovSolverPIB(param_list, E_first_guess, potential_func, Npoints)
        #secondary solver for wavefunction of Emin energy
        numerov_solver_secondary = NumerovSolverPIB(param_list, E_first_guess, potential_func, Npoints)
        Energy_guess = E_first_guess
        numerov_solver.run_solver()

        self.solve_particle_in_a_box(numerov_solver, numerov_solver_secondary, param_list, E_first_guess, Emin, Emax, Energy_guess, potential_func, Npoints, tol)

        self.plot_show(numerov_solver, matching=True)

    def calc_d(self, param_list, E_g):
        h_bar = param_list[0]
        mass = param_list[1]
        V0 = param_list[5]

        return h_bar/np.sqrt(2*mass*(V0 - E_g))

    def solve_finite_potential_well(self, solver, solver_secondary, param_list, E_first_guess, Emin, Emax, Energy_guess, potential_func, Npoints, tol):
        quant_num  = param_list[3]
        for i in range(1,Npoints):
            Energy_guess = 1/2 * (Emax + Emin)

            d = self.calc_d(param_list, Energy_guess)
            # # Use Numerov Method to solve
            psi = self.numerov_cooley_wavefunc_and_energy(solver, i, param_list, E_first_guess, Energy_guess, potential_func, Npoints)
            if i == 0:
                first_psi = solver.psi_left
                psi = first_psi

            # count nodes of wavefunction
            n_prime, _ = self.count_nodes(psi)
            if n_prime < quant_num:
                Emin = Energy_guess
                continue
            elif n_prime > quant_num:
                Emax = Energy_guess
                continue
            elif n_prime == quant_num:
                pass

            # Bisect the energy eigenvalue bounds to 
            # better approximate the boundary condition
            
            #get wavefunction for Emin
            solver_secondary.reset_parameters(param_list, Emin, potential_func, Npoints)
            solver_secondary.run_solver()
            psi_Emin = solver.psi_left
            if psi_Emin[-1] * psi[-1] > 0:
                Emin = Energy_guess
            elif psi_Emin[-1] * psi[-1] < 0:
                Emax = Energy_guess
            
            # Check energy eigenvalue convergance
            if np.abs(Emax - Emin) <= tol:
                break

    def finite_potential_well(self, param_list, potential_func, tol=1e-5):
        Npoints = self.get_value_from_user("Number of Points N", "Finite Potential Well")
        h_bar = param_list[0]
        mass = param_list[1]
        L = param_list[2]
        quant_num = param_list[3]
        x_matching_point = param_list[4]
        V0 = param_list[5]
        well_width = param_list[6]

        # To numerically solve for the nth eigenstate and the eigenvalue of the
        # time dependent Schrodinger equation:
        # Estimate Emin and Emax
        Emin = 0.1
        Emax = 10 *quant_num**2
        # Make an estimation for the energy
        E_first_guess = 1/2 * (Emax + Emin)
        while V0 < E_first_guess:
            print(f"Particle is unbounded V0: {V0} E:{E_first_guess}")
            try:
                V0 = float(input("New value for V0: -> "))
            except:
                continue
        
        param_list[5] = V0 #renew
        #calculate d
        d = self.calc_d(param_list, E_first_guess)

        # approximating the finite potential well by solving the particle in a box problem
        pib_parameters = [h_bar, mass, (L+2*d), quant_num, x_matching_point, V0, well_width]
        #main solver for wavefunction and eigen energies
        numerov_solver = NumerovSolverFPP(pib_parameters, E_first_guess, potential_func, Npoints)
        x_m = (numerov_solver.L + numerov_solver.well_width)/2
        numerov_solver.x_matching_point, numerov_solver.x_matching_point_index = numerov_solver.set_x_matching_point(x_m)
        pib_parameters = [h_bar, mass, (L+2*d), quant_num, x_m, V0, well_width]
        #secondary solver for wavefunction of Emin energy
        numerov_solver_secondary = NumerovSolverFPP(pib_parameters, E_first_guess, potential_func, Npoints)
        Energy_guess = E_first_guess
        numerov_solver.run_solver()

        self.solve_finite_potential_well(numerov_solver, numerov_solver_secondary, pib_parameters, E_first_guess, Emin, Emax, Energy_guess, potential_func, Npoints, tol)
        
        self.plot_show(numerov_solver, True)

    def potential_crosses_energy(self, mass, omega, E):

        def potential_minus_energy(x, mass, omega, E):
            return 0.5 * mass * (omega * x)**2 - E

        # Find the roots (values of x) where potential_harmonic_osc(x) - E = 0
        roots = sp_op.fsolve(potential_minus_energy, 0, args=(mass, omega, E))

        return roots

    def solve_quantum_harmonic_oscillator(self, solver, solver_secondary, param_list, E_first_guess, Emin, Emax, Energy_guess, potential_func, Npoints, tol):
        quant_num  = param_list[3]
        for i in range(1,Npoints):
            Energy_guess = 1/2 * (Emax + Emin)
            # # Use Numerov Method to solve
            psi = self.numerov_cooley_wavefunc_and_energy(solver, i, param_list, E_first_guess, Energy_guess, potential_func, Npoints)
            if i == 0:
                first_psi = solver.psi_left
                psi = first_psi

            # count nodes of wavefunction
            n_prime, _ = self.count_nodes(psi)
            if n_prime < quant_num:
                Emin = Energy_guess
                continue
            elif n_prime > quant_num:
                Emax = Energy_guess
                continue
            elif n_prime == quant_num:
                pass

            # Bisect the energy eigenvalue bounds to 
            # better approximate the boundary condition
            
            #get wavefunction for Emin
            solver_secondary.reset_parameters(param_list, Emin, potential_func, Npoints)
            solver_secondary.run_solver()
            psi_Emin = solver.psi_left
            if psi_Emin[-1] * psi[-1] > 0:
                Emin = Energy_guess
            elif psi_Emin[-1] * psi[-1] < 0:
                Emax = Energy_guess
            
            # Check energy eigenvalue convergance
            if np.abs(Emax - Emin) <= tol:
                break

    def quantum_harmonic_oscillator(self, param_list, potential_func, tol=1e-5):
        Npoints = self.get_value_from_user("Number of Points N", "Quantum Harmonic Oscillator")
        h_bar = param_list[0]
        mass = param_list[1]
        omega = param_list[2]
        quant_num = param_list[3]
        x_matching_point = param_list[4]

        # To numerically solve for the nth eigenstate and the eigenvalue of the
        # time dependent Schrodinger equation:
        # Estimate Emin and Emax
        Emin = 0.1
        Emax = 10 *quant_num**2
        # Make an estimation for the energy
        E_first_guess = 1/2 * (Emax + Emin)

        #main solver for wavefunction and eigen energies
        numerov_solver = NumerovSolverQHO(param_list, E_first_guess, potential_func, Npoints)
        #secondary solver for wavefunction of Emin energy
        numerov_solver_secondary = NumerovSolverQHO(param_list, E_first_guess, potential_func, Npoints)
        Energy_guess = E_first_guess
        numerov_solver.run_solver()

        self.solve_quantum_harmonic_oscillator(numerov_solver, numerov_solver_secondary, param_list, E_first_guess, \
                                               Emin, Emax, Energy_guess, potential_func, Npoints, tol)
        self.plot_show(numerov_solver, matching=True)

    def solve_poschl_teller_well(self, solver, solver_secondary, param_list, E_first_guess, Emin, Emax, Energy_guess, potential_func, Npoints, tol):
        quant_num  = param_list[3]
        for i in range(1,Npoints):
            Energy_guess = 1/2 * (Emax + Emin)
            # # Use Numerov Method to solve
            psi = self.numerov_cooley_wavefunc_and_energy(solver, i, param_list, E_first_guess, Energy_guess, potential_func, Npoints)
            if i == 0:
                first_psi = solver.psi_left
                psi = first_psi

            # count nodes of wavefunction
            n_prime, _ = self.count_nodes(psi)
            if n_prime < quant_num:
                Emin = Energy_guess
                continue
            elif n_prime > quant_num:
                Emax = Energy_guess
                continue
            elif n_prime == quant_num:
                pass

            # Bisect the energy eigenvalue bounds to 
            # better approximate the boundary condition
            
            #get wavefunction for Emin
            solver_secondary.reset_parameters(param_list, Emin, potential_func, Npoints)
            solver_secondary.run_solver()
            psi_Emin = solver.psi_left
            if psi_Emin[-1] * psi[-1] > 0:
                Emin = Energy_guess
            elif psi_Emin[-1] * psi[-1] < 0:
                Emax = Energy_guess
            
            # Check energy eigenvalue convergance
            if np.abs(Emax - Emin) <= tol:
                break

    def poschl_teller_well(self, param_list, potential_func, tol=1e-5):
        Npoints = self.get_value_from_user("Number of Points N", "Poschl-Teller Well")
        h_bar = param_list[0]
        mass = param_list[1]
        well_width = param_list[2]
        quant_num = param_list[3]
        x_matching_point = param_list[4]
        V0 = param_list[5]

        # To numerically solve for the nth eigenstate and the eigenvalue of the
        # time dependent Schrodinger equation:
        # Estimate Emin and Emax
        Emin = -10 * quant_num**2
        Emax = 10 *quant_num**2
        # Make an estimation for the energy
        E_first_guess = 1/2 * (Emax + Emin)


        #main solver for wavefunction and eigen energies
        numerov_solver = NumerovSolverPTW(param_list, E_first_guess, potential_func, Npoints)
        #secondary solver for wavefunction of Emin energy
        numerov_solver_secondary = NumerovSolverPTW(param_list, E_first_guess, potential_func, Npoints)
        Energy_guess = E_first_guess
        numerov_solver.run_solver()

        # self.solve_poschl_teller_well(numerov_solver, numerov_solver_secondary, param_list, E_first_guess, \
        #                                        Emin, Emax, Energy_guess, potential_func, Npoints, tol)
        self.solve_poschl_teller_well(numerov_solver, numerov_solver_secondary, param_list, E_first_guess, \
                                                Emin, Emax, Energy_guess, potential_func, Npoints, tol)
        diff = numerov_solver.Potential - numerov_solver.Energy
        if np.all(diff<0):
            print(f"Eigenstate solving for is incorrect for V0:{V0} and well_width: {well_width}\nEnergy:{numerov_solver.Energy}")
            print("Try changing V0 for the Poschl-Teller potential")
            valid = False
            V0 = None
            well_width = None
            while not valid:
                if V0 !="":
                    try:
                        V0 = input("New value for V0(press ENTER to skip): -> ")
                        if V0 == "":
                            continue
                        V0 = float(V0)
                        self.ptw_V0 = V0
                    except:
                        continue
                if well_width != "":
                    try:
                        well_width = input("New value for well_width(press ENTER to skip): -> ")
                        if well_width == "": 
                            valid = True
                            continue
                        well_width = float(well_width)
                        self.ptw_well_width = well_width
                        valid = True
                    except:
                        continue
            if valid:
                raise Exception("reset")
            

        self.plot_show(numerov_solver, matching=True)

    def solve_double_potential_well(self, solver, solver_secondary, param_list, E_first_guess, Emin, Emax, Energy_guess, potential_func, Npoints, tol):
        quant_num  = param_list[3]
        for i in range(1,Npoints):
            Energy_guess = 1/2 * (Emax + Emin)
            # # Use Numerov Method to solve
            psi = self.numerov_cooley_wavefunc_and_energy(solver, i, param_list, E_first_guess, Energy_guess, potential_func, Npoints)
            if i == 0:
                first_psi = solver.psi_left
                psi = first_psi

            # count nodes of wavefunction
            n_prime, _ = self.count_nodes(psi)
            if n_prime < quant_num:
                Emin = Energy_guess
                continue
            elif n_prime > quant_num:
                Emax = Energy_guess
                continue
            elif n_prime == quant_num:
                pass

            # Bisect the energy eigenvalue bounds to 
            # better approximate the boundary condition
            
            #get wavefunction for Emin
            solver_secondary.reset_parameters(param_list, Emin, potential_func, Npoints)
            solver_secondary.run_solver()
            psi_Emin = solver.psi_left
            if psi_Emin[-1] * psi[-1] > 0:
                Emin = Energy_guess
            elif psi_Emin[-1] * psi[-1] < 0:
                Emax = Energy_guess
            
            # Check energy eigenvalue convergance
            if np.abs(Emax - Emin) <= tol:
                break
    
    def double_well(self, param_list, potential_func, tol=1e-5):
        Npoints = self.get_value_from_user("Number of Points N", "Double Well")
        h_bar = param_list[0]
        mass = param_list[1]
        L = param_list[2]
        quant_num = param_list[3]
        x_matching_point = param_list[4]
        V0 = param_list[5]
        V0_barrier = param_list[6]
        well1_start= param_list[7]
        well1_end= param_list[8]
        well2_start= param_list[9]
        well2_end= param_list[10]

        # To numerically solve for the nth eigenstate and the eigenvalue of the
        # time dependent Schrodinger equation:
        # Estimate Emin and Emax
        Emin = 0.1
        Emax = 10 *quant_num**2
        # Make an estimation for the energy
        E_first_guess = 1/2 * (Emax + Emin)
        
        param_list[5] = V0 #renew

        # approximating the finite potential well by solving the particle in a box problem
        dwp_parameters = [h_bar, mass, L, quant_num, x_matching_point, V0, V0_barrier, well1_start, well1_end, well2_start, well2_end]
        #main solver for wavefunction and eigen energies
        numerov_solver = NumerovSolverDWP(dwp_parameters, E_first_guess, potential_func, Npoints)
        #secondary solver for wavefunction of Emin energy
        numerov_solver_secondary = NumerovSolverDWP(dwp_parameters, E_first_guess, potential_func, Npoints)
        Energy_guess = E_first_guess
        numerov_solver.run_solver()

        self.solve_double_potential_well(numerov_solver, numerov_solver_secondary, dwp_parameters, E_first_guess,
                                          Emin, Emax, Energy_guess, potential_func, Npoints, tol)
        diff = numerov_solver.Potential - numerov_solver.Energy
        if np.all(diff<0):
            print(f"Particle is unbounded V0: {V0} Energy:{numerov_solver.Energy}")
            valid = False
            V_height = None
            while True:
                try:
                    V_height = input("New value for V0(press ENTER to skip): -> ")
                    if V_height == "":
                        break
                    V_height = float(V_height)
                    self.dwp_V0 = V_height
                    symmetric = input("Should the in-between barrier gain the same potential (yes/no)?")
                    if "yes" in symmetric.lower() or "y" in symmetric.lower():
                        self.dwp_V0_barrier = V_height
                    break
                except:
                    continue
            raise Exception("reset") 

        self.plot_show(numerov_solver, matching=True)

    def potential_0(self, x, *args):
        return np.zeros(len(x))
    
    def potential_V0(self, x, L=1, well_width=1, V0=1, *args):
        V_outside = V0
        condition_lower = (L - well_width) / 2 <= x
        condition_upper = x <= (L + well_width) / 2

        return np.piecewise(x, [x < (L - well_width) / 2, condition_lower & condition_upper, x > (L + well_width) / 2],
                            [V_outside, 0, V_outside])

    def potential_harmonic_osc(self, x, mass, omega, *args):
        return 1/2 * mass* (omega*x)**2

    def potential_poschl_teller(self, x, a, V0):
        return -V0 / np.cosh(a * x)**2

    def potential_double_well(self, x, L=4, well1_start=0.5, well1_end=1.5, well2_start=2, well2_end=3, V0=1, V0_barrier=1):
        V_outside = V0
        condition_gap0 = (x<well1_start) #barrier 0 -> V0
        condition_well1 = (well1_start <= x) & (x <= well1_end) #inside well 1 -> V=0
        condition_gap1 = (well1_end < x) & (x <= well2_start) #barrier 1 -> V0
        condition_well2 = (well2_start < x) & (x <= well2_end) #inside well 2 -> V=0
        condition_gap2 = (well2_end < x) & (x <= L) #barrier 3 -> V0

        return np.piecewise(x, [condition_gap0, condition_well1, condition_gap1, condition_well2, condition_gap2],
                            [V_outside, 0, V0_barrier, 0, V_outside])

    def get_selection_message(self):
        return '''=================
Choose one of the following:
1. Particle In a Box Problem
2. Finite Potential Square Well
3. Quantum Harmonic Oscillator
4. Poschl-Teller Potential
5. Double Potential Well
-1 for EXIT
--> '''

    def main_function(self):
        selection_message = self.get_selection_message()
        user_response = None
        while not user_response or user_response !=-1:
            try:
                user_response = int(input(selection_message))
                if user_response == 1:
                    pib_param_list = self.pib_problem_parameters()
                    self.particle_in_a_box(pib_param_list, self.potential_0)
                elif user_response == 2:
                    fpw_param_list = self.fpw_problem_parameters()
                    self.finite_potential_well(fpw_param_list, self.potential_V0)
                elif user_response == 3:
                    qho_param_list = self.qho_problem_parameters()
                    self.quantum_harmonic_oscillator(qho_param_list, self.potential_harmonic_osc)
                elif user_response == 4:
                    ptw_param_list = self.ptw_problem_parameters()
                    self.poschl_teller_well(ptw_param_list, self.potential_poschl_teller)
                elif user_response == 5:
                    dwp_param_list = self.dwp_problem_parameters()
                    self.double_well(dwp_param_list, self.potential_double_well)
                else:
                    continue
            except Exception as e:
                print("ERROR:", e)
                continue
        print("Goodbye.")
        return None




def main():
    main_class_object = MainClass()
    main_class_object.main_function()

if __name__ == '__main__':
    main()