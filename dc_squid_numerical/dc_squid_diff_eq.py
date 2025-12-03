from scipy.integrate import solve_ivp
from scipy.constants import physical_constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import sin, pi
from rich.progress import track

phi0 = physical_constants['mag. flux quantum'][0]
kB = physical_constants['Boltzmann constant'][0]

class DcSquid:

    def __init__(self,
                 I0, 
                 R, 
                 L,
                 C,
                 T=0):
        self.criticalCurrent = I0
        self.temperature = T
        self.resistance = R
        self.inductance = L
        self.capacitance = C

        # independent variables
        # current sent through the SQUID, aka bias current
        self.current = 0
        # external flux in units of phi0
        self.flux = 0

        # derived parameters
        self.betaC = (2*pi/phi0)*self.criticalCurrent*self.resistance**2*self.capacitance # Stewart-McCumber parameter
        self.betaL = 2*self.inductance*self.criticalCurrent / phi0 # screening parameter
        self.omegaC = 2*pi*self.criticalCurrent*self.resistance/phi0 # characteristic time 

        # ODE parameters
        # number of time steps to calculate the diff eq into the future state, in units of self.omegaC
        self.tMax = 100
        self.tStep = 0.1

        # noise parameters
        self.gamma1 = 0
        self.gamma2 = 0

    def calculateParameters(self):
        """
        Used to recalculate parameters if anything has been assigned a new value
        """

        self.gamma1 = (2*pi/phi0)*kB*self.temperature/self.criticalCurrent
        self.gamma2 = (2*pi/phi0)*kB*self.temperature/self.criticalCurrent

        self.betaC = (2*pi/phi0)*self.criticalCurrent*self.resistance**2*self.capacitance
        self.betaL = 2*self.inductance*self.criticalCurrent / phi0
        self.omegaC = 2*pi*self.criticalCurrent*self.resistance/phi0

    @property
    def voltage(self):
        return self.data[:,5]
    
    @property
    def biasCurrent(self):
        return self.data[:,4]

    def dcSquidEquationOfMotion(self, t, y):
        
        delta1    = y[0]
        delta1dot = y[1]
        delta2    = y[2]
        delta2dot = y[3]

        # circulating current
        J = 1/(self.inductance*self.capacitance)*(delta1 - delta2 - 2*pi*self.flux)
        
        return np.array((
            delta1dot,
            (2*pi)/(phi0*self.capacitance)*(self.current/2 - self.criticalCurrent*sin(delta1) + self.gamma1*self.criticalCurrent) - J - 1/(self.resistance*self.capacitance)*delta1dot,
            delta2dot,
            (2*pi)/(phi0*self.capacitance)*(self.current/2 - self.criticalCurrent*sin(delta2) + self.gamma2*self.criticalCurrent) + J - 1/(self.resistance*self.capacitance)*delta2dot,
        ))
    
    def simulateIV(self, biasStart, biasStop, nPoints, plot=False):
        """
        Simulate the IV curve using the Runge-Kutta 4(5) method

        Args:
            biasStart: start bias current in Amps
            biasEnd: end bias current in Amps
            nPoints: number of bias points
            plot: show the plot
        """

        self.calculateParameters()

        self.data = np.zeros((nPoints, 6))
        sweepPoints = np.linspace(biasStart, biasStop, nPoints)
        self.biasSweepValues = sweepPoints

        state = [
            pi/4,
            0,
            pi/6,
            0]

        i = 0
        for bias in track(sweepPoints, description='Sweeping...'):
            self.current = bias
            sol = solve_ivp(
                self.dcSquidEquationOfMotion,
                [0, self.tMax/self.omegaC],
                state,
                method='RK45',
                max_step=1/self.omegaC*self.tStep)

            delta1    = sol.y[0]
            delta1dot = sol.y[1]
            delta2    = sol.y[2]
            delta2dot = sol.y[3]

            delta1_i    = delta1[:-self.tMax//4].mean()
            delta1dot_i = delta1dot[:-self.tMax//4].mean()
            delta2_i    = delta2[:-self.tMax//4].mean()
            delta2dot_i = delta2dot[:-self.tMax//4].mean()

            stateVector = np.array([delta1_i, delta1dot_i, delta2_i, delta2dot_i])

            averageVoltage = (phi0/(4*pi))*(delta1dot_i + delta2dot_i)

            self.data[i, :-2] = stateVector
            self.data[i,4] = bias
            self.data[i,5] = averageVoltage

            # update the state vector for the next iteration
            state = stateVector

            i += 1

        if plot:
            plt.plot(self.biasSweepValues*1e6, self.voltage*1e6, color="lightseagreen")
            plt.xlabel("Bias current (uA)", fontsize=14)
            plt.ylabel("Voltage (uV)", fontsize=14)
            plt.grid(alpha=0.6)
            plt.show()
    

    def simulateVPhi(self, fluxStart, fluxStop, nPoints, plot=False):
        """
        Simulate the V-phi curve using the Runge-Kutta 4(5) method 
        
        That is sweep an external flux through the SQUID loop and plot the voltage across the SQUID

        It should be periodic in multiples of phi0

        Args:
            fluxStart: start flux value in units of phi0
            fluxStop: end flux value in units of phi0
            nPoints: number of flux points
            plot: show the plot
        """

        self.calculateParameters()

        # clear data frame
        self.data = np.zeros((nPoints, 6))

        sweepPoints = np.linspace(fluxStart, fluxStop, nPoints)
        self.fluxSweepValues = sweepPoints

        state = [
            pi/4,
            0,
            pi/6,
            0]

        i = 0
        for fluxExternal in track(sweepPoints, description='Sweeping...'):
            self.flux = fluxExternal
            
            sol = solve_ivp(
                self.dcSquidEquationOfMotion,
                [0, self.tMax/self.omegaC],
                state,
                method='RK45',
                max_step=1/self.omegaC*self.tStep)
            
            delta1    = sol.y[0]
            delta1dot = sol.y[1]
            delta2    = sol.y[2]
            delta2dot = sol.y[3]

            delta1_i    = delta1[:-self.tMax//4].mean()
            delta1dot_i = delta1dot[:-self.tMax//4].mean()
            delta2_i    = delta2[:-self.tMax//4].mean()
            delta2dot_i = delta2dot[:-self.tMax//4].mean()

            stateVector = np.array([delta1_i, delta1dot_i, delta2_i, delta2dot_i])

            averageVoltage = (phi0/(4*pi))*(delta1dot_i + delta2dot_i)

            self.data[i, :-2] = stateVector
            self.data[i,4] = self.current
            self.data[i,5] = averageVoltage

            # update the state vector for the next iteration
            state = stateVector

            i += 1
        
        voltagePlot = self.data[:, 5]
        voltagePlotDimensionless = voltagePlot / (self.criticalCurrent * self.resistance)

        if plot:
            plt.plot(self.fluxSweepValues, voltagePlotDimensionless, color="deeppink")
            plt.xlabel(r"$\Phi_{ext}/\Phi_0$", fontsize=14)
            plt.ylabel(r"Voltage Across the SQUID $V/I_0 R$", fontsize=14)
            plt.grid(alpha=0.6)
            plt.xticks(np.arange(-2, 2.1, 0.25), rotation=90, fontsize=10)
            plt.tight_layout()
            plt.show()


def iv_curve(biasStart, biasStop, nPoints):
    """
    Simulate the IV curve
    """
    SQ = DcSquid(
        I0=10e-6, 
        R=20, 
        L=120e-12, 
        C=8e-14)
    
    SQ.temperature = 0

    SQ.simulateIV(biasStart, biasStop, nPoints)
    plt.plot(SQ.biasSweepValues*1e6, SQ.voltage*1e6, color="lightseagreen")
    plt.xlabel("Bias current (uA)", fontsize=14)
    plt.ylabel("Voltage (uV)", fontsize=14)
    plt.grid(alpha=0.6)
    plt.show()


def flux_sweep(bias_current):
    """
    Simulate a single V-phi curve

    Args:
    bias_current
        The bias current in Amperes
    """
    SQ = DcSquid(
        I0=10e-6, 
        R=20, 
        L=120e-12, 
        C=8e-14)
    
    SQ.current = bias_current
    SQ.simulateVPhi(-2, 2, 501, plot=True)


def flux_sweep_bias_sweep(bias_currents):
    """
    Simulate the V-phi curve at several different bias points
    """
    SQ = DcSquid(
        I0=10e-6, 
        R=20, 
        L=120e-12, 
        C=8e-14)
    
    SQ.temperature = 0
    
    cmap = mpl.colormaps['viridis']

    colors = cmap(np.linspace(0,1,len(bias_currents)))
    
    i = 0
    for b in bias_currents:
        SQ.current = b*SQ.criticalCurrent

        SQ.simulateVPhi(-2, 2, 501)
        plt.plot(SQ.fluxSweepValues, SQ.voltage/(SQ.criticalCurrent*SQ.resistance), label=r'$I_b=$' + f'{b:.2f}' + r"$I_0$", color=colors[i])
        i += 1

    plt.legend()
    plt.xlabel(r"$\Phi_{ext}/\Phi_0$", fontsize=14)
    plt.ylabel(r"Voltage Across the SQUID $V/I_0 R$", fontsize=14)
    plt.grid(alpha=0.6)
    plt.xticks(np.arange(-2, 2.1, 0.25), rotation=90, fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # example functions with SQUID parameters that will work correctly

    iv_curve(-50e-6, 50e-6, 501)

    flux_sweep(2.4*10e-6)

    flux_sweep_bias_sweep(np.linspace(0.1, 3.1, 8))



    # do it like this to define your own SQUID parameters

    SQ = DcSquid(I0=10e-6, R=20, L=120e-12, C=8e-14)
    SQ.current = 3.5*SQ.criticalCurrent
    SQ.simulateVPhi(-2, 2, 501, plot=True)