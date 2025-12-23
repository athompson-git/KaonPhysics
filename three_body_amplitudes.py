"""
3 body amplitudes for K -> pi pi X
Adapted from alplib Matrix elements classes.
"""

from constants import *
import numpy as np
from numpy import sqrt, pi
from scipy.integrate import quad


################################################################################
# Define generic superclasses for matrix elements
################################################################################


class MatrixElementDecay2:
    """
    Generic matrix elements for a 2-body decay
    m_parent: mass of the parent
    m1, m2: masses of daughters 1 and 2
    __call__: to be overridden by inheriting classes.
    """
    def __init__(self, m_parent, m1, m2):
        self.m_parent = m_parent
        self.m1 = m1
        self.m2 = m2

    def __call__(self):
        return 0.0




class MatrixElementDecay3:
    """
    Generic matrix elements for a 2-body decay
    m_parent: mass of the parent
    m1, m2, m3: masses of daughters 1 and 2
    __call__: to be overridden by inheriting classes.
    """
    def __init__(self, m_parent, m1, m2, m3):
        self.m_parent = m_parent
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def __call__(self, m122, m232):
        return 0.0

    def get_sp_from_dalitz(self, m122, m232):
        sp03 = (self.m_parent**2 + self.m3**2 - m122)/2
        sp01 = (self.m_parent**2 + self.m1**2 - m232)/2
        sp02 = (m122 + m232 - self.m1**2 - self.m3**2)/2
        sp13 = (self.m_parent**2 + self.m2**2 - m122 - m232)/2
        sp23 = (m232 - self.m2**2 - self.m3**2)/2
        sp12 = (m122 - self.m1**2 - self.m2**2)/2

        return sp01, sp02, sp03, sp12, sp13, sp23




################################################################################
# define matrix elements
################################################################################

G_8 = 8.49e-12  # MeV^-2

### F_PI in constants = 130.2 MeV, we use ~90, so we multiply by sqrt(2)
### where necessary in the below class defs.

class KL_to_pip_pim_X(MatrixElementDecay3):
    """
    K_L -> pi+ pi- X
    Involves contact + kaon emission + pion emission diagrams
    """
    def __init__(self, mX=17.0,
                 gPiPiKPrime=1.0,
                 gT3V=0.0,
                 gU3V=0.0):
        
        super().__init__(m_parent=M_KLONG, m1=M_PI, m2=M_PI, m3=mX)
        self.gPiPiKPrime = gPiPiKPrime
        self.gT3V = gT3V
        self.gU3V = gU3V
        self.mX = mX
    
    @property
    def mX(self):
        return self.m3
    
    @mX.setter
    def mX(self, value):
        self.m3 = value

    def M2Contact(self, m122, m232):
        prefact = (G_8 * F_PI/sqrt(2))**2 * self.gPiPiKPrime**2

    def M2ContactPion(self, m122, m232):
        prefact = (G_8 * F_PI/sqrt(2))**2 * self.gPiPiKPrime * self.gT3V

    def M2ContactKaon(self, m122, m232):
        prefact = (G_8 * F_PI/sqrt(2))**2 * self.gPiPiKPrime * self.gU3V

    def M2PionKaon(self, m122, m232):
        prefact = (G_8 * F_PI/sqrt(2))**2 * self.gU3V * self.gT3V

    def M2Pion(self, m122, m232):
        prefact = (G_8 * F_PI/sqrt(2))**2 * self.gT3V**2

    def M2Kaon(self, m122, m232):
        prefact = (G_8 * F_PI/sqrt(2))**2 * self.gU3V**2
    
    def __call__(self, m122, m232):
        prefact = (G_8 * F_PI/sqrt(2))**2 * self.gPiPiKPrime**2
        Ex = (self.m_parent**2 + self.m3**2 - m122)/(2*self.m_parent)

        return prefact * self.m_parent**2 * (np.power(Ex/self.m3, 2) - 1)




class KL_to_pi0_pi0_X(MatrixElementDecay3):
    """
    K_L -> pi+ pi- X
    """
    def __init__(self, mX=17.0, coupling_combination=1.0):
        super().__init__(m_parent=M_KLONG, m1=M_PI0, m2=M_PI0, m3=mX)
        self.gEff = coupling_combination
        self.mX = mX
    
    @property
    def mX(self):
        return self.m3
    
    @mX.setter
    def mX(self, value):
        self.m3 = value
    
    def __call__(self, m122, m232):
        prefact = 9*(G_8 * F_PI/sqrt(2))**2 * self.gEff**2
        Ex = (self.m_parent**2 + self.m3**2 - m122)/(2*self.m_parent)

        # symmetry factor of 1/(2!)
        return prefact * self.m_parent**2 * (np.power(Ex/self.m3, 2) - 1) / 2




class Kp_to_pip_pi0_X(MatrixElementDecay3):
    """
    K_L -> pi+ pi- X
    """
    def __init__(self, mX=17.0, coupling_combination=1.0):
        super().__init__(m_parent=M_K, m1=M_PI, m2=M_PI0, m3=mX)
        self.gEff = coupling_combination
        self.mX = self.m3
    
    @property
    def mX(self):
        return self.m3
    
    @mX.setter
    def mX(self, value):
        self.m3 = value
    
    def __call__(self, m122, m232):
        prefact = (G_8 * F_PI/sqrt(2))**2 * self.gEff**2
        return (prefact / (4*self.m3**2)) * (self.m_parent**4 - 2*self.m_parent**2 * (m122 + self.m3**2) \
                                             + (m122 - self.m3**2)**2)




class Kp_to_pip_XX(MatrixElementDecay3):
    """
    K+ -> pi+ X X
    """
    def __init__(self, mX=17.0,
                    gKPiPlusXX=1.0,
                    gKPiPlus=0.0,
                    gV3V=0.0,
                    gT3V=0.0):
        super().__init__(M_K, mX, mX, M_PI)

        self.gKPiPlus = gKPiPlus
        self.gKPiPlusXX = gKPiPlusXX
        self.gV3V = gV3V
        self.gT3V = gT3V
        self.mX = mX

    @property
    def mX(self):
        return self.m3
    
    @mX.setter
    def mX(self, value):
        self.m3 = value

    def __call__(self, m122, m232):
        prefactor = M_K**6 * KAPPA**2 / (2 * self.mX**4)
        x12 = m122 / (M_K**2)
        x1Pi = m232 / (M_K**2)
        return prefactor * (self.gKPiPlus * (self.gV3V - self.gT3V) * (1 - 2*x1Pi + PI_K_RATIO_SQ) \
                            + 2 * (self.gKPiPlusXX - self.gT3V * self.gV3V) * x12)**2




################################################################################
# Exact decay widths ###
################################################################################

KAPPA = 7.35e-8
PI_K_RATIO_SQ = (M_PI/M_K)**2

# K+ to pi+ gamma X
def dGamma_dxdy_KPiGammaX(y, x, gKPI, mX):
    prefactor = KAPPA**2 * ALPHA * M_K**3 / (16 * pi**2) / mX**2

    rX = (mX / M_K)**2
    x_pi_gamma = (M_K**2 + mX**2 + M_PI**2 - x - y) / M_K**2
    N0 = x_pi_gamma*y - PI_K_RATIO_SQ**2 * (1 - x_pi_gamma * (1 - 2 * y))
    D1 = PI_K_RATIO_SQ - x_pi_gamma
    D2 = PI_K_RATIO_SQ + rX - x_pi_gamma - x

    return prefactor * gKPI**2 * x * (N0) / (D1**2 * D2**2)

def dGamma_dx_KPiGammaX(x, gKPI, mX):
    E3star = (M_K**2 * (1 - x) - M_PI**2)/(2 * M_K * sqrt(x))
    E2star = (M_K**2 * x + mX**2)/(2 * M_K * sqrt(x))

    m23_max = (E2star + E3star)**2 - np.power(sqrt(E2star**2 - mX**2) - sqrt(E3star**2 - M_PI**2), 2)
    m23_min = (E2star + E3star)**2 - np.power(sqrt(E2star**2 - mX**2) + sqrt(E3star**2 - M_PI**2), 2)
    y_min = m23_min / M_K**2
    y_max = m23_max / M_K**2

    integral = quad(dGamma_dxdy_KPiGammaX, y_min, y_max, args=(x, gKPI, mX))[0]
    return integral




# Widths for K --> pi0 X
def Gamma_KSPi0X(mX, gKpi):
    pX = (M_K/2) * sqrt(1 - np.power(2 * mX / M_K, 2))

    return KAPPA**2 * pX**3 * gKpi**2 / (pi * mX**2)

def Gamma_KPlusPiPlusX(mX, gKpi):
    pX = (M_K/2) * sqrt(1 - np.power(2 * mX / M_K, 2))

    return KAPPA**2 * pX**3 * gKpi**2 / (4 * pi * mX**2)




# Widths for K+ --> pi+ X X
def dGamma_KPlusPiPlusXX(m12Sq, m1PiSq,
                         mX=17.0,
                         gKPiPlusXX=1.0,
                         gKPiPlus=0.0,
                         gV3V=0.0,
                         gT3V=0.0):
    prefactor = M_K**9 * KAPPA**2 / (512 * pi**3 * mX**4)
    x12 = m12Sq / (M_K**2)
    x1Pi = m1PiSq / (M_K**2)
    return prefactor * (gKPiPlus * (gV3V - gT3V) * (1 - 2*x1Pi + PI_K_RATIO_SQ) \
                        + 2 * (gKPiPlusXX - gT3V * gV3V) * x12)**2