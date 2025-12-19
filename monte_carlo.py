"""
Cross section class and MC
Adapted from alplib (https://github.com/athompson-git/alplib)
Copyright Adrian Thompson (2025) MIT License
"""

from constants import *
from three_body_amplitudes import *
import numpy as np
from numpy import arccos, arctan2, sin, cos, power


class Vector3:
    """
    Docstring for Vector3
    Init arguments:
    v1, v2, v3: (float) components of 3-vector
    """
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.vec = np.array([v1, v2, v3])
    
    def __str__(self):
        return "({0},{1},{2})".format(self.v1, self.v2, self.v3)
    
    def __add__(self, other):
        v1_new = self.v1 + other.v1
        v2_new = self.v2 + other.v2
        v3_new = self.v3 + other.v3
        return Vector3(v1_new, v2_new, v3_new)
    
    def __sub__(self, other):
        v1_new = self.v1 - other.v1
        v2_new = self.v2 - other.v2
        v3_new = self.v3 - other.v3
        return Vector3(v1_new, v2_new, v3_new)
    
    def __neg__(self):
        return Vector3(-self.v1, -self.v2, -self.v3)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):  # Scalar multiplication
            return Vector3(self.v1 * other, self.v2 * other, self.v3 * other)
        elif isinstance(other, Vector3):  # Dot product
            return np.dot(self.vec, other.vec)
        else:
            raise TypeError("Unsupported operand type for *: 'Vector' and '{}'".format(type(other)))
    
    def __rmul__(self, other):
        return self.__mul__(other)  # Reuse __mul__
    
    def unit_vec(self, eps=1e-40):
        v = self.mag()
        if v < eps:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.v1/v, self.v2/v, self.v3/v)
    
    def mag2(self):
        return np.dot(self.vec, self.vec)
    
    def mag(self):
        return np.sqrt(np.dot(self.vec, self.vec))

    def phi(self):
        return arctan2(self.v2, self.v1)
    
    def theta(self):
        return arccos(self.v3 / self.mag())
    
    def set_v3(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.vec = np.array([v1, v2, v3])




class LorentzVector:
    """
    Docstring for LorentzVector
    
    Init:
    p0 (float): Energy
    p1, p2, p3 (float): x, y, and z component momenta
    """
    def __init__(self, p0=0.0, p1=0.0, p2=0.0, p3=0.0):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.pmu = np.array([p0, p1, p2, p3])
        self.mt = np.array([1, -1, -1, -1])
        self.momentum3 = Vector3(self.p1, self.p2, self.p3)
    
    def __str__(self):
        return "({0},{1},{2},{3})".format(self.p0, self.p1, self.p2, self.p3)
    
    def __add__(self, other):
        p0_new = self.p0 + other.p0
        p1_new = self.p1 + other.p1
        p2_new = self.p2 + other.p2
        p3_new = self.p3 + other.p3
        return LorentzVector(p0_new, p1_new, p2_new, p3_new)
    
    def __sub__(self, other):
        p0_new = self.p0 - other.p0
        p1_new = self.p1 - other.p1
        p2_new = self.p2 - other.p2
        p3_new = self.p3 - other.p3
        return LorentzVector(p0_new, p1_new, p2_new, p3_new)
    
    def __mul__(self, other):
        return np.dot(self.pmu*other.pmu, self.mt)
    
    def __rmul__(self, other):
        return np.dot(self.pmu*other.pmu, self.mt)
    
    def mass2(self):
        return np.dot(self.pmu**2, self.mt)
    
    def mass(self):
        return np.sqrt(np.dot(self.pmu**2, self.mt))
    
    def energy(self):
        return self.p0
    
    def cosine(self):
        return self.p3 / self.momentum()
    
    def phi(self):
        return arctan2(self.p2, self.p1)
    
    def theta(self):
        return arccos(self.p3 / self.momentum())
    
    def momentum(self):
        return self.momentum3.mag()
    
    def set_p4(self, p0, p1, p2, p3):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.pmu = np.array([p0, p1, p2, p3])
        self.momentum3 = Vector3(self.p1, self.p2, self.p3)
    
    def get_3momentum(self):
        return Vector3(self.p1, self.p2, self.p3)
    
    def get_3velocity(self):
        return Vector3(self.p1/self.p0, self.p2/self.p0, self.p3/self.p0)



    
class Decay2Body:
    """
    2-body decay monte carlo class
    Init:
    p_parent: the LorentzVector decaying particle momentum
    m1: daughter particle 1 mass
    m2: daughter particle 2 mass
    """
    def __init__(self, p_parent: LorentzVector, m1, m2, n_samples=1000):
        self.mp = p_parent.mass()  # parent particle
        self.m1 = m1  # decay body 1
        self.m2 = m2  # decay body 2

        self.lv_p = p_parent

        self.n_samples = n_samples
        self.p1_cm_4vectors = []
        self.p1_lab_4vectors = []
        self.p2_cm_4vectors = []
        self.p2_lab_4vectors = []
        self.weights = np.array([])
    
    def set_new_decay(self, p_parent: LorentzVector, m1, m2):
        self.mp = p_parent.mass()  # parent particle
        self.m1 = m1  # decay body 1
        self.m2 = m2  # decay body 2

        self.lv_p = p_parent

        self.p1_cm_4vectors = []
        self.p1_lab_4vectors = []
        self.p2_cm_4vectors = []
        self.p2_lab_4vectors = []
        self.weights = np.array([])
    
    def decay(self):
        p_cm = power((self.mp**2 - (self.m2 - self.m1)**2)*(self.mp**2 - (self.m2 + self.m1)**2), 0.5)/(2*self.mp)
        e1_cm = sqrt(p_cm**2 + self.m1**2)
        e2_cm = sqrt(p_cm**2 + self.m2**2)

        # Draw random variates on the 2-sphere
        phi1_rnd = 2*pi*np.random.ranf(self.n_samples)
        theta1_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))

        v_in = -self.lv_p.get_3velocity()

        self.p1_cm_4vectors = [LorentzVector(e1_cm,
                            p_cm*cos(phi1_rnd[i])*sin(theta1_rnd[i]),
                            p_cm*sin(phi1_rnd[i])*sin(theta1_rnd[i]),
                            p_cm*cos(theta1_rnd[i])) for i in range(self.n_samples)]
        self.p2_cm_4vectors = [LorentzVector(e2_cm,
                            -p_cm*cos(phi1_rnd[i])*sin(theta1_rnd[i]),
                            -p_cm*sin(phi1_rnd[i])*sin(theta1_rnd[i]),
                            -p_cm*cos(theta1_rnd[i])) for i in range(self.n_samples)]
        self.p1_lab_4vectors = [lorentz_boost(p1, v_in) for p1 in self.p1_cm_4vectors]
        self.p2_lab_4vectors = [lorentz_boost(p2, v_in) for p2 in self.p2_cm_4vectors]
        self.weights = np.ones(self.n_samples) / self.n_samples
    
    def decay_from_flux(self):
        p_cm = power((self.mp**2 - (self.m2 - self.m1)**2)*(self.mp**2 - (self.m2 + self.m1)**2), 0.5)/(2*self.mp)
        e1_cm = sqrt(p_cm**2 + self.m1**2)
        e2_cm = sqrt(p_cm**2 + self.m2**2)

        # Draw random variates on the 2-sphere
        phi1_rnd = 2*pi*np.random.ranf(self.n_samples)
        theta1_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))
        phi2_rnd = np.pi + phi1_rnd
        theta2_rnd = np.pi - theta1_rnd

        v_in = [-lv.get_3velocity() for lv in self.lv_p]

        self.p1_cm_4vectors = [LorentzVector(e1_cm,
                            p_cm*cos(phi1_rnd[i])*sin(theta1_rnd[i]),
                            p_cm*sin(phi1_rnd[i])*sin(theta1_rnd[i]),
                            p_cm*cos(theta1_rnd[i])) for i in range(self.n_samples)]
        self.p2_cm_4vectors = [LorentzVector(e2_cm,
                            p_cm*cos(phi2_rnd[i])*sin(theta2_rnd[i]),
                            p_cm*sin(phi2_rnd[i])*sin(theta2_rnd[i]),
                            p_cm*cos(theta2_rnd[i])) for i in range(self.n_samples)]
        self.p1_lab_4vectors = [lorentz_boost(p1, v_in) for p1 in self.p1_cm_4vectors]
        self.p2_lab_4vectors = [lorentz_boost(p2, v_in) for p2 in self.p2_cm_4vectors]
        self.weights = np.ones(self.n_samples)




class Decay3Body:
    """
    Performs a weighted MC sampling of the differential 3-body decay width
    by choosing a final-state particle of interest (particle #3 by convention)
    and drawing random variates in the rest-frame of the parent; particle 3
    has a random angle on the 2-sphere and a random energy between some E_max and E_min.
    The weight is given by dGamma / dE_3 dOmega
    Finally, we boost to the lab frame with the appropriate Jacobian factor.

    Init:
    mtrx2 (MatrixElementDecay3): class for 3-body decays, should be called based on Dalitz
    variable conventions.
    p (LorentzVector): parent momentum
    n_samples (int): number of MC samples to generate
    total_width (float): Total width for parent, used to normalize the partial BR.
    """
    def __init__(self, mtrx2: MatrixElementDecay3, p: LorentzVector, n_samples=1000, total_width=None):
        self.mtrx2 = mtrx2
        self.n_samples = n_samples

        self.parent_p4 = p

        self.m_parent = mtrx2.m_parent
        self.m1 = mtrx2.m1
        self.m2 = mtrx2.m2
        self.m3 = mtrx2.m3

        self.p3_cm_4vectors = []
        self.p3_lab_4vectors = []
        self.weights = np.array([])

        if total_width is None:
            self.total_width = self.partial_width()
        else:
            self.total_width = total_width
    
    def set_masses(self, m1=None, m2=None, m3=None, m_parent=None):
        if m1 is not None:
            self.m1 = m1
            self.mtrx2.m1 = m1
        if m2 is not None:
            self.m2 = m2
            self.mtrx2.m2 = m2
        if m3 is not None:
            self.m3 = m3
            self.mtrx2.m3 = m3
        if m_parent is not None:
            self.m_parent = m_parent
            self.mtrx2.m_parent = m_parent
        self.total_width = self.partial_width()

    def dGammadE3(self, E3):
        m212 = self.m_parent**2 + self.m3**2 - 2*self.m_parent*E3
        e2star = np.clip((m212 - self.m1**2 + self.m2**2)/(2*sqrt(m212)), a_min=self.m2, a_max=np.inf)
        e3star = np.clip((self.m_parent**2 - m212 - self.m3**2)/(2*sqrt(m212)), a_min=self.m3, a_max=np.inf)

        m223Max = (e2star + e3star)**2 - (sqrt(e2star**2 - self.m2**2) - sqrt(e3star**2 - self.m3**2))**2
        m223Min = (e2star + e3star)**2 - (sqrt(e2star**2 - self.m2**2) + sqrt(e3star**2 - self.m3**2))**2

        def MatrixElement2(m223):
            return self.mtrx2(m212, m223)

        return (2*self.m_parent)/(32*power(2*pi*self.m_parent, 3))*quad(MatrixElement2, m223Min, m223Max)[0]

    def partial_width(self):
        ea_max = (self.m_parent**2 + self.m3**2 - (self.m2 + self.m1)**2)/(2*self.m_parent)
        return quad(self.dGammadE3, self.m3, ea_max)[0]

    def simulate_decay(self):
        # Simulates the weighted-MC 3-body decay, outputting the 4-vectors of particle #3
        # TODO(AT): add calculations for particles 1,2
        ea_min = self.m3
        ea_max = (self.m_parent**2 + self.m3**2 - self.m2**2 - self.m1**2)/(2*self.m_parent)

        # Boost to lab frame
        beta = self.parent_p4.momentum() / self.parent_p4.energy()
        boost = power(1-beta**2, -0.5)
        beta_parent = -self.parent_p4.get_3velocity()

        # Draw random variate energies and angles in the parent rest frame
        e3_rnd = np.random.uniform(ea_min, ea_max, self.n_samples)
        p3_rnd = sqrt(e3_rnd**2 - self.m3**2)
        theta3_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))
        phi3_rnd = np.random.uniform(0, 2*pi, self.n_samples)

        self.p3_cm_4vectors = [LorentzVector(e3_rnd[i],
                            -p3_rnd[i]*cos(phi3_rnd[i])*sin(theta3_rnd[i]),
                            -p3_rnd[i]*sin(phi3_rnd[i])*sin(theta3_rnd[i]),
                            -p3_rnd[i]*cos(theta3_rnd[i])) for i in range(self.n_samples)]
        self.p3_lab_4vectors = [lorentz_boost(p1, beta_parent) for p1 in self.p3_cm_4vectors]

        # Draw weights from the PDF: (Jacobian) * dGamma/dE_CM * MC volume
        mc_factor = (ea_max - ea_min)/self.total_width/self.n_samples
        jacobian = np.array([self.p3_lab_4vectors[i].momentum()/self.p3_cm_4vectors[i].momentum()
                             for i in range(self.n_samples)])

        self.weights = np.array([mc_factor*(jacobian[i])*self.dGammadE3(e3_rnd[i]) \
                                    for i in range(self.n_samples)])




def lorentz_boost(momentum: LorentzVector, v: Vector3):
    """
    Lorentz boost momentum to a new frame with velocity v
    :param momentum: four vector
    :param v: velocity of new frame, 3-dimention
    :return: boosted momentum
    """
    n = v.unit_vec().vec
    beta = v.mag()
    if beta == 0.0:
        return momentum
    gamma = 1/np.sqrt(1-beta**2)
    mat = np.array([[gamma, -gamma*beta*n[0], -gamma*beta*n[1], -gamma*beta*n[2]],
                    [-gamma*beta*n[0], 1+(gamma-1)*n[0]*n[0], (gamma-1)*n[0]*n[1], (gamma-1)*n[0]*n[2]],
                    [-gamma*beta*n[1], (gamma-1)*n[1]*n[0], 1+(gamma-1)*n[1]*n[1], (gamma-1)*n[1]*n[2]],
                    [-gamma*beta*n[2], (gamma-1)*n[2]*n[0], (gamma-1)*n[2]*n[1], 1+(gamma-1)*n[2]*n[2]]])
    boosted_p4 = mat @ momentum.pmu
    return LorentzVector(boosted_p4[0], boosted_p4[1], boosted_p4[2], boosted_p4[3])