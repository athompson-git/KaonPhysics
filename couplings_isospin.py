"""
Coupling definitions for kaon decays in the T, U, V isospin basis
Copyright Adrian Thompson (2025) MIT License
"""

# Singlet couplings
def gT0V(gVu=0.0, gVd=0.0):
    return gVu + gVd

def gT0A(gAu=0.0, gAd=0.0):
    return gAu + gAd

def gU0V(gVd=0.0, gVs=0.0):
    return gVd + gVs

def gU0A(gAd=0.0, gAs=0.0):
    return gAd + gAs

def gV0V(gVu=0.0, gVs=0.0):
    return gVu + gVs

def gV0A(gAu=0.0, gAs=0.0):
    return gAu + gAs


# Triplet couplings
def gT3V(gVu=0.0, gVd=0.0):
    return gVu - gVd

def gT3A(gAu=0.0, gAd=0.0):
    return gAu - gAd

def gU3V(gVd=0.0, gVs=0.0):
    return gVd - gVs

def gU3A(gAd=0.0, gAs=0.0):
    return gAd - gAs

def gV3V(gVu=0.0, gVs=0.0):
    return gVu - gVs

def gV3A(gAu=0.0, gAs=0.0):
    return gAu - gAs




# Coupling class to track individual vector and axial-vector couplings
# at the u, d, s level
class IsoSpinCouplings:
    def __init__(self, gVu=0.0, gVd=0.0, gVs=0.0,
                 gAu=0.0, gAd=0.0, gAs=0.0):
        self._gVu = gVu
        self._gVd = gVd
        self._gVs = gVs
        self._gAu = gAu
        self._gAd = gAd
        self._gAs = gAs
    
    def set_couplings(self, *,
                      gVu=None, gVd=None, gVs=None,
                      gAu=None, gAd=None, gAs=None):
        """
        Update quark-level couplings.
        Only arguments that are not None are modified.
        """
        if gVu is not None:
            self._gVu = float(gVu)
        if gVd is not None:
            self._gVd = float(gVd)
        if gVs is not None:
            self._gVs = float(gVs)

        if gAu is not None:
            self._gAu = float(gAu)
        if gAd is not None:
            self._gAd = float(gAd)
        if gAs is not None:
            self._gAs = float(gAs)

    # Read-only fundamental couplings
    @property
    def gVu(self): return self._gVu

    @property
    def gVd(self): return self._gVd

    @property
    def gVs(self): return self._gVs

    @property
    def gAu(self): return self._gAu

    @property
    def gAd(self): return self._gAd

    @property
    def gAs(self): return self._gAs

    # T isospin (u, d)
    @property
    def gT0V(self): return self.gVu + self.gVd

    @property
    def gT3V(self): return self.gVu - self.gVd

    @property
    def gT0A(self): return self.gAu + self.gAd

    @property
    def gT3A(self): return self.gAu - self.gAd

    # U isospin (d, s)
    @property
    def gU0V(self): return self.gVd + self.gVs

    @property
    def gU3V(self): return self.gVd - self.gVs

    @property
    def gU0A(self): return self.gAd + self.gAs

    @property
    def gU3A(self): return self.gAd - self.gAs

    # V isospin (u, s)
    @property
    def gV0V(self): return self.gVu + self.gVs

    @property
    def gV3V(self): return self.gVu - self.gVs

    @property
    def gV0A(self): return self.gAu + self.gAs

    @property
    def gV3A(self): return self.gAu - self.gAs