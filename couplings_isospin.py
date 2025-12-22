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
