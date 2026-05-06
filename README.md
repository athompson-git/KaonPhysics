# KaonPhysics
## _Kaon decay constraints on vector bosons coupled to non-conserved currents_

This repository contains partial libraries for the computation of the emission of vector and axial-vector bosons
coupled to non-conserved currents in kaon decays using the $\Delta S = 1$ chiral perturbation theory (ChPT).


Based on work in collaboration with M. Hostert, M. Pospelov, A. Thompson [arXiv:2602.19479](https://arxiv.org/abs/2602.19479),
accepted for publication in _Physcial Review D_, 2026.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20058561.svg)](https://doi.org/10.5281/zenodo.20058561)

#### Requirements:
* matplotlib 3.10.8
* numpy 2.3.5
* scipy 1.16.3
See also requirements.txt; you can use it via ```pip install -r requirements.txt```.


## Computing the 3-body and 2-body amplitudes and monte carlo
The relevant helper functions, classes, and physical constant definitions are contained in the following files:
* ```monte_carlo.py``` for helper functions
* ```three_body_amplitudes.py``` for matrix elements and analytic decay widths
* ```couplings_isospin.py``` for dictionary converting quark-level (u,d,s) couplings to the T, U, V spin basis
* ```constants.py``` stores general physics constants and unit conventions
The key calculational ingredients are stored in ```three_body_amplitudes.py```. The classes in this file inherit from
the ```MatrixElementDecay3``` class structure which captures the kinematic structure of general 3-body decays using
Dalitz variables. The specific computation of these amplitudes was carried out, as explained in the paper, by
1. Expanding out the ChPT $\Delta S = 1$ Lagrangian density $\mathcal{L}_{p^2, G_8}^{\Delta S = 1}$ in powers of $1/F$
2. Performing a change of basis to rotate away kinetic mixing terms
3. Extracting the relevant 3-point and 4-point operators and their coefficients
4. Applying the Feynman rules using these operators to compute each kaon decay channel


## Jupyter Notebooks: Computing limits from NA48/2, NA62, and KTeV data

These general limits were computed in the following Jupyter notebooks
* KPlus_to_gamma_X_piplus.ipynb: uses NA48/2 for $K^+ \to \pi^+ \gamma X$ analysis
* KtoX_param_space.ipynb: combined all analysis channels and sensitivity plot

Note that partial amplitudes neglecting some diagrams were used in the practical computation of these limits;
this is allowed for when we consider the flavor universal choice $g_u^{V,A} = g_d^{V,A} = g_s^{V,A}$ in which case
most of the IB diagrams (radiation off the initial kaon leg or final state pion leg) drop away, leaving us only with
contact interactions. A fuller exploration of the 6D parameter space for general $g_{u,d,s}^{V,A}$ can be accomplished
using the generalized amplitudes given in ```three_body_amplitudes.py```.

Further information can be found in [arXiv:2602.19479](https://arxiv.org/abs/2602.19479).


### Kaon factory source references
Digitized data from kaon factories are kept in ```data/NA48``` and ```data/NA62``` and were used in deriving the associated limits on the new physics couplings. The source references and plots can be found below:

1. KTeV Collaboration, A. Alavi-Harati et al., “Search for the $K_L \to \pi^0 \pi^0 e^+ e^-$ Decay in the KTeV Experiment,” Phys. Rev. Lett. 89 (2002) 211801, [arXiv:hep-ex/0210056](https://arxiv.org/abs/hep-ex/0210056).
2. NA48/2 Collaboration, J. R. Batley et al., "First observation and study of the $K^{\pm} \to \pi^{\pm} \pi^0 e^+ e^-$ decay," Phys. Lett. B 788 (2019) 552-561, [arXiv:1809.02873](https://arxiv.org/abs/1809.02873).
3. NA48 Collaboration, A. Lai et al., "Investigation of $K_{L,S} \to \pi^+ \pi^- e^+ e^-$ decays," Eur. Phys. J. C 30 (2003) 33-49.
4. NA48/2 Collaboration, J. R. Batley et al., "First Observation and Measurement of the Decay $K^{\pm} \to \pi^{\pm} e^+ e^- \gamma$," Phys. Lett. B 659 (2008) 493-499, [arXiv:0711.4313](https://arxiv.org/abs/0711.4313).
5. NA62 Collaboration, E. Cortina Gil et al., "Search for $K^+$ decays into the $\pi^+ e^+ e^- e^+ e^-$ final state," Phys. Lett. B 846 (2023) 138193, [arXiv:2307.04579](https://arxiv.org/abs/2307.04579).
6. NA48/2 Collaboration, J. R. Batley et al., "Precise measurement of the $K^{\pm} \to \pi^{\pm} e^+ e^-$ decay," Phys. Lett. B 677 (2009) 246-254, [arXiv:0903.3130](https://arxiv.org/abs/0903.3130).


##### Contact
a.thompson@northwestern.edu