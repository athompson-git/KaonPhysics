# KaonPhysics
## Emission of vector and axial-vectors in kaon decays via the $\Delta s = 1$ ChPT

Based on work in collaboration with M. Hostert, M. Pospelov, A. Thompson

Requirements:

matplotlib 3.10.8

numpy 2.3.5

scipy 1.16.3

## Computing the 3-body and 2-body amplitudes and monte carlo
* ```monte_carlo.py``` for helper functions
* ```three_body_amplitudes``` for matrix elements and analytic decay widths

## Computing limits from NA48/2, NA62, and KTeV data
* KPlus_to_gamma_X_piplus.ipynb: uses NA48/2 for $K^+ \to \pi^+ \gamma X$ analysis
* KtoX_param_space.ipynb: combined all analysis channels and sensitivity plot