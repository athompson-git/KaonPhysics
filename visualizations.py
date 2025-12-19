import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from mpl_toolkits.mplot3d import Axes3D

# draw random variates for axial vector couplings
def mc_planes():
    n_pts = 1000000

    gu_list = []
    gd_list = []
    gs_list = []

    # constraints
    for i in range(n_pts):
        gu = np.random.uniform(-1e-3, 1e-3, 1)
        gd = np.random.uniform(-1e-3, 1e-3, 1)
        gs = np.random.uniform(-1e-3, 1e-3, 1)

        # KL to pi0 pi0 X
        if abs(gd + gs/3) > 1.73e-5:
            continue

        # K+ to pi+ pi0 X
        if abs(2*gu + gd + gs) > 1.2e-4:
            continue

        # KL to pi+ pi- X
        #if abs(-2*gu + 5*gd + gs) > 1.2e-4:
        #    continue

        gu_list.append(gu)
        gd_list.append(gd)
        gs_list.append(gs)


    gu_list = np.array(gu_list)
    gd_list = np.array(gd_list)
    gs_list = np.array(gs_list)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    ax.scatter((gu_list), (gd_list), (gs_list), marker=".", color='k')

    # projections
    ax.scatter(gu_list, 1e-3*np.ones_like(gu_list), gs_list, marker='+', color='gainsboro')
    ax.scatter(-1e-3*np.ones_like(gs_list), gd_list, gs_list, marker='+', color='gainsboro')
    ax.scatter(gu_list, gd_list, -1e-3*np.ones_like(gs_list), marker='+', color='gainsboro')

    ax.set_xlabel(r"$g_u^A$")
    ax.set_ylabel(r"$g_d^A$")
    ax.set_zlabel(r"$g_s^A$")

    ax.set_xlim((-1e-3,1e-3))
    ax.set_ylim((-1e-3,1e-3))
    ax.set_zlim((-1e-3,1e-3))
    plt.show()


def analytic_planes():


    # Define ranges for gu, gd, gs
    gu = np.linspace(-2e-4, 2e-4, 50)
    gd = np.linspace(-2e-4, 2e-4, 50)
    gu, gd = np.meshgrid(gu, gd)

    # Compute gs for each plane
    gs_KL_pi0pi0 = -3 * gd  # gd + gs/3 = 0
    gs_Kp_pipi0 = -(2 * gu + gd)  # 2gu + gd + gs = 0
    gs_KL_pipm = 2 * gu - 5 * gd  # -2gu + 5gd + gs = 0

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the planes (gray, semi-transparent)
    ax.plot_surface(gu, gd, gs_KL_pi0pi0, color='b', alpha=1.0, rstride=1, cstride=1, linewidth=0)
    ax.plot_surface(gu, gd, gs_Kp_pipi0, color='r', alpha=1.0, rstride=1, cstride=1, linewidth=0)
    ax.plot_surface(gu, gd, gs_KL_pipm, color='g', alpha=1.0, rstride=1, cstride=1, linewidth=0)

    # Define bounds
    cond1 = np.abs(gd + gs_KL_pi0pi0 / 3) < 1.73e-5
    cond2 = np.abs(2 * gu + gd + gs_Kp_pipi0) < 1.2e-4
    cond3 = np.abs(-2 * gu + 5 * gd + gs_KL_pipm) < 1.2e-4
    intersect_mask = cond1 & cond2 & cond3

    # Compute approximate intersection volume (as gs value)
    gs_intersect = np.mean([gs_KL_pi0pi0, gs_Kp_pipi0, gs_KL_pipm], axis=0)

    # Plot intersection volume region in color
    ax.plot_surface(gu, gd, gs_intersect, facecolors=np.where(intersect_mask[..., None],
                                                              [0, 0.5, 1, 0.8], [0, 0, 0, 0]),
                                                              rstride=1, cstride=1, linewidth=0)

    # Labels
    ax.set_xlabel("$g_u$")
    ax.set_ylabel("$g_d$")
    ax.set_zlabel("$g_s$")
    ax.set_title("Intersection of decay constraints in $(g_u, g_d, g_s)$ space")


    plt.tight_layout()
    plt.show()



def main():
    #analytic_planes()

    mc_planes()


if __name__ == "__main__":
    main()