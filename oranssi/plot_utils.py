import matplotlib.pyplot as plt
import numpy as np
from oranssi.utils import get_su_2_operators, get_su_n_operators
from oranssi.circuit_tools import get_all_su_n_directions

plt.rc('font', family='serif')
LABELSIZE=15
LINEWIDTH = 3
MARKERSIZE = 10
blues = plt.get_cmap('Blues')
reds = plt.get_cmap('Reds')

def spherical_to_state(phi, theta):
    """
    Create single qubit state living on the Bloch sphere from the angles `phi` and `theta`

    Args:
        phi: Polar angle on Bloch sphere
        theta: Azimuthal angle on Bloch sphere

    Returns:
        2x2 density matrix array

    """
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    return np.outer(psi, psi.conj())


def plot_bloch_sphere_2d(phi: np.ndarray, theta: np.ndarray, costs: np.ndarray, ax, fig):
    """
    Given phi and theta, which describe points on a sphere of radius one, plot the cost landscape
    on the sphere

    Args:
        phi: Vector with `K` polar angles
        theta: Vector with `L` azimuthal angles
        costs: K x L matrix with costs at points (phi_k, theta_l)
        ax: Matplotlib axes
        fig: Matplotlib figure

    Returns:
        inplace

    """
    im = ax.contourf(phi, theta, costs, cmap='Blues')
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    fig.colorbar(im, ax=ax)


def state_to_spherical(rho):
    """
    Returns the coordinates on the Bloch sphere given a quantum state.

    Args:
        rho: 2x2 density matrix.

    Returns:
        phi, theta, the polar and azimuthal angles on the Bloch sphere.

    """
    assert np.isclose(np.trace(rho), 1), '`rho` is not trace 1'
    paulis = get_su_2_operators()
    coeffs = [np.trace(rho @ paulis[i]).real for i in range(3)]
    theta = np.arccos(coeffs[2])
    phi = np.arctan2(coeffs[1], coeffs[0])
    phi %= (2 * np.pi)
    theta %= (2 * np.pi)
    if np.isclose(phi, 2*np.pi, atol=1e-3):
        phi = 0.
    if np.isclose(theta, 2*np.pi, atol=1e-3):
        theta = 0
    return phi, theta


def change_label_fontsize(ax: plt.Axes, newsize: int):
    """
    Change all labels on the axes to the new size.

    Args:
        *ax (plt.Axes)*:
            Pyplot axes object.

        *newsize (int)*:
            New label size.

    Returns (inplace):
        None

    """
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(newsize)


def plot_su16_directions_separate(nqubits:int, unitaries, observables, device):
    omegas = []
    for uni in unitaries:
        omegas.append(get_all_su_n_directions(uni, observables, nqubits, device))

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(16, 16)

    keys = list(omegas[0].keys())

    axs[0, 0].set_prop_cycle('color', plt.cm.Blues(
        np.linspace(0, 1, sum(1 if k.count('I') == 3 else 0 for k in keys) + 1)))
    axs[0, 1].set_prop_cycle('color', plt.cm.Reds(
        np.linspace(0, 1, sum(1 if k.count('I') == 2 else 0 for k in keys) + 1)))
    axs[1, 0].set_prop_cycle('color', plt.cm.Greens(
        np.linspace(0, 1, sum(1 if k.count('I') == 1 else 0 for k in keys) + 1)))
    axs[1, 1].set_prop_cycle('color', plt.cm.Purples(
        np.linspace(0, 1, sum(1 if k.count('I') == 0 else 0 for k in keys) + 1)))

    stepstotal = len([om['XXXX'] for om in omegas])
    omegas_length_m = {0: np.zeros(stepstotal), 1: np.zeros(stepstotal), 2: np.zeros(stepstotal),
                       3: np.zeros(stepstotal)}

    for k in omegas[0].keys():
        if k.count('I') == 3:
            omegas_length_m[3] += np.abs([om[k] for om in omegas])
            axs[0, 0].plot([om[k] for om in omegas], label=k)
        if k.count('I') == 2:
            omegas_length_m[2] += np.abs([om[k] for om in omegas])
            axs[0, 1].plot([om[k] for om in omegas], label=k)
        if k.count('I') == 1:
            omegas_length_m[1] += np.abs([om[k] for om in omegas])
            axs[1, 0].plot([om[k] for om in omegas], label=k)
        if k.count('I') == 0:
            omegas_length_m[0] += np.abs([om[k] for om in omegas])
            axs[1, 1].plot([om[k] for om in omegas], label=k)
    for a in axs.flatten():
        change_label_fontsize(a, 15)
        a.legend()
        a.set_xlabel('Step')
        a.set_ylabel(r'$\omega_i$')
        a.legend()
    return fig, axs

def plot_su16_directions(nqubits:int, unitaries, observables, device):
    omegas = []
    for uni in unitaries:
        omegas.append(get_all_su_n_directions(uni, observables, device))

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)

    stepstotal = len([om['XXXX'] for om in omegas])
    omegas_length_m = {0: np.zeros(stepstotal), 1: np.zeros(stepstotal), 2: np.zeros(stepstotal),
                       3: np.zeros(stepstotal)}

    for k in omegas[0].keys():
        if k.count('I') == 3:
            omegas_length_m[3] += np.abs([om[k] for om in omegas])
        if k.count('I') == 2:
            omegas_length_m[2] += np.abs([om[k] for om in omegas])
        if k.count('I') == 1:
            omegas_length_m[1] += np.abs([om[k] for om in omegas])
        if k.count('I') == 0:
            omegas_length_m[0] += np.abs([om[k] for om in omegas])

    axs.plot([0 for _ in range(len(omegas_length_m[0]))], label='Min.', color='Black',
             linestyle='--')

    cmap = plt.get_cmap('PuOr')
    for i in range(4):
        axs.plot(omegas_length_m[i], label=rf'SU$({2 ** (4 - i)})$', color=cmap((i + 1) / 5),
                 linewidth=LINEWIDTH)
        axs.set_xlabel('Step')
        axs.set_ylabel(r'$\sum_i |\omega_i|$')
    axs.legend()
    change_label_fontsize(axs, LABELSIZE)
    return fig, axs


def plot_su8_directions_individually(unitaries, observables, device):
    omegas = []
    for uni in unitaries:
        omegas.append(get_all_su_n_directions(uni, observables, device))

    su4_names = []
    su8_names = []
    print(su8_names)
    stepstotal = len([om['XX'] for om in omegas])
    omegas_length_m = {0: np.zeros(stepstotal), 1: np.zeros(stepstotal),}
    omegas_su4 = []
    omegas_su8 = []
    for k in omegas[0].keys():
        if k.count('I') == 1:
            su4_names.append(k)
            omegas_length_m[1] += np.abs([om[k] for om in omegas])
            omegas_su4.append([om[k] for om in omegas])
        if k.count('I') == 0:
            su8_names.append(k)
            omegas_length_m[0] += np.abs([om[k] for om in omegas])
            omegas_su8.append([om[k] for om in omegas])
    omegas_su4 = np.array(omegas_su4)
    omegas_su8 = np.array(omegas_su8)
    fig, axs = plt.subplots(1, 1)

    for i,om in enumerate(omegas_su4):
        if not np.allclose(om, 0):
            axs.plot(om, label=''.join(list(su4_names[i])),  linewidth=LINEWIDTH)

    for i,om in enumerate(omegas_su8):
        if not np.allclose(om,0):
            axs.plot(om, label=''.join(list(su8_names[i])),  linewidth=LINEWIDTH)
    axs.legend()

    change_label_fontsize(axs, LABELSIZE)
    return fig, axs


def plot_su8_directions(nqubits:int, unitaries, observables, device):
    omegas = []
    for uni in unitaries:
        omegas.append(get_all_su_n_directions(uni, observables, device))

    su4_names = []
    su8_names = []
    print(su8_names)
    stepstotal = len([om['XX'] for om in omegas])
    omegas_length_m = {0: np.zeros(stepstotal), 1: np.zeros(stepstotal),}
    omegas_su4 = []
    omegas_su8 = []
    for k in omegas[0].keys():
        if k.count('I') == 1:
            su4_names.append(k)
            omegas_length_m[1] += np.abs([om[k] for om in omegas])
            omegas_su4.append([om[k] for om in omegas])
        if k.count('I') == 0:
            su8_names.append(k)
            omegas_length_m[0] += np.abs([om[k] for om in omegas])
            omegas_su8.append([om[k] for om in omegas])
    omegas_su4 = np.array(omegas_su4)
    omegas_su8 = np.array(omegas_su8)
    for i,om in enumerate(omegas_su4):
        if not np.allclose(om, 0):
            plt.plot(om, label=''.join(list(su4_names[i])))

    for i,om in enumerate(omegas_su8):
        if not np.allclose(om,0):
            plt.plot(om, label=''.join(list(su8_names[i])))
    plt.legend()
    plt.show()
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    axs.plot([0 for _ in range(len(omegas_length_m[0]))], label='Min.', color='Black',
             linestyle='--')

    cmap = plt.get_cmap('Set1')
    for i in range(2):
        axs.plot(omegas_length_m[i], label=rf'SU$({2 ** (2 - i)})$', color=cmap((i + 1) / 5),
                 linewidth=LINEWIDTH)
        axs.set_xlabel('Step')
        axs.set_ylabel(r'$\sum_i |\omega_i|$')
    axs.legend()
    change_label_fontsize(axs, LABELSIZE)
    return fig, axs