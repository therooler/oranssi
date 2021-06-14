import matplotlib.pyplot as plt
import numpy as np
from oranssi.utils import get_su_2_operators

plt.rc('font', family='serif')


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
