import matplotlib.pyplot as plt
import numpy as np
from oranssi.utils import get_su_2_operators

plt.rc('font', family='serif')


def spherical_to_state(phi, theta):
    phi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    return np.outer(phi, phi.conj())


def plot_bloch_sphere_2d(phi: np.ndarray, theta: np.ndarray, costs: np.ndarray, ax, fig):
    im = ax.contourf(phi, theta, costs, cmap='Blues')
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    fig.colorbar(im, ax=ax)


def state_to_spherical(rho):
    assert np.isclose(np.trace(rho), 1), '`rho` is not trace 1'
    paulis = get_su_2_operators()
    coeffs = [np.trace(rho @ paulis[i]).real for i in range(3)]
    theta = np.arccos(coeffs[2])
    phi = np.arctan2(coeffs[1], coeffs[0])
    return phi, theta
