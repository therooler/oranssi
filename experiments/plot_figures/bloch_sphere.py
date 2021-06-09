import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from oranssi.plot_utils import plot_bloch_sphere_2d, spherical_to_state, state_to_spherical
from oranssi.utils import get_su_2_operators, save_path_creator
from oranssi.optimizers import local_custom_su_lie_optimizer

nqubits = 1
dev = qml.device('default.qubit', wires=nqubits)


def ps_optimization(eta, nsteps):
    @qml.qnode(dev)
    def circuit_obs(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.RZ(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliX(0))

    @qml.qnode(dev)
    def circuit_state(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.RZ(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.state()

    # Grad shift
    def cost_gradients(params):
        return circuit_obs(params)

    params = [0.4, 1.2]

    gradients = qml.grad(cost_gradients)
    costs_optimizer = []
    states = []
    for step in range(nsteps):
        params = [params[i] - eta * gradients(params)[i] for i in range(2)]
        costs_optimizer.append(circuit_obs(params))
        states.append(circuit_state(params))
        if step > 2:
            if np.isclose(costs_optimizer[-1], costs_optimizer[-2], atol=1e-3):
                break

    phi_optimizer = []
    theta_optimizer = []
    print(states[0])
    for state in states:
        rho_state = np.outer(state, state.conj())
        p, t = state_to_spherical(rho_state)
        phi_optimizer.append(p)
        theta_optimizer.append(t)
    return phi_optimizer, theta_optimizer, costs_optimizer


def riemann_optimization(eta, nsteps):
    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.RZ(params[0], wires=0)
        qml.RY(-params[1], wires=0)
        return qml.state()

    paulis = get_su_2_operators()
    observables = [qml.PauliX(0)]
    params = [0.4, 1.2]

    def cost_function(rho):
        return np.trace(paulis[0] @ rho)

    costs_optimizer, states = local_custom_su_lie_optimizer(circuit, params, observables, dev, eta=eta,
                                                            layer_patern=[(1, 0)], return_state=True, nsteps=nsteps)
    phi_optimizer = []
    theta_optimizer = []
    costs_converted = []
    print(states[0])

    for state in states:
        rho_state = np.outer(state, state.conj())
        costs_converted.append(cost_function(rho_state))
        p, t = state_to_spherical(rho_state)
        phi_optimizer.append(p)
        theta_optimizer.append(t)

    return phi_optimizer, theta_optimizer, costs_optimizer


def plot():
    save_path = save_path_creator('./figures', '')

    gran = 20
    nsteps = 100
    eta = 0.1

    phi = np.linspace(0, 2 * np.pi, gran)
    theta = np.linspace(0, np.pi, gran)
    paulis = get_su_2_operators()

    # Bloch sphere 2D plot
    fig, ax = plt.subplots(1, 1)

    def cost_function(rho):
        return np.trace(paulis[0] @ rho)

    phi_phi, theta_theta = np.meshgrid(phi, theta)
    costs = []
    for p, t in zip(phi_phi.flatten(), theta_theta.flatten()):
        costs.append(cost_function(spherical_to_state(p, t)).real)
    costs_bloch = np.array(costs).reshape((gran, gran))
    plot_bloch_sphere_2d(phi, theta, costs_bloch, ax, fig)

    # get optimizers
    phi_riemann, theta_riemann, costs_riemann = riemann_optimization(eta, nsteps)
    phi_ps, theta_ps, costs_ps = ps_optimization(eta, nsteps)

    # plot s
    ax.plot(phi_riemann, theta_riemann, color='black', linewidth=2, label=fr'Riemann $ \eta = $ {eta:1.3f}')
    ax.plot(phi_ps, theta_ps, color='darkgray', linewidth=2, label=fr'PS $\eta = $ {eta:1.3f}', linestyle='--')
    plt.legend()
    ax.set_title(r'Bloch sphere trajectories')
    ax.plot(phi_riemann[0], theta_riemann[0], 'o', color='black')
    ax.plot(phi_riemann[-1], theta_riemann[-1], 'o', color='black')
    ax.annotate(r'$k = $ 0',
                xy=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2), xycoords='data',
                xytext=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2),
                horizontalalignment='right', verticalalignment='top')
    ax.annotate(rf'$k = ${len(costs_riemann)}',
                xy=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2), xycoords='data',
                xytext=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2),
                horizontalalignment='right', verticalalignment='top')
    fig.savefig(save_path + 'bloch_sphere_optimizer.pdf')
    plt.show()


if __name__ == '__main__':
    plot()
