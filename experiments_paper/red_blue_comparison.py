import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import copy
from oranssi.plot_utils import plot_bloch_sphere_2d, spherical_to_state, state_to_spherical, \
    change_label_fontsize, LABELSIZE, MARKERSIZE, LINEWIDTH, reds, blues
from oranssi.utils import get_su_2_operators, save_path_creator
from oranssi.optimizers import exact_lie_optimizer, parameter_shift_optimizer

nqubits = 1
dev = qml.device('default.qubit', wires=nqubits)
plt.rc('font', family='serif')


def plot_difference_2_obs_sep_figs():
    def ps_optimization(eta, nsteps, params):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(0)]

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        costs_optimizer, states, params = parameter_shift_optimizer(circuit, params, observables,
                                                                    dev,
                                                                    eta=eta, return_state=True,
                                                                    return_params=True,
                                                                    nsteps=nsteps)
        phi_optimizer = []
        theta_optimizer = []
        costs_converted = []

        for state in states:
            rho_state = np.outer(state, state.conj())
            costs_converted.append(cost_function(rho_state))
            p, t = state_to_spherical(rho_state)
            phi_optimizer.append(p)
            theta_optimizer.append(t)
        params = np.array(params) % (2 * np.pi)
        return phi_optimizer, theta_optimizer, params[:, 0], params[:, 1], costs_optimizer

    def riemann_optimization(eta, nsteps, params):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)
            return qml.state()

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(0)]

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        costs_optimizer, states = exact_lie_optimizer(circuit, params, observables, dev,
                                                      eta=eta,
                                                      return_state=True, nsteps=nsteps)
        phi_optimizer = []
        theta_optimizer = []
        costs_converted = []

        for state in states:
            rho_state = np.outer(state, state.conj())
            costs_converted.append(cost_function(rho_state))
            p, t = state_to_spherical(rho_state)
            phi_optimizer.append(p)
            theta_optimizer.append(t)

        return phi_optimizer, theta_optimizer, costs_optimizer

    def plot():
        save_path = save_path_creator('./figures', '')

        gran = 25
        nsteps = 100
        eta = 0.05

        phi = np.linspace(0, 2 * np.pi, gran)
        theta = np.linspace(0, np.pi, gran)
        paulis = get_su_2_operators()

        # Bloch sphere 2D plot
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(8, 4)

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        phi_phi, theta_theta = np.meshgrid(phi, theta)
        costs = []
        for p, t in zip(phi_phi.flatten(), theta_theta.flatten()):
            costs.append(cost_function(spherical_to_state(p, t)).real)
        costs_bloch = np.array(costs).reshape((gran, gran))
        plot_bloch_sphere_2d(phi, theta, costs_bloch, axs, fig)
        params = [np.pi / 3, np.pi / 5]
        # get optimizers
        phi_riemann, theta_riemann, costs_riemann = riemann_optimization(eta, nsteps, params)
        phi_ps, theta_ps, param_phi_ps, param_theta_ps, costs_ps = ps_optimization(eta, nsteps,
                                                                                   params)

        # plot s
        axs.plot(phi_riemann, theta_riemann, color='black', linewidth=LINEWIDTH,
                 label=fr'Riemann', markersize=MARKERSIZE)
        axs.plot(phi_ps, theta_ps, color='black', linewidth=LINEWIDTH, linestyle='--',
                 label=fr'Param. Shift.', markersize=MARKERSIZE)
        axs.legend(prop={'size': 18})
        # axs.set_title(r'Bloch sphere trajectories')
        axs.plot(phi_riemann[0], theta_riemann[0], 'o', color='black')
        axs.plot(phi_riemann[-1], theta_riemann[-1], 'o', color='black')
        axs.annotate(r'$k = $ 0',
                     xy=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2), xycoords='data',
                     xytext=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2),
                     horizontalalignment='right', verticalalignment='top', fontsize=18)
        axs.annotate(rf'$k = ${len(costs_riemann)}',
                     xy=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2), xycoords='data',
                     xytext=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2),
                     horizontalalignment='right', verticalalignment='top', fontsize=18)
        change_label_fontsize(axs, 18)
        axs.set_aspect('equal')
        fig.savefig(save_path + 'bloch_sphere_difference_optimizer_2_obs_blue.pdf')

        def circuit_cost(params, **kwargs):
            a, b = params
            qml.RY(a, wires=0)
            qml.RZ(b, wires=0)

        observables = [qml.PauliX(0), qml.PauliY(0)]
        H = qml.Hamiltonian([1.0 for _ in range(len(observables))], observables)
        circuit_cost = qml.ExpvalCost(circuit_cost, H, dev)

        @qml.qnode(dev)
        def circuit_state(a, b, **kwargs):
            qml.RY(a, wires=0)
            qml.RZ(b, wires=0)
            return qml.state()

        alpha = np.linspace(0, 2 * np.pi, gran)
        beta = np.linspace(0, 2 * np.pi, gran)
        alpha_alpha, beta_beta = np.meshgrid(alpha, beta)
        costs = []
        for a, b in zip(alpha_alpha.flatten(), beta_beta.flatten()):
            costs.append(circuit_cost([a, b]))
        costs_bloch = np.array(costs).reshape((gran, gran))

        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        im = axs.contourf(alpha, beta, costs_bloch, cmap='Reds')
        axs.set_xlabel(r'$\alpha$')
        axs.set_ylabel(r'$\beta$')
        fig.colorbar(im, ax=axs)
        # plot s
        axs.plot(param_phi_ps, param_theta_ps, color='black', linewidth=LINEWIDTH,
                 label=fr'Param. Shift.')
        axs.legend(prop={'size': 18})
        # axs.set_title(r'Bloch sphere trajectories')
        axs.plot(param_phi_ps[0], param_theta_ps[0], 'o', color='black', markersize=MARKERSIZE)
        axs.plot(param_phi_ps[-1], param_theta_ps[-1], 'o', color='black', markersize=MARKERSIZE)
        axs.annotate(r'$k = $ 0',
                     xy=(param_phi_ps[0] + 0.2, param_theta_ps[0] + 0.4), xycoords='data',
                     xytext=(param_phi_ps[0] + 0.2, param_theta_ps[0] + 0.4),
                     horizontalalignment='right', verticalalignment='top', fontsize=18)
        axs.annotate(rf'$k = ${len(costs_ps)}',
                     xy=(param_phi_ps[-1] + 0.2, param_theta_ps[-1] + 0.4), xycoords='data',
                     xytext=(param_phi_ps[-1] + 0.2, param_theta_ps[-1] + 0.4),
                     horizontalalignment='right', verticalalignment='top', fontsize=18)
        change_label_fontsize(axs, LABELSIZE)
        fig.savefig(save_path + 'bloch_sphere_difference_optimizer_2_obs_red.pdf')
        plt.show()

    plot()


if __name__ == '__main__':
    plot_difference_2_obs_sep_figs()
