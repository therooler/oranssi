import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import copy
from oranssi.plot_utils import plot_bloch_sphere_2d, spherical_to_state, state_to_spherical, \
    change_label_fontsize
from oranssi.utils import get_su_2_operators, save_path_creator
from oranssi.optimizers import exact_lie_optimizer, parameter_shift_optimizer

nqubits = 1
dev = qml.device('default.qubit', wires=nqubits)
plt.rc('font', family='serif')


def plot_equal():
    def ps_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0)]
        params = [0.4, 1.2]

        def cost_function(rho):
            return np.trace(paulis[0] @ rho)

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
        return params[:, 0], params[:, 1], costs_optimizer

    def riemann_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            return qml.state()

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0)]
        params = [0.4, 1.2]

        def cost_function(rho):
            return np.trace(paulis[0] @ rho)

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

        gran = 20
        nsteps = 100
        eta = 0.1

        phi = np.linspace(0, 2 * np.pi, gran)
        theta = np.linspace(0, np.pi, gran)
        paulis = get_su_2_operators()

        # Bloch sphere 2D plot
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18, 8)

        def cost_function(rho):
            return np.trace(paulis[0] @ rho)

        phi_phi, theta_theta = np.meshgrid(phi, theta)
        costs = []
        for p, t in zip(phi_phi.flatten(), theta_theta.flatten()):
            costs.append(cost_function(spherical_to_state(p, t)).real)
        costs_bloch = np.array(costs).reshape((gran, gran))
        plot_bloch_sphere_2d(phi, theta, costs_bloch, axs[0], fig)

        # get optimizers
        phi_riemann, theta_riemann, costs_riemann = riemann_optimization(eta, nsteps)
        phi_ps, theta_ps, costs_ps = ps_optimization(eta, nsteps)

        # plot s
        axs[0].plot(phi_riemann, theta_riemann, color='black', linewidth=2,
                    label=fr'Riemann $ \eta = $ {eta:1.3f}')
        axs[0].legend()
        axs[0].set_title(r'Bloch sphere trajectories')
        axs[0].plot(phi_riemann[0], theta_riemann[0], 'o', color='black')
        axs[0].plot(phi_riemann[-1], theta_riemann[-1], 'o', color='black')
        axs[0].annotate(r'$k = $ 0',
                        xy=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2), xycoords='data',
                        xytext=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        axs[0].annotate(rf'$k = ${len(costs_riemann)}',
                        xy=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2), xycoords='data',
                        xytext=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2),
                        horizontalalignment='right', verticalalignment='top')

        def circuit_cost(params, **kwargs):
            a, b = params
            qml.RY(a, wires=0)

        observables = [qml.PauliX(0)]
        H = qml.Hamiltonian([1.0 for _ in range(len(observables))], observables)
        circuit_cost = qml.ExpvalCost(circuit_cost, H, dev)

        @qml.qnode(dev)
        def circuit_state(a, b, **kwargs):
            qml.RY(a, wires=0)
            return qml.state()

        alpha = np.linspace(0, 2 * np.pi, gran)
        beta = np.linspace(0, 2 * np.pi, gran)
        alpha_alpha, beta_beta = np.meshgrid(alpha, beta)
        costs = []
        for a, b in zip(alpha_alpha.flatten(), beta_beta.flatten()):
            costs.append(circuit_cost([a, b]))
        costs_bloch = np.array(costs).reshape((gran, gran))

        im = axs[1].contourf(alpha, beta, costs_bloch, cmap='Reds')
        axs[1].set_xlabel(r'$\alpha$')
        axs[1].set_ylabel(r'$\beta$')
        fig.colorbar(im, ax=axs[1])
        # plot s
        axs[1].plot(phi_ps, theta_ps, color='black', linewidth=2, label=fr'PS $\eta = $ {eta:1.3f}')
        axs[1].legend()
        axs[1].set_title(r'Bloch sphere trajectories')
        axs[1].plot(phi_ps[0], theta_ps[0], 'o', color='black')
        axs[1].plot(phi_ps[-1], theta_ps[-1], 'o', color='black')
        axs[1].annotate(r'$k = $ 0',
                        xy=(phi_ps[0] + 0.2, theta_ps[0] + 0.2), xycoords='data',
                        xytext=(phi_ps[0] + 0.2, theta_ps[0] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        axs[1].annotate(rf'$k = ${len(costs_ps)}',
                        xy=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2), xycoords='data',
                        xytext=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        for ax in axs:
            change_label_fontsize(ax, 15)
        fig.savefig(save_path + 'bloch_sphere_equal_optimizer.pdf')
        plt.show()

    plot()


def plot_difference():
    def ps_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0)]
        params = [0.4, 1.2]

        def cost_function(rho):
            return np.trace(paulis[0] @ rho)

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
        return params[:, 0], params[:, 1], costs_optimizer

    def riemann_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)
            return qml.state()

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0)]
        params = [0.4, 1.2]

        def cost_function(rho):
            return np.trace(paulis[0] @ rho)

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

        gran = 20
        nsteps = 100
        eta = 0.1

        phi = np.linspace(0, 2 * np.pi, gran)
        theta = np.linspace(0, np.pi, gran)
        paulis = get_su_2_operators()

        # Bloch sphere 2D plot
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18, 8)

        def cost_function(rho):
            return np.trace(paulis[0] @ rho)

        phi_phi, theta_theta = np.meshgrid(phi, theta)
        costs = []
        for p, t in zip(phi_phi.flatten(), theta_theta.flatten()):
            costs.append(cost_function(spherical_to_state(p, t)).real)
        costs_bloch = np.array(costs).reshape((gran, gran))
        plot_bloch_sphere_2d(phi, theta, costs_bloch, axs[0], fig)

        # get optimizers
        phi_riemann, theta_riemann, costs_riemann = riemann_optimization(eta, nsteps)
        phi_ps, theta_ps, costs_ps = ps_optimization(eta, nsteps)

        # plot s
        axs[0].plot(phi_riemann, theta_riemann, color='black', linewidth=2,
                    label=fr'Riemann $ \eta = $ {eta:1.3f}')
        axs[0].legend()
        axs[0].set_title(r'Bloch sphere trajectories')
        axs[0].plot(phi_riemann[0], theta_riemann[0], 'o', color='black')
        axs[0].plot(phi_riemann[-1], theta_riemann[-1], 'o', color='black')
        axs[0].annotate(r'$k = $ 0',
                        xy=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2), xycoords='data',
                        xytext=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        axs[0].annotate(rf'$k = ${len(costs_riemann)}',
                        xy=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2), xycoords='data',
                        xytext=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2),
                        horizontalalignment='right', verticalalignment='top')

        def circuit_cost(params, **kwargs):
            a, b = params
            qml.RY(a, wires=0)
            qml.RZ(b, wires=0)

        observables = [qml.PauliX(0)]
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

        im = axs[1].contourf(alpha, beta, costs_bloch, cmap='Reds')
        axs[1].set_xlabel(r'$\alpha$')
        axs[1].set_ylabel(r'$\beta$')
        fig.colorbar(im, ax=axs[1])
        # plot s
        axs[1].plot(phi_ps, theta_ps, color='black', linewidth=2, label=fr'PS $\eta = $ {eta:1.3f}')
        axs[1].legend()
        axs[1].set_title(r'Bloch sphere trajectories')
        axs[1].plot(phi_ps[0], theta_ps[0], 'o', color='black')
        axs[1].plot(phi_ps[-1], theta_ps[-1], 'o', color='black')
        axs[1].annotate(r'$k = $ 0',
                        xy=(phi_ps[0] + 0.2, theta_ps[0] + 0.2), xycoords='data',
                        xytext=(phi_ps[0] + 0.2, theta_ps[0] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        axs[1].annotate(rf'$k = ${len(costs_ps)}',
                        xy=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2), xycoords='data',
                        xytext=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        for ax in axs:
            change_label_fontsize(ax, 15)
        fig.savefig(save_path + 'bloch_sphere_difference_optimizer.pdf')
        plt.show()

    plot()


def plot_difference_2_obs():
    def ps_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(0)]
        params = [0.4, 1.2]

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
        return params[:, 0], params[:, 1], costs_optimizer

    def riemann_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)
            return qml.state()

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(0)]
        params = [0.4, 1.2]

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

        gran = 20
        nsteps = 100
        eta = 0.1

        phi = np.linspace(0, 2 * np.pi, gran)
        theta = np.linspace(0, np.pi, gran)
        paulis = get_su_2_operators()

        # Bloch sphere 2D plot
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18, 8)

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        phi_phi, theta_theta = np.meshgrid(phi, theta)
        costs = []
        for p, t in zip(phi_phi.flatten(), theta_theta.flatten()):
            costs.append(cost_function(spherical_to_state(p, t)).real)
        costs_bloch = np.array(costs).reshape((gran, gran))
        plot_bloch_sphere_2d(phi, theta, costs_bloch, axs[0], fig)

        # get optimizers
        phi_riemann, theta_riemann, costs_riemann = riemann_optimization(eta, nsteps)
        phi_ps, theta_ps, costs_ps = ps_optimization(eta, nsteps)

        # plot s
        axs[0].plot(phi_riemann, theta_riemann, color='black', linewidth=2,
                    label=fr'Riemann $ \eta = $ {eta:1.3f}')
        axs[0].legend()
        axs[0].set_title(r'Bloch sphere trajectories')
        axs[0].plot(phi_riemann[0], theta_riemann[0], 'o', color='black')
        axs[0].plot(phi_riemann[-1], theta_riemann[-1], 'o', color='black')
        axs[0].annotate(r'$k = $ 0',
                        xy=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2), xycoords='data',
                        xytext=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        axs[0].annotate(rf'$k = ${len(costs_riemann)}',
                        xy=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2), xycoords='data',
                        xytext=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2),
                        horizontalalignment='right', verticalalignment='top')

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

        im = axs[1].contourf(alpha, beta, costs_bloch, cmap='Reds')
        axs[1].set_xlabel(r'$\alpha$')
        axs[1].set_ylabel(r'$\beta$')
        fig.colorbar(im, ax=axs[1])
        # plot s
        axs[1].plot(phi_ps, theta_ps, color='black', linewidth=2, label=fr'PS $\eta = $ {eta:1.3f}')
        axs[1].legend()
        axs[1].set_title(r'Bloch sphere trajectories')
        axs[1].plot(phi_ps[0], theta_ps[0], 'o', color='black')
        axs[1].plot(phi_ps[-1], theta_ps[-1], 'o', color='black')
        axs[1].annotate(r'$k = $ 0',
                        xy=(phi_ps[0] + 0.2, theta_ps[0] + 0.2), xycoords='data',
                        xytext=(phi_ps[0] + 0.2, theta_ps[0] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        axs[1].annotate(rf'$k = ${len(costs_ps)}',
                        xy=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2), xycoords='data',
                        xytext=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2),
                        horizontalalignment='right', verticalalignment='top')
        for ax in axs:
            change_label_fontsize(ax, 15)
        fig.savefig(save_path + 'bloch_sphere_difference_optimizer_2_obs.pdf')
        plt.show()

    plot()


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
        return phi_optimizer,theta_optimizer,params[:, 0], params[:, 1], costs_optimizer

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

        gran = 20
        nsteps = 100
        eta = 0.1

        phi = np.linspace(0, 2 * np.pi, gran)
        theta = np.linspace(0, np.pi, gran)
        paulis = get_su_2_operators()

        # Bloch sphere 2D plot
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        phi_phi, theta_theta = np.meshgrid(phi, theta)
        costs = []
        for p, t in zip(phi_phi.flatten(), theta_theta.flatten()):
            costs.append(cost_function(spherical_to_state(p, t)).real)
        costs_bloch = np.array(costs).reshape((gran, gran))
        plot_bloch_sphere_2d(phi, theta, costs_bloch, axs, fig)
        params = [np.pi/3, np.pi/2]
        # get optimizers
        phi_riemann, theta_riemann, costs_riemann = riemann_optimization(eta, nsteps, params)
        phi_ps, theta_ps, param_phi_ps, param_theta_ps, costs_ps = ps_optimization(eta, nsteps, params)

        # plot s
        axs.plot(phi_riemann, theta_riemann, color='black', linewidth=2,
                    label=fr'Riemann')
        axs.plot(phi_ps, theta_ps, color='black', linewidth=2,linestyle='--',
                 label=fr'Param. Shift.')
        axs.legend(prop={'size':18})
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
        axs.plot(param_phi_ps, param_theta_ps, color='black', linewidth=2, label=fr'Param. Shift.')
        axs.legend(prop={'size':18})
        # axs.set_title(r'Bloch sphere trajectories')
        axs.plot(param_phi_ps[0], param_theta_ps[0], 'o', color='black')
        axs.plot(param_phi_ps[-1], param_theta_ps[-1], 'o', color='black')
        axs.annotate(r'$k = $ 0',
                        xy=(param_phi_ps[0] + 0.2, param_theta_ps[0] + 0.4), xycoords='data',
                        xytext=(param_phi_ps[0] + 0.2, param_theta_ps[0] + 0.4),
                        horizontalalignment='right', verticalalignment='top', fontsize=18)
        axs.annotate(rf'$k = ${len(costs_ps)}',
                        xy=(param_phi_ps[-1] + 0.2, param_theta_ps[-1] + 0.4), xycoords='data',
                        xytext=(param_phi_ps[-1] + 0.2, param_theta_ps[-1] + 0.4),
                        horizontalalignment='right', verticalalignment='top', fontsize=18)
        change_label_fontsize(axs, 18)
        fig.savefig(save_path + 'bloch_sphere_difference_optimizer_2_obs_red.pdf')
        plt.show()

    plot()

def plot_difference_2_obs_qng():
    def ps_optimization(eta, nsteps,params):
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
                                                                    nsteps=nsteps, tol=1e-9)
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
        return states, phi_optimizer, theta_optimizer, costs_optimizer

    def qng_grad_optimization(eta, nsteps, params):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(0)]
        params = np.array([1.68, 3.16])

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        opt = qml.QNGOptimizer(stepsize=eta, diag_approx=False)

        qngd_param_history = [params]
        costs_optimizer = []
        H = qml.Hamiltonian([1.0 for _ in range(len(observables))], observables)
        cost_fn = qml.ExpvalCost(circuit, H, dev)

        @qml.qnode(dev)
        def get_state(params, **kwargs):
            circuit(params)
            return qml.state()

        states = []
        for n in range(nsteps):

            # Take step
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            qngd_param_history.append(params)
            costs_optimizer.append(prev_energy)
            states.append(get_state(params))
            # Compute energy
            energy = cost_fn(params)

            # Calculate difference between new and old energies
            conv = np.abs(energy - prev_energy)

            if conv <= 1e-5:
                break
        phi_optimizer = []
        theta_optimizer = []
        costs_converted = []

        for state in states:
            rho_state = np.outer(state, state.conj())
            costs_converted.append(cost_function(rho_state))
            p, t = state_to_spherical(rho_state)
            phi_optimizer.append(p)
            theta_optimizer.append(t)
        qngd_param_history = np.array(qngd_param_history) % (2 * np.pi)
        return states, phi_optimizer, theta_optimizer, costs_optimizer

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

        return states, phi_optimizer, theta_optimizer, costs_optimizer

    def plot():
        save_path = save_path_creator('./figures', '')

        gran = 40
        nsteps = 500
        eta = 0.01

        phi = np.linspace(0, 2 * np.pi, gran)
        theta = np.linspace(0, np.pi, gran)
        paulis = get_su_2_operators()

        params = [0.4, 1.2]

        # get optimizers
        psi_riemann, phi_riemann, theta_riemann, costs_riemann = riemann_optimization(10*eta, nsteps, copy.copy(params))
        psi_ps, phi_ps, theta_ps, costs_ps = ps_optimization(eta, nsteps,copy.copy(params))
        psi_qng, phi_qng, theta_qng, costs_qng = qng_grad_optimization(eta, nsteps,np.array(params))
        closeness = []
        for state in psi_qng:
            fids = [np.abs(np.vdot(state,s)) for s in psi_riemann]
            idx = np.argmax(fids)
            closeness.append([idx, fids[idx]])

        print(closeness)
        # fig, axs = plt.subplots(1, 1)
        # # plot s
        # axs.plot(np.array(np.abs(psi_riemann)), color='black', linewidth=2,
        #          label=fr'Riemann $ \eta = $ {10 * eta:1.3f}')
        # axs.plot(np.array(np.abs(psi_ps)), color='red', linewidth=2,
        #          label=fr'PS $ \eta = $ {eta:1.3f}')
        # axs.plot(np.array(np.abs(psi_qng)), color='green', linewidth=2,
        #          label=fr'QNG $ \eta = $ {eta:1.3f}')
        # axs.legend()
        # axs.set_title(r'Bloch sphere trajectories')
        #
        # plt.show()
        # Bloch sphere 2D plot
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        phi_phi, theta_theta = np.meshgrid(phi, theta)
        costs = []
        for p, t in zip(phi_phi.flatten(), theta_theta.flatten()):
            costs.append(cost_function(spherical_to_state(p, t)).real)
        costs_bloch = np.array(costs).reshape((gran, gran))
        plot_bloch_sphere_2d(phi, theta, costs_bloch, axs, fig)


        # plot s
        axs.plot(phi_riemann, theta_riemann, color='black', linewidth=2,
                 label=fr'Riemann $ \eta = $ {10*eta:1.3f}')
        axs.plot(phi_ps, theta_ps, color='red', linewidth=2,
                 label=fr'PS $ \eta = $ {eta:1.3f}')
        axs.plot(phi_qng, theta_qng, color='green', linewidth=2,
                 label=fr'QNG $ \eta = $ {eta:1.3f}')
        axs.legend()
        axs.set_title(r'Bloch sphere trajectories')
        axs.plot(phi_riemann[0], theta_riemann[0], 'o', color='black')
        axs.plot(phi_riemann[-1], theta_riemann[-1], 'o', color='black')
        axs.annotate(r'$k = $ 0',
                     xy=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2), xycoords='data',
                     xytext=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2),
                     horizontalalignment='right', verticalalignment='top')
        axs.annotate(rf'$k = ${len(costs_riemann)}',
                     xy=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2), xycoords='data',
                     xytext=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2),
                     horizontalalignment='right', verticalalignment='top')

        # def circuit_cost(params, **kwargs):
        #     a, b = params
        #     qml.RY(a, wires=0)
        #     qml.RZ(b, wires=0)
        #
        # observables = [qml.PauliX(0), qml.PauliY(0)]
        # H = qml.Hamiltonian([1.0 for _ in range(len(observables))], observables)
        # circuit_cost = qml.ExpvalCost(circuit_cost, H, dev)
        #
        # @qml.qnode(dev)
        # def circuit_state(a, b, **kwargs):
        #     qml.RY(a, wires=0)
        #     qml.RZ(b, wires=0)
        #     return qml.state()
        #
        # alpha = np.linspace(0, 2 * np.pi, gran)
        # beta = np.linspace(0, 2 * np.pi, gran)
        # alpha_alpha, beta_beta = np.meshgrid(alpha, beta)
        # costs = []
        # for a, b in zip(alpha_alpha.flatten(), beta_beta.flatten()):
        #     costs.append(circuit_cost([a, b]))
        # costs_bloch = np.array(costs).reshape((gran, gran))
        #
        # im = axs[1].contourf(alpha, beta, costs_bloch, cmap='Reds')
        # axs[1].set_xlabel(r'$\alpha$')
        # axs[1].set_ylabel(r'$\beta$')
        # fig.colorbar(im, ax=axs[1])
        # # plot s
        # axs[1].plot(phi_ps, theta_ps, color='black', linewidth=2, label=fr'PS $\eta = $ {eta:1.3f}')
        # axs[1].legend()
        # axs[1].set_title(r'Bloch sphere trajectories')
        # axs[1].plot(phi_ps[0], theta_ps[0], 'o', color='black')
        # axs[1].plot(phi_ps[-1], theta_ps[-1], 'o', color='black')
        # axs[1].annotate(r'$k = $ 0',
        #                 xy=(phi_ps[0] + 0.2, theta_ps[0] + 0.2), xycoords='data',
        #                 xytext=(phi_ps[0] + 0.2, theta_ps[0] + 0.2),
        #                 horizontalalignment='right', verticalalignment='top')
        # axs[1].annotate(rf'$k = ${len(costs_ps)}',
        #                 xy=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2), xycoords='data',
        #                 xytext=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2),
        #                 horizontalalignment='right', verticalalignment='top')
        # for ax in axs:
        #     change_label_fontsize(ax, 15)
        fig.savefig(save_path + 'bloch_sphere_difference_optimizer_2_obs_qng.pdf')
        plt.show()

    plot()


def plot_difference_3_obs_qng():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    plt.rc('font', family='serif')
    def ps_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)
            qml.RY(params[2], wires=1)
            qml.RZ(params[3], wires=1)

        observables = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(0)@qml.PauliZ(1)]
        params = [0.4, 2.4, 0.4, 2.4]


        costs_optimizer, states, params = parameter_shift_optimizer(circuit, params, observables,
                                                                    dev,
                                                                    eta=eta, return_state=True,
                                                                    return_params=True,
                                                                    nsteps=nsteps, tol=1e-9)

        return states, costs_optimizer

    def qng_grad_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)
            qml.RY(params[2], wires=1)
            qml.RZ(params[3], wires=1)

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(0)@qml.PauliZ(1)]
        params = np.array([0.4, 2.4,0.4, 2.4])

        opt = qml.QNGOptimizer(stepsize=eta, diag_approx=False)

        qngd_param_history = [params]
        costs_optimizer = []
        H = qml.Hamiltonian([1.0 for _ in range(len(observables))], observables)
        cost_fn = qml.ExpvalCost(circuit, H, dev)

        @qml.qnode(dev)
        def get_state(params, **kwargs):
            circuit(params)
            return qml.state()

        states = []
        for n in range(100):

            # Take step
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            qngd_param_history.append(params)
            costs_optimizer.append(prev_energy)
            states.append(get_state(params))
            # Compute energy
            energy = cost_fn(params)

            # Calculate difference between new and old energies
            conv = np.abs(energy - prev_energy)

            if conv <= 1e-5:
                break

        return states, costs_optimizer

    def riemann_optimization(eta, nsteps):
        def circuit(params, **kwargs):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=0)
            qml.RY(params[2], wires=1)
            qml.RZ(params[3], wires=1)
            return qml.state()

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(0)@qml.PauliZ(1)]
        params = [0.4, 2.4,0.4, 2.4]


        costs_optimizer, states = exact_lie_optimizer(circuit, params, observables, dev,
                                                      eta=eta,
                                                      return_state=True, nsteps=nsteps)
        phi_optimizer = []
        theta_optimizer = []
        costs_converted = []

        return states, costs_optimizer

    def plot():
        save_path = save_path_creator('./figures', '')

        nsteps = 500
        eta = 0.001


        # get optimizers
        psi_riemann, costs_riemann = riemann_optimization(10*eta, nsteps)
        psi_ps, costs_ps = ps_optimization(eta, nsteps)
        psi_qng, costs_qng = qng_grad_optimization(eta, nsteps)
        closeness = []
        for state in psi_qng:
            fids = [np.abs(np.vdot(state,s)) for s in psi_riemann]
            idx = np.argmax(fids)
            closeness.append([idx, fids[idx]])

        print(closeness)
        fig, axs = plt.subplots(1,1)
        # plot s
        axs.plot(np.array(psi_riemann).real, color='black', linewidth=2,
                 label=fr'Riemann $ \eta = $ {10*eta:1.3f}')
        axs.plot(np.array(psi_ps).real, color='red', linewidth=2,
                 label=fr'PS $ \eta = $ {eta:1.3f}')
        axs.plot(np.array(psi_qng).real, color='green', linewidth=2,
                 label=fr'QNG $ \eta = $ {eta:1.3f}')
        axs.legend()
        axs.set_title(r'Bloch sphere trajectories')

        plt.show()

    plot()


def plot_difference_4_obs_qng():
    def ps_optimization(eta, nsteps,params):
        def circuit(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(0)]


        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        costs_optimizer, states, params = parameter_shift_optimizer(circuit, params, observables,
                                                                    dev,
                                                                    eta=eta, return_state=True,
                                                                    return_params=True,
                                                                    nsteps=nsteps, tol=1e-9)
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
        return states, phi_optimizer, theta_optimizer, costs_optimizer

    def qng_grad_optimization(eta, nsteps, params):
        def circuit(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        paulis = get_su_2_operators()
        observables = [qml.PauliX(0), qml.PauliY(0)]

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        opt = qml.QNGOptimizer(stepsize=eta, diag_approx=False)

        qngd_param_history = [params]
        costs_optimizer = []
        H = qml.Hamiltonian([1.0 for _ in range(len(observables))], observables)
        cost_fn = qml.ExpvalCost(circuit, H, dev)

        @qml.qnode(dev)
        def get_state(params, **kwargs):
            circuit(params)
            return qml.state()

        states = []
        for n in range(nsteps):

            # Take step
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            qngd_param_history.append(params)
            costs_optimizer.append(prev_energy)
            states.append(get_state(params))
            # Compute energy
            energy = cost_fn(params)

            # Calculate difference between new and old energies
            conv = np.abs(energy - prev_energy)

            if conv <= 1e-5:
                break
        phi_optimizer = []
        theta_optimizer = []
        costs_converted = []

        for state in states:
            rho_state = np.outer(state, state.conj())
            costs_converted.append(cost_function(rho_state))
            p, t = state_to_spherical(rho_state)
            phi_optimizer.append(p)
            theta_optimizer.append(t)
        qngd_param_history = np.array(qngd_param_history) % (2 * np.pi)
        return states, phi_optimizer, theta_optimizer, costs_optimizer

    def riemann_optimization(eta, nsteps, params):
        def circuit(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
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

        return states, phi_optimizer, theta_optimizer, costs_optimizer

    def plot():
        save_path = save_path_creator('./figures', '')

        gran = 40
        nsteps = 500
        eta = 0.01

        phi = np.linspace(0, 2 * np.pi, gran)
        theta = np.linspace(0, np.pi, gran)
        paulis = get_su_2_operators()

        params = [0.4, 1.2]

        # get optimizers
        psi_riemann, phi_riemann, theta_riemann, costs_riemann = riemann_optimization(10*eta, nsteps, copy.copy(params))
        psi_ps, phi_ps, theta_ps, costs_ps = ps_optimization(eta, nsteps,copy.copy(params))
        print(params)
        psi_qng, phi_qng, theta_qng, costs_qng = qng_grad_optimization(eta, nsteps, np.array(params))
        closeness = []
        for state in psi_qng:
            fids = [np.abs(np.vdot(state,s)) for s in psi_riemann]
            idx = np.argmax(fids)
            closeness.append([idx, fids[idx]])

        print(closeness)
        # fig, axs = plt.subplots(1, 1)
        # # plot s
        # axs.plot(np.array(np.abs(psi_riemann)), color='black', linewidth=2,
        #          label=fr'Riemann $ \eta = $ {10 * eta:1.3f}')
        # axs.plot(np.array(np.abs(psi_ps)), color='red', linewidth=2,
        #          label=fr'PS $ \eta = $ {eta:1.3f}')
        # axs.plot(np.array(np.abs(psi_qng)), color='green', linewidth=2,
        #          label=fr'QNG $ \eta = $ {eta:1.3f}')
        # axs.legend()
        # axs.set_title(r'Bloch sphere trajectories')
        #
        # plt.show()
        # Bloch sphere 2D plot
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)

        def cost_function(rho):
            return np.trace((paulis[0] + paulis[1]) @ rho)

        phi_phi, theta_theta = np.meshgrid(phi, theta)
        costs = []
        for p, t in zip(phi_phi.flatten(), theta_theta.flatten()):
            costs.append(cost_function(spherical_to_state(p, t)).real)
        costs_bloch = np.array(costs).reshape((gran, gran))
        plot_bloch_sphere_2d(phi, theta, costs_bloch, axs, fig)


        # plot s
        axs.plot(phi_riemann, theta_riemann, color='black', linewidth=2,
                 label=fr'Riemann $ \eta = $ {10*eta:1.3f}')
        axs.plot(phi_ps, theta_ps, color='red', linewidth=2,
                 label=fr'PS $ \eta = $ {eta:1.3f}')
        axs.plot(phi_qng, theta_qng, color='green', linewidth=2,
                 label=fr'QNG $ \eta = $ {eta:1.3f}')
        axs.legend()
        axs.set_title(r'Bloch sphere trajectories')
        axs.plot(phi_riemann[0], theta_riemann[0], 'o', color='black')
        axs.plot(phi_riemann[-1], theta_riemann[-1], 'o', color='black')
        axs.annotate(r'$k = $ 0',
                     xy=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2), xycoords='data',
                     xytext=(phi_riemann[0] + 0.2, theta_riemann[0] + 0.2),
                     horizontalalignment='right', verticalalignment='top')
        axs.annotate(rf'$k = ${len(costs_riemann)}',
                     xy=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2), xycoords='data',
                     xytext=(phi_riemann[-1] + 0.2, theta_riemann[-1] + 0.2),
                     horizontalalignment='right', verticalalignment='top')

        # def circuit_cost(params, **kwargs):
        #     a, b = params
        #     qml.RY(a, wires=0)
        #     qml.RZ(b, wires=0)
        #
        # observables = [qml.PauliX(0), qml.PauliY(0)]
        # H = qml.Hamiltonian([1.0 for _ in range(len(observables))], observables)
        # circuit_cost = qml.ExpvalCost(circuit_cost, H, dev)
        #
        # @qml.qnode(dev)
        # def circuit_state(a, b, **kwargs):
        #     qml.RY(a, wires=0)
        #     qml.RZ(b, wires=0)
        #     return qml.state()
        #
        # alpha = np.linspace(0, 2 * np.pi, gran)
        # beta = np.linspace(0, 2 * np.pi, gran)
        # alpha_alpha, beta_beta = np.meshgrid(alpha, beta)
        # costs = []
        # for a, b in zip(alpha_alpha.flatten(), beta_beta.flatten()):
        #     costs.append(circuit_cost([a, b]))
        # costs_bloch = np.array(costs).reshape((gran, gran))
        #
        # im = axs[1].contourf(alpha, beta, costs_bloch, cmap='Reds')
        # axs[1].set_xlabel(r'$\alpha$')
        # axs[1].set_ylabel(r'$\beta$')
        # fig.colorbar(im, ax=axs[1])
        # # plot s
        # axs[1].plot(phi_ps, theta_ps, color='black', linewidth=2, label=fr'PS $\eta = $ {eta:1.3f}')
        # axs[1].legend()
        # axs[1].set_title(r'Bloch sphere trajectories')
        # axs[1].plot(phi_ps[0], theta_ps[0], 'o', color='black')
        # axs[1].plot(phi_ps[-1], theta_ps[-1], 'o', color='black')
        # axs[1].annotate(r'$k = $ 0',
        #                 xy=(phi_ps[0] + 0.2, theta_ps[0] + 0.2), xycoords='data',
        #                 xytext=(phi_ps[0] + 0.2, theta_ps[0] + 0.2),
        #                 horizontalalignment='right', verticalalignment='top')
        # axs[1].annotate(rf'$k = ${len(costs_ps)}',
        #                 xy=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2), xycoords='data',
        #                 xytext=(phi_ps[-1] + 0.2, theta_ps[-1] + 0.2),
        #                 horizontalalignment='right', verticalalignment='top')
        # for ax in axs:
        #     change_label_fontsize(ax, 15)
        fig.savefig(save_path + 'bloch_sphere_difference_optimizer_2_obs_qng.pdf')
        plt.show()

    plot()

if __name__ == '__main__':
    # plot_equal()
    # plot_difference()
    # plot_difference_2_obs()
    plot_difference_2_obs_sep_figs()
    # plot_difference_2_obs_qng()
    # plot_difference_4_obs_qng()
