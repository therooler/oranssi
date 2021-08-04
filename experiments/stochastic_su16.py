import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import algebra_squared_su_lie_optimizer

from oranssi.circuit_tools import get_all_su_n_directions, get_hamiltonian_matrix
import numpy as np
from oranssi.plot_utils import change_label_fontsize

np.set_printoptions(precision=3)

plt.rc('font', family='serif')


def tfim_qubits():
    nqubits = 4
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 1.0

    def circuit(params, **kwargs):
        # LOWEST EIGENSTATE
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        # PERTURBATION
        qml.RY(params[0], wires=0)

        return qml.state()

    observables = [qml.PauliX(i) for i in range(nqubits)] + \
                  [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
                  [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]
    H = get_hamiltonian_matrix(nqubits, observables)
    eigvals = np.linalg.eigvalsh(H)
    print(f'Spectrum: {eigvals}')
    params = [0.1]
    costs_exact, unitary = algebra_squared_su_lie_optimizer(circuit, params, observables,
                                                            dev, eta=eta,
                                                            return_unitary=True,
                                                            nsteps=15, escape_tol=1e-5)
    omegas = []
    for uni in unitary:
        omegas.append(get_all_su_n_directions(uni, observables, nqubits, dev))
        print(omegas[-1])

    fig1, ax = plt.subplots(1, 1)
    fig1.set_size_inches(8, 8)
    cmap = plt.get_cmap('Reds')
    ax.plot(costs_exact, label=r'Lie $SU(2^n)$ Stoch.', color=cmap(0.3), zorder=-1)
    # axs.plot(costs_exact_unp, label=r'Lie $SU(2^n)$', color=cmap(0.2), zorder=-1)
    ax.plot(range(len(costs_exact)), [-5.226251859505502 for _ in range(len(costs_exact))],
             label='Min.',
             color='gray', linestyle='--', zorder=-1)
    # for i, p in enumerate(perturbations):
    #     if i == 0:
    #         axs.scatter(p, costs_exact_p[p], color=cmap(0.1), label='Perturbation', s=15, zorder=1)
    #     else:
    #         axs.scatter(p, costs_exact_p[p], color=cmap(0.1), s=15, zorder=1)
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(16, 16)

    keys = list(omegas[0].keys())

    axs[0,0].set_prop_cycle('color',plt.cm.Blues(np.linspace(0,1,sum(1 if k.count('I')==3 else 0 for k in keys)+1)))
    axs[0,1].set_prop_cycle('color',plt.cm.Reds(np.linspace(0,1,sum(1 if k.count('I')==2 else 0 for k in keys)+1)))
    axs[1,0].set_prop_cycle('color',plt.cm.Greens(np.linspace(0,1,sum(1 if k.count('I')==1 else 0 for k in keys)+1)))
    axs[1,1].set_prop_cycle('color',plt.cm.Purples(np.linspace(0,1,sum(1 if k.count('I')==0 else 0 for k in keys)+1)))

    for k in omegas[0].keys():
        if k.count('I')==3:
            axs[0,0].plot([om[k] for om in omegas], label=k)
        if k.count('I')==2:
            axs[0,1].plot([om[k] for om in omegas], label=k)
        if k.count('I')==1:
            axs[1,0].plot([om[k] for om in omegas], label=k)
        if k.count('I')==0:
            axs[1,1].plot([om[k] for om in omegas], label=k)
    for a in axs.flatten():
        change_label_fontsize(a, 15)
        a.legend()
        a.set_xlabel('Step')
        a.set_ylabel(r'$\langle H \rangle$')
        a.set_title(fr'Different optimizers for $\eta = {eta}$')
        a.set_xlabel('Step')
        a.set_ylabel(r'$\omega$')
        a.legend()
    fig.savefig('./figures' + f'/stochastic_su16_optimizer_tfim.pdf')
    plt.show()


if __name__ == '__main__':
    # two_observables_2_qubits()
    tfim_qubits()
