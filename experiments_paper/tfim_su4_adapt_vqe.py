import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import approximate_lie_optimizer
from oranssi.opt_tools import AdaptVQELayer
from oranssi.circuit_tools import get_all_su_n_directions, get_hamiltonian_matrix
import numpy as np
from oranssi.plot_utils import change_label_fontsize, LINEWIDTH, LABELSIZE, MARKERSIZE, \
    plot_su16_directions, plot_su16_directions_separate

np.set_printoptions(precision=3)

plt.rc('font', family='serif')


def tfim_qubits():
    nqubits = 4
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 1.0

    def circuit(params, **kwargs):
        # LOWEST EIGENSTATE
        for n in range(nqubits):
            qml.Hadamard(wires=n)


        return qml.state()

    observables = [qml.PauliX(i) for i in range(nqubits)] + \
                  [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
                  [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]
    H = get_hamiltonian_matrix(nqubits, observables)
    eigvals = np.linalg.eigvalsh(H)
    print(f'Spectrum: {eigvals}')
    params = []
    costs_exact, unitary = approximate_lie_optimizer(circuit, params, observables,
                                                         dev, layers=[AdaptVQELayer(dev, observables)],
                                                     eta=eta,
                                                         return_unitary=True,
                                                         nsteps=15, escape_tol=1e-5)
    omegas = []
    for uni in unitary:
        omegas.append(get_all_su_n_directions(uni, observables, dev))
        print(omegas[-1])

    fig1, ax = plt.subplots(1, 1)
    fig1.set_size_inches(8, 8)
    cmap = plt.get_cmap('Reds')
    ax.plot(costs_exact, label=r'Lie $SU(2^n)$ Stoch.', color=cmap(0.3), zorder=-1)
    # axs.plot(costs_exact_unp, label=r'Lie $SU(2^n)$', color=cmap(0.2), zorder=-1)
    ax.plot(range(len(costs_exact)), [-5.226251859505502 for _ in range(len(costs_exact))],
            label='Min.',
            color='gray', linestyle='--', zorder=-1)

    fig1, ax = plt.subplots(1, 1)
    fig1.set_size_inches(8, 8)
    cmap = plt.get_cmap('Reds')
    # axs.plot(costs_exact_unp, label=r'Lie $SU(2^n)$', color=cmap(0.2), zorder=-1)
    ax.plot(range(len(costs_exact)), [-5.226251859505502 for _ in range(len(costs_exact))],
            label='Min.', color='gray', linestyle='--', zorder=-1)
    ax.plot(costs_exact, label=r'Lie $SU(2^n)$ Stoch.', color=cmap(0.7), zorder=-1,
            linewidth=LINEWIDTH)

    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\langle H \rangle$')
    change_label_fontsize(ax, LABELSIZE)
    fig1.savefig('./figures' + f'/cost_su4_optimizer_tfim.pdf')
    fig, axs = plot_su16_directions_separate(nqubits, unitary, observables, device=dev)

    fig.savefig('./figures' + f'/adapt_su16_optimizer_tfim.pdf')

    fig2, axs2 = plot_su16_directions(nqubits, unitary, observables, device=dev)
    fig2.savefig('./figures' + f'/adapt_su16_abs_optimizer_tfim.pdf')
    plt.show()


if __name__ == '__main__':
    # two_observables_2_qubits()
    tfim_qubits()
