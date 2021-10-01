import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import os
from oranssi.optimizers import approximate_lie_optimizer, parameter_shift_optimizer
from oranssi.plot_utils import LABELSIZE, MARKERSIZE, LINEWIDTH, change_label_fontsize, \
    get_all_su_n_directions, plot_su16_directions
from oranssi.opt_tools import ZassenhausLayer
from oranssi.circuit_tools import get_hamiltonian_matrix


def two_observables_3_qubits_zassenhaus():
    nqubits = 3
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.5

    observables = [qml.PauliX(0), qml.PauliX(1), qml.PauliY(2), qml.PauliZ(2) @ qml.PauliY(1)]
    params = [0.2, 1.4]
    # costs_parameter_shift = parameter_shift_optimizer(circuit, params, observables, dev, eta=eta)
    H = get_hamiltonian_matrix(nqubits, observables)
    gs_en = np.min(np.linalg.eigvalsh(H))

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.RZ(params[0], wires=2)
        qml.RY(params[1], wires=0)
        qml.RY(params[1], wires=1)
        qml.RY(params[1], wires=2)
        return qml.state()

    costs_exact = approximate_lie_optimizer(circuit, params, observables, dev,
                                            layers=[ZassenhausLayer(dev, observables)], eta=eta)
    cmap = plt.cm.get_cmap('Set1')
    fig, axs = plt.subplots(1, 1)
    # axs.plot(costs_parameter_shift, label=r'PS', color=cmap(0.3), linewidth=LINEWIDTH)
    axs.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2), linewidth=LINEWIDTH)
    axs.plot(range(len(costs_exact)), [gs_en for _ in range(len(costs_exact))], label='Minimum',
             color='gray', linestyle='--', linewidth=LINEWIDTH - 1)
    axs.legend()
    change_label_fontsize(axs, LABELSIZE)
    axs.set_xlabel('Step')
    axs.set_ylabel(r'$\langle X_1 + X_2 \rangle$')
    plt.tight_layout()
    plt.savefig(
        './figures' + f'/two_observable_local_lie_optimizers_ps_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def tfim_zassenhaus(nqubits):
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.5

    def circuit(params, **kwargs):
        # LOWEST EIGENSTATE
        for n in range(nqubits):
            qml.Hadamard(wires=n)

        return qml.state()

    observables = [qml.PauliX(i) for i in range(nqubits)] + \
                  [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
                  [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]

    params = []
    # costs_parameter_shift = parameter_shift_optimizer(circuit, params, observables, dev, eta=eta)
    H = get_hamiltonian_matrix(nqubits, observables)
    gs_en = np.min(np.linalg.eigvalsh(H))
    print(f'GS energy: {gs_en}')
    ratio = (8,2) # -> (#SU4, #SU8 Zass)
    costs_exact, unitaries, zassenhaus = approximate_lie_optimizer(circuit, params, observables,
                                                                   dev,
                                                                   layers=[ZassenhausLayer(dev,
                                                                                           observables,ratio = ratio)],
                                                                   eta=eta,
                                                                   return_unitary=True,
                                                                   return_zassenhaus=True)
    fig, axs = plt.subplots(1, 1)
    # axs.plot(costs_parameter_shift, label=r'PS', color=cmap(0.3), linewidth=LINEWIDTH)
    axs.plot(np.abs(costs_exact-gs_en), label=r'Lie $SU(2^n)$', color='gray', linewidth=LINEWIDTH, zorder=-1)
    axs.plot(range(len(costs_exact)), [gs_en for _ in range(len(costs_exact))], label='Minimum',
             color='gray', linestyle='--', linewidth=LINEWIDTH - 1, zorder=-0.5)

    firstsu2, firstsu4, firstsu8 = True, True, True

    for i in range(len(zassenhaus)):
        if zassenhaus[i] == 'su4':
            if firstsu4:
                axs.scatter(i, np.abs(np.abs(costs_exact[i]-gs_en)), color='crimson', linestyle='--',
                            label='SU(4) layer', zorder=1)
                firstsu4 = False
            else:
                axs.scatter(i, np.abs(costs_exact[i]-gs_en), color='crimson', linestyle='--', zorder=1)
        elif zassenhaus[i] == 'su8':
            if firstsu8:
                axs.scatter(i, np.abs(costs_exact[i]-gs_en), color='navy', linestyle='--',
                            label='Zass. SU(8) layer', zorder=1)
                firstsu8 = False
            else:
                axs.scatter(i, np.abs(costs_exact[i]-gs_en), color='navy', linestyle='--', zorder=1)
    axs.set_yscale('log')
    axs.legend()
    change_label_fontsize(axs, LABELSIZE)
    axs.set_xlabel('Step')
    axs.set_ylabel(r'$\langle H \rangle$')
    axs.set_title(fr'Ratio SU4: Zass. SU(8) = {ratio[0]}:{ratio[1]} ')
    plt.tight_layout()
    plt.savefig(
        './figures' + f'/su8_zass_optimizer_tfim_{nqubits}.pdf')
    plt.show()

    if not os.path.exists('./data/tfim_su8_zass'):
        os.makedirs('./data/tfim_su8_zass')
    if nqubits<=4:
        omegas = []
        for i, uni in enumerate(unitaries):
            np.save(f'./data/tfim_su8_zass/uni_{i}', uni)
            omegas.append(get_all_su_n_directions(uni, observables, dev))
        fig, axs = plot_su16_directions(nqubits, unitaries, observables, dev)

        for i in range(len(zassenhaus)):
            if zassenhaus[i] == 'su4':
                if firstsu4:
                    axs.scatter(i, np.abs(np.abs(costs_exact[i] - gs_en)), color='crimson',
                                linestyle='--',
                                label='SU(4) layer', zorder=1)
                    firstsu4 = False
                else:
                    axs.scatter(i, np.abs(costs_exact[i] - gs_en), color='crimson', linestyle='--',
                                zorder=1)
            elif zassenhaus[i] == 'su8':
                if firstsu8:
                    axs.scatter(i, np.abs(costs_exact[i] - gs_en), color='navy', linestyle='--',
                                label='Zass. SU(8) layer', zorder=1)
                    firstsu8 = False
                else:
                    axs.scatter(i, np.abs(costs_exact[i] - gs_en), color='navy', linestyle='--',
                                zorder=1)
        fig.savefig('./figures' + f'/su8_directions_zass_optimizer_tfim_{nqubits}.pdf')
        plt.show()


if __name__ == '__main__':
    # two_observables_3_qubits_zassenhaus()
    tfim_zassenhaus(6)
