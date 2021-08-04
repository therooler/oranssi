import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import exact_lie_optimizer, local_su_2_lie_optimizer, \
    local_su_4_lie_optimizer, parameter_shift_optimizer
from oranssi.circuit_tools import get_full_operator
from oranssi.utils import get_su_2_operators, get_su_4_operators
import matplotlib as mpl
import numpy as np


def two_observables_2_qubits_parameter_shift():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    observables = [qml.PauliY(0), qml.PauliX(1)]
    params = [0.1, 1.2]

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.state()

    cmap = plt.get_cmap('rainbow')
    colors = cmap(range(256))
    new_colors = []
    for i in range(0, 256, 256 // 30):
        new_colors.append(colors[i])
    su_2_colors = new_colors[:12]
    su_4_colors = new_colors[12:]
    # EXACT Lie directions
    costs_exact, states = exact_lie_optimizer(circuit, params, observables, dev, eta=eta,
                                              return_state=True)
    # Plot lie directions
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(18, 8)

    omegas_su2 = []
    full_observables = [get_full_operator(obs.matrix, obs.wires, nqubits) for obs in observables]
    paulis, names_su2 = get_su_2_operators(return_names=True)
    full_paulis_total = []
    for i in range(nqubits):
        full_paulis_total.append(
            [get_full_operator(p, (i,), nqubits) for p in paulis])
    for phi in states:
        omega = np.zeros((len(observables), 2, 3))
        phi = phi[:, np.newaxis]
        for o, obs in enumerate(full_observables):
            for p, full_paulis in enumerate(full_paulis_total):
                for j, pauli in enumerate(full_paulis):
                    omega[o, p, j] = (phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi)[0, 0].imag
        omegas_su2.append(omega)
    omegas_su2 = np.array(omegas_su2)
    for o, obs in enumerate(observables):
        for p in range(nqubits):
            for j in range(3):
                if not np.allclose(omegas_su2[:, o, p, j], 0):
                    cidx = np.ravel_multi_index(np.array([o, p, j]), (len(observables), 2, 3))
                    axs[0].plot(omegas_su2[:, o, p, j],
                                label=fr'{obs.name}, q={p}, $\omega$({names_su2[2]})',
                                color=su_2_colors[cidx])
    omegas_su4 = []
    paulis, names_su4 = get_su_4_operators(return_names=True)
    full_paulis_total = []
    for i in range(0, nqubits, 2):
        full_paulis_total.append(
            [get_full_operator(p, (i, i + 1), nqubits) for p in paulis])
    for phi in states:
        omega = np.zeros((len(observables), 1, 9))
        phi = phi[:, np.newaxis]
        for o, obs in enumerate(full_observables):
            for p, full_paulis in enumerate(full_paulis_total):
                for j, pauli in enumerate(full_paulis):
                    omega[o, p, j] = (phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi)[
                        0, 0].imag
        omegas_su4.append(omega)
    omegas_su4 = np.array(omegas_su4)
    len(su_4_colors)
    for o, obs in enumerate(observables):
        for p in range(nqubits // 2):
            for j in range(9):
                if not np.allclose(omegas_su4[:, o, p, j], 0):
                    cidx = np.ravel_multi_index(np.array([o, p, j]), (len(observables), 1, 9))
                    axs[0].plot(omegas_su4[:, o, p, j],
                                label=fr'{obs.name}, q=({p}{p + 1}), $\omega$({names_su4[j]})',
                                color=su_4_colors[cidx])
    # SU(2) local Lie directions
    costs_lie_2, omegas = local_su_2_lie_optimizer(circuit, params, observables, dev, eta=eta,
                                                   return_omegas=True)

    for o, obs in enumerate(observables):
        for p in range(nqubits):
            for j in range(3):
                if not np.allclose(omegas[:, o, p, j], 0):
                    cidx = np.ravel_multi_index(np.array([o, p, j]), (len(observables), 2, 3))
                    axs[1].plot(omegas[:, o, p, j],
                                label=fr'{obs.name}, q={p}, $\omega$({names_su2[j]})',
                                color=su_2_colors[cidx])
    # SU(4) local Lie directions
    costs_lie_4, omegas = local_su_4_lie_optimizer(circuit, params, observables, dev, eta=eta,
                                                   return_omegas=True)
    for o, obs in enumerate(observables):
        for p in range(nqubits // 2):
            for j in range(9):
                if not np.allclose(omegas[:, o, p, j], 0):
                    cidx = np.ravel_multi_index(np.array([o, p, j]), (len(observables), 1, 9))
                    axs[2].plot(omegas[:, o, p, j],
                                label=fr'{obs.name}, q=({p}{p + 1}), $\omega$({names_su4[j]})',
                                color=su_4_colors[cidx])
    for ax in axs:
        ax.set_xlabel('step')
        ax.set_ylabel('Lie algebra coefficient $i \omega$ ')
        ax.set_ylim([-2., 2.])
        ax.legend(prop={'size': 7})
    # 2* 9, 2*2*3
    axs[0].set_title('Exact')
    axs[1].set_title('SU(2) Local')
    axs[2].set_title('SU(4) Local')
    fig.suptitle(f'Non-zero directions on the Lie algebra for 2 qubit circuit, learning rate $\eta$={eta:1.3f}')
    plt.savefig(
        './figures' + f'/lie_algebra_directions_local_lie_optimizers_II_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()
    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_lie_2, label=r'Lie SU(4) local', color=cmap(0.4))
    plt.plot(costs_lie_4, label=r'Lie SU(2) local', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(len(costs_exact)), [-2.0 for _ in range(len(costs_exact))], label='Min.',
             color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_1 + X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {eta}$')
    plt.savefig(
        './figures' + f'/two_observable_local_lie_optimizers_II_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


if __name__ == '__main__':
    two_observables_2_qubits_parameter_shift()
