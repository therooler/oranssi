import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import exact_lie_optimizer, parameter_shift_optimizer
from oranssi.circuit_tools import get_hamiltonian_matrix
from oranssi.plot_utils import LABELSIZE, LINEWIDTH, change_label_fontsize
import numpy as np


def two_observables_2_qubits_parameter_shift():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.5

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        # return qml.state()

    observables = [qml.PauliX(1), qml.PauliX(0), qml.PauliY(1)]
    params = [0.1, 1.2]
    costs_parameter_shift = parameter_shift_optimizer(circuit, params, observables, dev, eta=eta)
    H = get_hamiltonian_matrix(nqubits, observables)
    eigvals = np.linalg.eigvalsh(H)
    print(eigvals)

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.state()

    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=eta)
    cmap = plt.cm.get_cmap('Set1')
    fig, axs = plt.subplots(1, 1)
    axs.plot(costs_parameter_shift, label=r'PS', color=cmap(0.3), linewidth=LINEWIDTH)
    axs.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2), linewidth=LINEWIDTH)
    axs.plot(range(max(len(costs_exact),len(costs_parameter_shift))),
             [min(eigvals) for _ in range(max(len(costs_exact),  len(costs_parameter_shift)))],
             label='Minimum',
             color='gray', linestyle='--', linewidth=LINEWIDTH - 1)
    axs.legend()
    change_label_fontsize(axs, LABELSIZE)
    axs.set_xlabel('Step')
    axs.set_ylabel(r'$\langle X_1 + X_2 + Y_2\rangle$')
    plt.tight_layout()
    plt.savefig(
        './figures' + f'/two_observable_local_lie_optimizers_ps_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


if __name__ == '__main__':
    two_observables_2_qubits_parameter_shift()
