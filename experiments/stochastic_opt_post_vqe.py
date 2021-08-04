import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import algebra_squared_su_lie_optimizer

from oranssi.circuit_tools import circuit_state_from_unitary, get_hamiltonian_matrix
import numpy as np
from oranssi.utils import get_su_2_operators
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

        # SECOND LOWEST
        # qml.Hadamard(wires=0)
        # qml.CNOT(wires=[0, 1])
        # qml.PauliZ(wires=1)

        # PERTURBATION
        qml.RY(params[0], wires=0)

        # qml.CNOT(wires=[0, 1])
        # qml.PauliZ(wires=1)
        # qml.RY(np.pi, wires=0)
        # qml.RY(-np.pi, wires=1)
        # qml.RZ(params[1], wires=0)
        # qml.RZ(params[1], wires=1)
        return qml.state()

    observables = [qml.PauliX(i) for i in range(nqubits)] + \
                  [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
                  [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]
    H = get_hamiltonian_matrix(nqubits, observables)
    eigvals = np.linalg.eigvalsh(H)
    print(f'Spectrum: {eigvals}')
    params = [0.1]
    costs_exact, gates = algebra_squared_su_lie_optimizer(circuit, params, observables,
                                                            dev, eta=eta,
                                                            return_unitary=False, return_gates=True,
                                                            nsteps=15, escape_tol=1e-5)
    print(gates)

    def circuit_lie(params):
        for g in gates:
            qml.Pauli
        pass
    # diag2 = unitary_p.conj().T @ H @unitary_p
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(16, 6)
    cmap = plt.get_cmap('Reds')
    axs.plot(costs_exact, label=r'Lie $SU(2^n)$ Stoch.', color=cmap(0.3), zorder=-1)
    # axs.plot(costs_exact_unp, label=r'Lie $SU(2^n)$', color=cmap(0.2), zorder=-1)
    axs.plot(range(len(costs_exact)), [-5.226251859505502 for _ in range(len(costs_exact))],
             label='Min.',
             color='gray', linestyle='--', zorder=-1)
    # for i, p in enumerate(perturbations):
    #     if i == 0:
    #         axs.scatter(p, costs_exact_p[p], color=cmap(0.1), label='Perturbation', s=15, zorder=1)
    #     else:
    #         axs.scatter(p, costs_exact_p[p], color=cmap(0.1), s=15, zorder=1)

    axs.legend()
    axs.set_xlabel('Step')
    axs.set_ylabel(r'$\langle H \rangle$')
    axs.set_title(fr'Different optimizers for $\eta = {eta}$')
    change_label_fontsize(axs, 15)
    fig.savefig('./figures' + f'/stochastic_optimizer_tfim.pdf')
    plt.show()


if __name__ == '__main__':
    tfim_qubits()
