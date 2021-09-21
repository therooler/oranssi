import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import exact_lie_optimizer
import numpy as np
from oranssi.utils import get_su_2_operators
from oranssi.plot_utils import change_label_fontsize, LABELSIZE, MARKERSIZE, LINEWIDTH

np.set_printoptions(precision=3)

plt.rc('font', family='serif')


def two_observables_2_qubits():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

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

        return qml.state()

    observables = [qml.PauliX(1), qml.PauliX(0),qml.PauliY(0)]
    paulis = get_su_2_operators(True)
    H = np.kron(paulis[1], paulis[0]) + np.kron(paulis[2], paulis[0])+ np.kron(paulis[0], paulis[1])
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    params = [0.1, ]
    print(eigenvalues)
    print(eigenvectors)
    print(qml.QNode(circuit, dev)(params))
    costs_exact_unp, unitary_unp = exact_lie_optimizer(circuit, params, observables, dev,
                                                       eta=eta,
                                                       return_unitary=True,
                                                       nsteps=100)
    diag1 = unitary_unp.conj().T @ H @unitary_unp
    costs_exact_p, perturbations, unitary_p = exact_lie_optimizer(circuit, params, observables,
                                                                    dev,
                                                                    eta=eta,
                                                                    perturbation=True,
                                                                    return_perturbations=True,
                                                                    return_unitary=True,
                                                                    nsteps=100)

    diag2 = unitary_p.conj().T @ H @unitary_p
    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(16, 6)
    plt.set_cmap('Blues')
    im1 = axs[0].matshow(diag1.real)
    axs[0].set_title('Unperturbed diagonalization')
    plt.colorbar(im1,ax=axs[0])
    im2 = axs[1].matshow(diag2.real)
    axs[1].set_title('Perturbed diagonalization')
    plt.colorbar(im2,ax=axs[1])
    plt.tight_layout()
    cmap = plt.cm.get_cmap('Set2')

    axs[2].plot(costs_exact_p, label=r'Lie $SU(2^n)$ Pert.', color=cmap(0.3), zorder=0, linewidth=LINEWIDTH)
    axs[2].plot(costs_exact_unp, label=r'Lie $SU(2^n)$', color=cmap(0.2), zorder=0, linewidth=LINEWIDTH)
    axs[2].plot(range(len(costs_exact_p)), [eigenvalues[0] for _ in range(len(costs_exact_p))],
             label='Minimum',
             color='black', linestyle='--', zorder=-1, linewidth=LINEWIDTH-1)
    axs[2].plot(range(len(costs_exact_p)), [eigenvalues[1] for _ in range(len(costs_exact_p))],
                label='Eigenvalue 1',
                color='gray', linestyle='--', zorder=-1, linewidth=LINEWIDTH-1)
    for i, p in enumerate(perturbations):
        if i == 0:
            axs[2].scatter(p, costs_exact_p[p], color=cmap(0.1), label='Perturbation', s=MARKERSIZE, zorder=1)
        else:
            axs[2].scatter(p, costs_exact_p[p], color=cmap(0.1), s=MARKERSIZE, zorder=1)

    axs[2].legend()
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel(r'$\langle X_1 +Y_1 + X_2 \rangle$')
    axs[2].set_title(fr'Different optimizers for $\eta = {eta}$')
    for ax in axs:
        change_label_fontsize(ax, LABELSIZE)
    fig.tight_layout()
    fig.savefig('./figures' + f'/eigenstate_perturbation.pdf')
    plt.show()


if __name__ == '__main__':
    two_observables_2_qubits()
