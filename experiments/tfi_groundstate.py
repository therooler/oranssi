import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import exact_lie_optimizer, local_su_2_lie_optimizer, \
    local_su_4_lie_optimizer, local_custom_su_lie_optimizer, parameter_shift_optimizer


def su_exact_optimizer(nqubits):
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    def circuit(params, **kwargs):
        for n in range(nqubits):
            qml.Hadamard(wires=n)
        return qml.state()

    params = []
    observables = [qml.PauliX(i) for i in range(nqubits)] + \
                  [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
                  [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]

    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=eta)

    cmap = plt.cm.get_cmap('Set1')

    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    # plt.plot(range(len(costs_exact)), [-5.226251859505502 for _ in range(len(costs_exact))],
    plt.plot(range(len(costs_exact)), [-10.25166179096609 for _ in range(len(costs_exact))],
             label='Min.', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle H \rangle$')
    plt.title(fr'{nqubits}-qubit transverse field Ising-model for $\eta = {eta}$')
    plt.savefig(
        './figures' + f'/exact_optimizer_tfim_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def su_2_local_optimizer(nqubits):
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.5

    def circuit(params, **kwargs):
        for n in range(nqubits):
            qml.Hadamard(wires=n)
            qml.RZ(params[0], wires=n)
        for n in range(nqubits - 1):
            qml.CNOT(wires=[n, n + 1])
        qml.CNOT(wires=[nqubits - 1, 0])
        for n in range(nqubits):
            qml.RY(params[1], wires=n)

        return qml.state()

    params = [0.3, 0.6]
    observables = [qml.PauliX(i) for i in range(nqubits)] + \
                  [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
                  [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]

    costs_exact = local_su_2_lie_optimizer(circuit, params, observables, dev, eta=eta)

    cmap = plt.cm.get_cmap('Set1')

    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(len(costs_exact)), [-5.226251859505502 for _ in range(len(costs_exact))],
             label='Min.', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle H \rangle$')
    plt.title(fr'{nqubits}-qubit transverse field Ising-model for $\eta = {eta}$')
    plt.savefig(
        './figures' + f'/su_2_local_optimizer_tfim_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def su_4_local_optimizer(nqubits):
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.5

    def circuit(params, **kwargs):
        for n in range(nqubits):
            qml.Hadamard(wires=n)
            qml.RZ(params[0], wires=n)
        for n in range(nqubits - 1):
            qml.CNOT(wires=[n, n + 1])
        qml.CNOT(wires=[nqubits - 1, 0])
        for n in range(nqubits):
            qml.RY(params[1], wires=n)

        return qml.state()

    params = [0.3, 0.6]
    observables = [qml.PauliX(i) for i in range(nqubits)] + \
                  [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
                  [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]

    costs_exact = local_su_4_lie_optimizer(circuit, params, observables, dev, eta=eta)

    cmap = plt.cm.get_cmap('Set1')

    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(len(costs_exact)), [-5.226251859505502 for _ in range(len(costs_exact))],
             label='Min.', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle H \rangle$')
    plt.title(fr'{nqubits}-qubit transverse field Ising-model for $\eta = {eta}$')
    plt.savefig(
        './figures' + f'/su_4_local_optimizer_tfim_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def su_4_2_local_optimizer(nqubits):
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 1.0

    def circuit(params, **kwargs):
        for n in range(nqubits):
            qml.Hadamard(wires=n)
            qml.RZ(params[0], wires=n)
        for n in range(nqubits - 1):
            qml.CNOT(wires=[n, n + 1])
        qml.CNOT(wires=[nqubits - 1, 0])
        for n in range(nqubits):
            qml.RY(params[1], wires=n)
        return qml.state()

    params = [0.3, 0.6,0.3, 0.2]

    observables = [qml.PauliX(i) for i in range(nqubits)] + \
                  [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
                  [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]

    costs_exact = local_custom_su_lie_optimizer(circuit, params, observables, dev, eta=eta,
                                                layer_pattern=[(1, 0), (2, 0), (2, 1)], nsteps=100,
                                                tol=1e-3, trotterize=True)
    # costs_exact = parameter_shift_optimizer(circuit, params, observables, dev, eta=eta, nsteps=100,
    #                                         tol=1e-3)
    cmap = plt.cm.get_cmap('Set1')

    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(len(costs_exact)), [-5.226251859505502 for _ in range(len(costs_exact))],
             label='Min.', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle H \rangle$')
    plt.title(fr'{nqubits}-qubit transverse field Ising-model for $\eta = {eta}$')
    plt.savefig(
        './figures' + f'/su_4_local_optimizer_tfim_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


if __name__ == '__main__':
    # su_exact_optimizer(nqubits=4)
    # su_2_local_optimizer(nqubits=4)
    # su_4_local_optimizer(nqubits=4)
    su_4_2_local_optimizer(nqubits=4)
