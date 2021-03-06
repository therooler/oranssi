import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import exact_lie_optimizer, local_su_2_lie_optimizer, \
    local_su_4_lie_optimizer, parameter_shift_optimizer


def single_observable_1_qubit():
    nqubits = 1
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.RZ(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.state()

    observables = [qml.PauliX(0)]
    params = [0.1, 1.2]
    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=eta)
    costs_local_2 = local_su_2_lie_optimizer(circuit, params, observables, dev, eta=eta)
    costs_custom = local_su_2_lie_optimizer(circuit, params, observables, dev, eta=eta,
                                            layer_pattern=([1, 0]))
    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_custom, label=r'Lie $SU(2^n) (custom) $', color=cmap(0.5))
    plt.plot(costs_local_2, label=r'Lie $SU_{loc_2}(2^n)$', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(20), [-1.0 for _ in range(20)], label='Min.', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {eta}$')
    plt.savefig(
        './figures' + f'/single_observable_local_lie_optimizers_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def single_observable_2_qubits():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.1

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.state()

    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    costs_local_4 = local_su_4_lie_optimizer(circuit, params, observables, dev, eta=eta)
    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=eta)
    costs_local_2 = local_su_2_lie_optimizer(circuit, params, observables, dev, eta=eta)
    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_local_4, label=r'Lie $SU_{loc_4}(2^n)$', color=cmap(0.4))
    plt.plot(costs_local_2, label=r'Lie $SU_{loc_2}(2^n)$', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(len(costs_local_4)), [-1.0 for _ in range(len(costs_local_4))], label='Min.',
             color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {eta}$')
    plt.savefig(
        './figures' + f'/single_observable_local_lie_optimizers_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def two_observables_2_qubits():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.state()

    observables = [qml.PauliX(1), qml.PauliX(0)]
    params = [0.1, 1.2]
    costs_local_4 = local_su_4_lie_optimizer(circuit, params, observables, dev, eta=eta)
    # costs_local_4 = su_local_layer_lie_optimizer(circuit, params, observables, dev, eta=0.2)
    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=eta)
    costs_local_2 = local_su_2_lie_optimizer(circuit, params, observables, dev, eta=eta)
    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_local_4, label=r'Lie $SU_{loc_4}(2^n)$', color=cmap(0.4))
    plt.plot(costs_local_2, label=r'Lie $SU_{loc_2}(2^n)$', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(len(costs_local_4)), [-2.0 for _ in range(len(costs_local_4))], label='Min.',
             color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_1 + X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {eta}$')
    plt.savefig('./figures' + f'/two_observable_local_lie_optimizers_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def two_observables_two_qubits_trotterized():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.state()

    observables = [qml.PauliX(1), qml.PauliX(0)]
    params = [0.1, 1.2]
    costs_local_4 = local_su_4_lie_optimizer(circuit, params, observables, dev, eta=eta)
    costs_local_4_trotterized = local_su_4_lie_optimizer(circuit, params, observables, dev, eta=eta,
                                                         trotterize=True)
    print(costs_local_4_trotterized)
    print(costs_local_4)
    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_local_4, label=r'Lie $SU_{loc_4}(2^n)$', color=cmap(0.4))
    plt.plot(costs_local_4_trotterized, label=r'Lie $SU_{loc_4}(2^n) $ (Trotterized)',
             color=cmap(0.3))
    plt.plot(range(20), [-2.0 for _ in range(20)], label='Min.', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_1 + X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {eta}$')
    plt.savefig(
        './figures' + f'/two_observable_local_lie_optimizers_trotterized_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def two_observables_2_qubits_parameter_shift():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        # return qml.state()

    observables = [qml.PauliX(1), qml.PauliX(0)]
    params = [0.1, 1.2]
    costs_parameter_shift = parameter_shift_optimizer(circuit, params, observables, dev, eta=eta)

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
    plt.plot(costs_parameter_shift, label=r'PS', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(len(costs_exact)), [-2.0 for _ in range(len(costs_exact))], label='Min.',
             color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_1 + X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {eta}$')
    plt.savefig('./figures' + f'/two_observable_local_lie_optimizers_ps_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


def single_observable_1_qubit_parameter_shift():
    nqubits = 1
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    def circuit(params, **kwargs):
        # qml.Hadamard(wires=0)
        qml.RY(params[0], wires=0)
        # qml.RZ(params[1], wires=0)
        # return qml.state()

    observables = [qml.PauliX(0)]
    params = [0.4,1.2]
    costs_parameter_shift = parameter_shift_optimizer(circuit, params, observables, dev, eta=eta)
    params = [0.4, 1.2]
    def circuit(params, **kwargs):
        # qml.Hadamard(wires=0)
        qml.RY(params[0], wires=0)
        # qml.RZ(params[1], wires=0)
        return qml.state()

    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=eta)

    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_parameter_shift, label=r'PS', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(max(len(costs_exact), len(costs_parameter_shift))),
             [-1.0 for _ in range(max(len(costs_exact), len(costs_parameter_shift)))], label='Min.',
             color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_1 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {eta}$')
    plt.savefig('./figures' + f'/single_observable_local_lie_optimizers_ps_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()

def single_observable_2_qubits_parameter_shift():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RY(params[0], wires=0)
        qml.RY(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        # return qml.state()

    observables = [qml.PauliX(1)]
    params = [1.2, 0.8]
    costs_parameter_shift = parameter_shift_optimizer(circuit, params, observables, dev, eta=eta)

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RY(params[0], wires=0)
        qml.RY(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.state()

    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=eta)
    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_parameter_shift, label=r'PS', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(len(costs_parameter_shift)), [-1.0 for _ in range(len(costs_parameter_shift))], label='Min.',
             color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {eta}$')
    plt.savefig('./figures' + f'/single_observable_local_lie_optimizers_ps_nq_{nqubits}_{eta:1.3f}.pdf')
    plt.show()


if __name__ == '__main__':
    # single_observable_1_qubit()
    # single_observable_2_qubits()
    # two_observables_2_qubits()
    # two_observables_two_qubits_trotterized()
    # two_observables_2_qubits_parameter_shift()
    single_observable_1_qubit_parameter_shift()
    # single_observable_2_qubits_parameter_shift()
