import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import exact_lie_optimizer, local_su_2_lie_optimizer, \
    local_su_4_lie_optimizer


def main():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)

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
    costs_local_4 = local_su_4_lie_optimizer(circuit, params, observables, dev, eta=0.2)
    # costs_local_4 = su_local_layer_lie_optimizer(circuit, params, observables, dev, eta=0.2)
    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=0.2)
    costs_local_2 = local_su_2_lie_optimizer(circuit, params, observables, dev, eta=0.2)
    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_local_4, label=r'Lie $SU_{loc_4}(2^n)$', color=cmap(0.4))
    plt.plot(costs_local_2, label=r'Lie $SU_{loc_2}(2^n)$', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(20), [-2.0 for _ in range(20)], label='Min.', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_1 + X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {0.2}$')
    plt.savefig('./figures' + f'/two_observable_local_lie_optimizers_nq_{nqubits}_{0.2:1.3f}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
