import pytest
import pennylane as qml
import numpy as np
from oranssi.optimizers import exact_lie_optimizer, parameter_shift_optimizer
from test_fixtures import circuit_1, circuit_2, circuit_1_bad_return_types


def test_single_qubit_single_observable_example():
    nqubits = 1
    dev = qml.device('default.qubit', wires=nqubits)
    eta = 0.2

    def circuit(params, **kwargs):
        qml.RY(params[0], wires=0)


    observables = [qml.PauliX(0)]
    params = [0.2, ]
    costs_parameter_shift = parameter_shift_optimizer(circuit, params, observables, dev, eta=eta)

    def circuit(params, **kwargs):
        qml.RY(params[0], wires=0)
        return qml.state()

    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=eta)

    assert all(np.isclose(x, y) for x, y in zip(costs_parameter_shift, costs_exact))
