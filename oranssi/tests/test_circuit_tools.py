import pytest
import pennylane as qml
import numpy as np
from oranssi.circuit_tools import get_full_operator, get_ops_from_qnode, \
    circuit_observable_from_unitary, \
    circuit_state_from_unitary
from oranssi.optimizers import exact_lie_optimizer, local_su_2_lie_optimizer
from test_fixtures import circuit_1, circuit_1_bad_return_types


def test_get_full_unitary(circuit_1):
    circuit, device, param_shape = circuit_1
    nqubits = len(device.wires)
    params = np.random.randn(*param_shape)

    circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    state_numpy = np.zeros((2 ** nqubits), dtype=complex)
    state_numpy[0] = 1.
    state_numpy = circuit_unitary @ state_numpy

    qnode = qml.QNode(circuit, device)
    qnode.construct([params], {})
    state_pennylane = qnode(params)
    assert np.allclose(state_numpy, state_pennylane)


def test_circuit_observable_from_unitary():
    device = qml.device('default.qubit', wires=2)
    qnode = qml.QNode(circuit_observable_from_unitary, device)
    with pytest.raises(AssertionError,
                       match='kwargs must MUST contain keys `unitary` and `observable`,'):
        qnode()


def test_circuit_state_from_unitary():
    device = qml.device('default.qubit', wires=2)
    qnode = qml.QNode(circuit_state_from_unitary, device)
    with pytest.raises(AssertionError, match='kwargs must MUST contain key `unitary`,'):
        qnode()
