import pytest
import pennylane as qml
import numpy as np
import itertools as it

from oranssi.circuit_tools import get_full_operator, get_ops_from_qnode, \
    circuit_observable_from_unitary, \
    circuit_state_from_unitary
from oranssi.utils import get_su_4_operators, get_su_2_operators
from test_fixtures import circuit_1, circuit_1_bad_return_types, circuit_3, circuit_4_state_obs


def test_get_full_unitary_2qb(circuit_1):
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


def test_get_full_unitary_4qb(circuit_3):
    circuit, device, param_shape = circuit_3
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


@pytest.mark.parametrize('circuit_4_state_obs', [2, 3, 4], indirect=True)
def test_observable_calculation_loc_1(circuit_4_state_obs):
    circuit, circuit_state, device, param_shape = circuit_4_state_obs
    nqubits = len(device.wires)
    dev = qml.device('default.qubit', wires=nqubits)

    np.random.seed(324234)
    params = np.random.randn(2)

    get_state = qml.QNode(circuit_state, dev)
    get_obs = qml.QNode(circuit, dev)

    paulis, directions = get_su_2_operators(return_names=True)

    for n in range(nqubits):
        single_qml_obs = [qml.PauliX(n), qml.PauliY(n), qml.PauliZ(n)]
        for i, obs in enumerate(single_qml_obs):
            o = get_obs(params, observable=obs)
            state = get_state(params)
            o_np = state.conj().T @ (get_full_operator(paulis[i], (n,), nqubits)) @ state
            assert np.isclose(o, o_np)


@pytest.mark.parametrize('circuit_4_state_obs', [2, 3, 4], indirect=True)
def test_observable_calculation_loc_2(circuit_4_state_obs):
    circuit, circuit_state, device, param_shape = circuit_4_state_obs
    nqubits = len(device.wires)
    dev = qml.device('default.qubit', wires=nqubits)

    np.random.seed(324234)
    params = np.random.randn(2)

    get_state = qml.QNode(circuit_state, dev)
    get_obs = qml.QNode(circuit, dev)

    paulis, directions = get_su_4_operators(return_names=True)

    for comb in it.combinations(range(nqubits), r=2):
        single_qml_obs = [qml.PauliX(comb[0]) @ qml.PauliX(comb[1]),
                          qml.PauliX(comb[0]) @ qml.PauliY(comb[1]),
                          qml.PauliX(comb[0]) @ qml.PauliZ(comb[1]),
                          qml.PauliY(comb[0]) @ qml.PauliX(comb[1]),
                          qml.PauliY(comb[0]) @ qml.PauliY(comb[1]),
                          qml.PauliY(comb[0]) @ qml.PauliZ(comb[1]),
                          qml.PauliZ(comb[0]) @ qml.PauliX(comb[1]),
                          qml.PauliZ(comb[0]) @ qml.PauliY(comb[1]),
                          qml.PauliZ(comb[0]) @ qml.PauliZ(comb[1])
                          ]
        for i, obs in enumerate(single_qml_obs):
            o = get_obs(params, observable=obs)
            state = get_state(params)
            o_np = state.conj().T @ (get_full_operator(paulis[i], comb, nqubits)) @ state
            assert np.isclose(o, o_np)
