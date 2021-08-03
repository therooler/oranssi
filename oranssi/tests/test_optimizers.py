import pytest
import pennylane as qml
import numpy as np
from oranssi.optimizers import exact_lie_optimizer, local_su_2_lie_optimizer, local_su_4_lie_optimizer, \
    circuit_state_from_unitary
from oranssi.opt_tools import LocalLieAlgebraLayer
from test_fixtures import circuit_1, circuit_2, circuit_1_bad_return_types


### Test LieLayer ###


def test_assert_circuit_LocalLieLayer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)

    with pytest.raises(AssertionError, match='Only SU'):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 3, len(device.wires))


def test_assert_nqubits_LocalLieLayer(circuit_2):
    circuit, device, param_shape = circuit_2
    observables = [qml.PauliX(1)]
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)

    with pytest.raises(AssertionError, match='`nqubits` must be even, received'):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires))


def test_assert_eta_LocalLieLayer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)

    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), eta=-1)
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), eta=10.)
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), eta=10)
    LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), eta=0.1)


def test_assert_stride_LocalLieLayer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)

    with pytest.raises(AssertionError, match='`stride` must be'):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), stride=2)
    with pytest.raises(AssertionError, match='`stride` must be'):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), stride=-1)
    LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), stride=0)
    LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), stride=1)


def test_assert_unitary_error_check_LocalLieLayer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)

    with pytest.raises(AssertionError, match='`unitary_error_check` must be a boolean, '):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), unitary_error_check='check')
    LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), unitary_error_check=True)


def test_assert_trotterize_LocalLieLayer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)

    with pytest.raises(AssertionError, match='`trotterize` must be a boolean, '):
        LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), trotterize='check')
    LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, 2, len(device.wires), trotterize=True)

### Test optimizer assertions ###

def test_assert_nsteps_exact_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError,
                       match='`nsteps` must be an integer between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, nsteps=-1)
    with pytest.raises(AssertionError,
                       match='`nsteps` must be an integer between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, nsteps=0.1)
    exact_lie_optimizer(circuit, params, observables, device, nsteps=10)


def test_assert_eta_exact_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=-1.0)
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=10.)
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=10)
    exact_lie_optimizer(circuit, params, observables, device, eta=0.10)


def test_assert_tol_exact_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`tol` must be an float between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, tol=-1.0)
    with pytest.raises(AssertionError, match='`tol` must be an float between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, tol=10)
    exact_lie_optimizer(circuit, params, observables, device, tol=0.10)


@pytest.mark.parametrize("circuit_1_bad_return_types", ['float', 'observable'], indirect=True)
def test_assert_circuit_exact_lie_optimizer(circuit_1_bad_return_types):
    circuit, device, param_shape = circuit_1_bad_return_types
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`circuit` must return a state'):
        exact_lie_optimizer(circuit, params, observables, device, eta=0.2)


def test_assert_nsteps_su_2_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError,
                       match='`nsteps` must be an integer between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, nsteps=-1)
    with pytest.raises(AssertionError,
                       match='`nsteps` must be an integer between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, nsteps=0.1)
    local_su_2_lie_optimizer(circuit, params, observables, device, nsteps=10)


def test_assert_eta_su_2_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=-1.0)
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=10.)
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=10)
    local_su_2_lie_optimizer(circuit, params, observables, device, eta=0.10)


def test_assert_tol_su_2_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`tol` must be an float between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, tol=-1.0)
    with pytest.raises(AssertionError, match='`tol` must be an float between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, tol=10)
    local_su_2_lie_optimizer(circuit, params, observables, device, tol=0.10)


@pytest.mark.parametrize("circuit_1_bad_return_types", ['float', 'observable'], indirect=True)
def test_assert_circuit_exact_lie_optimizer(circuit_1_bad_return_types):
    circuit, device, param_shape = circuit_1_bad_return_types
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`circuit` must return a state'):
        exact_lie_optimizer(circuit, params, observables, device, eta=0.2)

def test_assert_nsteps_su_4_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError,
                       match='`nsteps` must be an integer between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, nsteps=-1)
    with pytest.raises(AssertionError,
                       match='`nsteps` must be an integer between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, nsteps=0.1)
    local_su_4_lie_optimizer(circuit, params, observables, device, nsteps=10)


def test_assert_eta_su_4_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=-1.0)
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=10.)
    with pytest.raises(AssertionError, match='`eta` must be an float between 0 and 1, '):
        exact_lie_optimizer(circuit, params, observables, device, eta=10)
    local_su_4_lie_optimizer(circuit, params, observables, device, eta=0.10)


def test_assert_tol_su_4_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`tol` must be an float between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, tol=-1.0)
    with pytest.raises(AssertionError, match='`tol` must be an float between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, tol=10)
    local_su_4_lie_optimizer(circuit, params, observables, device, tol=0.10)


@pytest.mark.parametrize("circuit_1_bad_return_types", ['float', 'observable'], indirect=True)
def test_assert_circuit_exact_lie_optimizer(circuit_1_bad_return_types):
    circuit, device, param_shape = circuit_1_bad_return_types
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`circuit` must return a state'):
        exact_lie_optimizer(circuit, params, observables, device, eta=0.2)


@pytest.mark.parametrize("circuit_1_bad_return_types", ['float', 'observable'], indirect=True)
def test_assert_circuit_exact_lie_optimizer(circuit_1_bad_return_types):
    circuit, device, param_shape = circuit_1_bad_return_types
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`circuit` must return a state'):
        exact_lie_optimizer(circuit, params, observables, device, eta=0.2)


def test_assert_circuit_local_su_4_lie_optimizer(circuit_2):
    circuit, device, param_shape = circuit_2
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`nqubits` must be even, received'):
        local_su_4_lie_optimizer(circuit, params, observables, device, eta=0.2)


### Test optimization ###

def test_optimization_exact_lie_optimizer_single_obs(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = exact_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -1, atol=1e-2)


def test_optimization_exact_su_2_lie_optimizer_single_obs(circuit_2):
    circuit, device, param_shape = circuit_2
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = local_su_2_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-2)
    assert np.isclose(costs[-1], -1, atol=5e-2)


def test_optimization_exact_su_4_lie_optimizer_single_obs(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = local_su_4_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-2)
    assert np.isclose(costs[-1], -1, atol=5e-2)


def test_optimization_exact_lie_optimizer_double_obs(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(0), qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = exact_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -2, atol=1e-2)


def test_optimization_exact_su_2_lie_optimizer_double_obs(circuit_2):
    circuit, device, param_shape = circuit_2
    observables = [qml.PauliX(0), qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = local_su_2_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -2, atol=5e-2)


def test_optimization_exact_su_4_lie_optimizer_double_obs(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(0), qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = local_su_4_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-2)
    assert np.isclose(costs[-1], -2, atol=5e-2)
