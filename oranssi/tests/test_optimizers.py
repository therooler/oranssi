import pytest
import pennylane as qml
import numpy as np
from oranssi.optimizers import exact_lie_optimizer, local_su_2_lie_optimizer
from test_fixtures import circuit_1, circuit_1_bad_return_types


def test_observables_exact_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1), qml.Identity(0)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='Only Pauli Observables are supported'):
        exact_lie_optimizer(circuit, params, observables, device, eta=0.2)


def test_nsteps_exact_lie_optimizer(circuit_1):
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


def test_eta_exact_lie_optimizer(circuit_1):
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


def test_tol_exact_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`tol` must be an float between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, tol=-1.0)
    with pytest.raises(AssertionError, match='`tol` must be an float between 0 and infinity, '):
        exact_lie_optimizer(circuit, params, observables, device, tol=10)
    exact_lie_optimizer(circuit, params, observables, device, tol=0.10)


@pytest.mark.parametrize("circuit_1_bad_return_types", ['float', 'observable'], indirect=True)
def test_circuit_exact_lie_optimizer(circuit_1_bad_return_types):
    circuit, device, param_shape = circuit_1_bad_return_types
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`circuit` must return a state'):
        exact_lie_optimizer(circuit, params, observables, device, eta=0.2)


def test_optimization_exact_lie_optimizer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = exact_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -1, atol=1e-2)


@pytest.mark.parametrize("circuit_1_bad_return_types", ['float', 'observable'], indirect=True)
def test_circuit_exact_lie_optimizer(circuit_1_bad_return_types):
    circuit, device, param_shape = circuit_1_bad_return_types
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    with pytest.raises(AssertionError, match='`circuit` must return a state'):
        exact_lie_optimizer(circuit, params, observables, device, eta=0.2)
