import pytest
import pennylane as qml
import numpy as np
from oranssi.optimizers import exact_lie_optimizer, approximate_lie_optimizer, \
    parameter_shift_optimizer
from oranssi.opt_tools import LieLayer, LocalLieAlgebraLayer, CustomDirectionLieAlgebraLayer, StochasticLieAlgebraLayer, AdaptVQELayer, SU8_AlgebraLayer
from test_fixtures import circuit_1, circuit_2, circuit_4, circuit_1_bad_return_types


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


### Test optimization ###

def test_optimization_exact_lie_optimizer_single_obs(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = exact_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -1, atol=1e-2)


def test_optimization_exact_lie_optimizer_double_obs(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(0), qml.PauliX(1)]
    params = [0.1, 1.2]
    costs = exact_lie_optimizer(circuit, params, observables, device, eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -2, atol=1e-2)

def test_optimization_approx_lie_optimizer_single_obs_LocalLieAlgebraLayer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    layers = [LocalLieAlgebraLayer(device ,observables, locality=2, nqubits=2)]
    costs = approximate_lie_optimizer(circuit, params, observables, device, layers=layers,eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -1, atol=1e-2)

def test_optimization_approx_lie_optimizer_single_obs_CustomDirectionLieAlgebraLayer(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    layers = [CustomDirectionLieAlgebraLayer(device ,observables, directions=['X', 'Z', 'XX'])]
    costs = approximate_lie_optimizer(circuit, params, observables, device, layers=layers,eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -1, atol=1e-2)

def test_optimization_approx_lie_optimizer_single_obs_CustomDirectionLieAlgebraLayer_trot(circuit_1):
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    layers = [CustomDirectionLieAlgebraLayer(device ,observables, trotterize=True, directions=['X', 'Z', 'XX'])]
    costs = approximate_lie_optimizer(circuit, params, observables, device, layers=layers,eta=0.2, tol=1e-3)
    assert np.isclose(costs[-1], -1, atol=1e-2)

def test_optimization_approx_lie_optimizer_single_obs_StochasticLieAlgebraLayer(circuit_1):
    np.random.seed(200)
    circuit, device, param_shape = circuit_1
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    layers = [StochasticLieAlgebraLayer(device, observables)]
    costs = approximate_lie_optimizer(circuit, params, observables, device, layers=layers,
                                      eta=0.2, tol=None, nsteps=100)
    assert np.isclose(costs[-1], -1, atol=0.25)

def test_optimization_approx_lie_optimizer_single_obs_AdaptVQE(circuit_4):
    np.random.seed(200)
    circuit, device, param_shape = circuit_4
    observables = [qml.PauliX(1)]
    params = [[0.1, 1.2]]
    layers = [AdaptVQELayer(device, observables)]
    costs = approximate_lie_optimizer(circuit, params, observables, device, layers=layers,
                                      eta=0.2, tol=None, nsteps=10)
    assert np.isclose(costs[-1], -1, atol=0.25)


def test_optimization_approx_lie_optimizer_single_obs_SU8_Layer(circuit_2):
    np.random.seed(200)
    circuit, device, param_shape = circuit_2
    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    layers = [SU8_AlgebraLayer(device, observables)]
    costs = approximate_lie_optimizer(circuit, params, observables, device, layers=layers,
                                      eta=0.2, tol=None, nsteps=10)
    assert np.isclose(costs[-1], -1, atol=0.10)
