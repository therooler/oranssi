import pennylane as qml
import numpy as np
import scipy.linalg as ssla

from typing import List
import itertools as it

from oranssi.circuit_tools import get_full_operator, get_ops_from_qnode, circuit_state_from_unitary, \
    circuit_observable_from_unitary, param_shift_comm
from oranssi.opt_tools import AdaptVQELayer, LieLayer


def parameter_shift_optimizer(circuit, params: List, observables: List, device: qml.Device,
                              **kwargs):
    """
    Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
    the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
    friendly.

    Args:
        circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
        params: List of parameters for the circuit. If no parameters, should be empty list.
        observables: List of PennyLane observables.
        device: PennyLane device.
        layer_patern: Pattern of the gates that should be applied
        **kwargs: Possible optimizer arguments:
            - nsteps: Maximum steps for the optimizer to take.
            - eta: Learning rate.
            - tol: Tolerance on the cost for early stopping.

    Returns:
        List of floats corresponding to the cost.
    """
    if hasattr(circuit(params), 'return_type'):
        raise AttributeError('`circuit` must not return anything')

    nqubits = len(device.wires)

    nsteps_optimizer = kwargs.get('nsteps', 40)
    assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
        f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
    eta = kwargs.get('eta', 0.1)
    assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
        f'`eta` must be an float between 0 and 1, received {eta}'
    tol = kwargs.get('tol', 1e-3)
    assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
        f'`tol` must be an float between 0 and infinity, received {tol}'
    return_state = kwargs.get('return_state', False)
    assert (isinstance(return_state, bool)), \
        f'`return_state` must be a boolean, received {return_state}'
    return_params = kwargs.get('return_params', False)
    assert (isinstance(return_state, bool)), \
        f'`return_params` must be a boolean, received {return_params}'
    if return_state:
        def circuit_state(params):
            circuit(params)
            return qml.state()

        circuit_state = qml.QNode(circuit_state, device)
    print(f"--------------------------------")
    print(f"- Parameter shift optimization -")
    print(f"--------------------------------")
    print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
    print(f"--------------------------------")

    cost_exact = []
    states = []
    params_per_step = []
    opt = qml.GradientDescentOptimizer(eta)
    H = qml.Hamiltonian([1.0 for _ in range(len(observables))], observables)
    cost_fn = qml.ExpvalCost(circuit, H, device)
    if return_state:
        states.append(circuit_state(params))
    if return_params:
        params_per_step.append(np.copy(params))

    cost_exact.append(cost_fn(params))

    for step in range(nsteps_optimizer):
        params, cost = opt.step_and_cost(cost_fn, params)
        cost_exact.append(cost_fn(params))
        if return_state:
            states.append(circuit_state(params))
        if return_params:
            params_per_step.append(np.copy(params))
        if step > 2:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
                break
    print(f"Final cost = {cost_exact[-1]}")
    returnables = [cost_exact, states, params_per_step]
    boolean_returnables = [True, return_state, return_params]
    if sum(boolean_returnables[1:]) < 1:
        return cost_exact
    else:
        return tuple(
            returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])


def exact_lie_optimizer(circuit, params: List, observables: List, device: qml.Device, **kwargs):
    """
    Riemannian gradient flow on the unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by way of the matrix
    exponential. Not hardware friendly.

    Args:
        circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
        params: List of parameters for the circuit. If no parameters, should be empty list.
        observables: List of PennyLane observables.
        device: PennyLane device.
        **kwargs: Possible optimizer arguments:
            - nsteps: Maximum steps for the optimizer to take.
            - eta: Learning rate.
            - tol: Tolerance on the cost for early stopping.

    Returns:
        List of floats corresponding to the cost.
    """
    # assert all(isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ)) for o in observables), \
    #     f"Only Pauli Observables are supported, received " \
    #     f"{[o for o in observables if not isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ))]}"
    if hasattr(circuit(params), 'return_type'):
        assert circuit(
            params).return_type.name == 'State', f"`circuit` must return a state, received" \
                                                 f" {circuit(params).return_type}"
    else:
        raise AssertionError(f"`circuit` must return a state, "
                             f"received {type(circuit(params))}")
    circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
    nqubits = len(device.wires)

    nsteps_optimizer = kwargs.get('nsteps', 40)
    assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
        f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
    eta = kwargs.get('eta', 0.1)
    assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
        f'`eta` must be an float between 0 and 1, received {eta}'
    tol = kwargs.get('tol', 1e-3)
    assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
        f'`tol` must be an float between 0 and infinity, received {tol}'
    return_state = kwargs.get('return_state', False)
    assert (isinstance(return_state, bool) & (0. <= tol <= np.inf)), \
        f'`return_state` must be a boolean, received {return_state}'
    perturbation = kwargs.get('perturbation', False)
    assert isinstance(perturbation, bool), \
        f'`perturbation` must be a boolean, received {perturbation}'
    return_perturbations = kwargs.get('return_perturbations', False)
    if return_perturbations:
        assert perturbation, f'`return_perturbations` is {return_perturbations}, but `perturbation` is {perturbation}'
    assert isinstance(return_perturbations, bool), \
        f'`return_perturbations` must be a boolean, received {return_perturbations}'

    return_unitary = kwargs.get('return_unitary', False)
    assert isinstance(return_unitary, bool), \
        f'`return_unitary` must be a boolean, received {return_unitary}'

    print(f"------------------------------------------------------------")
    print(f"- Riemannian optimization on SU(p) with matrix exponential -")
    print(f"------------------------------------------------------------")
    print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
    print(f"------------------------------------------------------------")
    # convert the circuit to a single unitary
    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    # Initialize qnodes
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
    circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)

    # initializze cost
    cost_exact = []
    states = []
    perturbations = []
    if return_state:
        states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
    cost_exact.append(0)
    for o in observables:
        cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
                                                               observable=o)
    # optimization
    for step in range(nsteps_optimizer):
        cost_exact.append(0)
        for o in observables:
            # get rho from the circuit
            phi = circuit_state_from_unitary_qnode(unitary=circuit_unitary)
            rho = np.outer(phi, phi.conj().T)
            # calculate the commutator on the algebra
            H = param_shift_comm(rho, lambda t: ssla.expm(
                -1j * t / 2 * get_full_operator(o.matrix, o.wires, nqubits)))
            # avoid matrix exponential by diagonalizing
            S, V = np.linalg.eigh(H)
            U_riemann_exact = (V @ np.diag(np.exp(-1j * eta / 2 * S)) @ V.conj().T)
            # update the circuit unitary
            circuit_unitary = U_riemann_exact @ circuit_unitary
        if return_state:
            states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
        for o in observables:
            # update cost
            cost_exact[step + 1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
                                                                          observable=o)
        # check early stopping.
        if step > 3:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                if perturbation:
                    perturbations.append(step)
                    print(f'Cost difference between steps < {tol}, perturbing at step {step}...')
                    random_matrix = np.random.randn(2 ** nqubits, 2 ** nqubits) * 0.1
                    random_matrix = 0.5 * (random_matrix - random_matrix.T)
                    H = -1j * random_matrix
                    # avoid matrix exponential by diagonalizing
                    S, V = np.linalg.eigh(H)
                    U_riemann_exact = (V @ np.diag(np.exp(-1j * eta / 2 * S)) @ V.conj().T)
                    # update the circuit unitary
                    circuit_unitary = U_riemann_exact @ circuit_unitary
                else:
                    print(
                        f'Cost difference between steps < {tol}, stopping early at step {step}...')
                    break
    print(f"Final cost = {cost_exact[-1]}")
    returnables = [cost_exact, states, perturbations, circuit_unitary]
    boolean_returnables = [True, return_state, return_perturbations, return_unitary]
    if sum(boolean_returnables[1:]) < 1:
        return cost_exact
    else:
        return tuple(
            returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])


def approximate_lie_optimizer(circuit, params: List, observables: List, device: qml.Device, layers,
                              **kwargs):
    """
    Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
    the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
    friendly.

    Args:
        circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
        params: List of parameters for the circuit. If no parameters, should be empty list.
        observables: List of PennyLane observables.
        device: PennyLane device.
        layer_pattern: Pattern of the gates that should be applied
        **kwargs: Possible optimizer arguments:
            - nsteps: Maximum steps for the optimizer to take.
            - eta: Learning rate.
            - tol: Tolerance on the cost for early stopping.

    Returns:
        List of floats corresponding to the cost.
    """
    if hasattr(circuit(params), 'return_type'):
        assert circuit(
            params).return_type.name == 'State', f"`circuit` must return a state, received" \
                                                 f" {circuit(params).return_type}"
    else:
        raise AssertionError(f"`circuit` must return a state, "
                             f"received {type(circuit(params))}")

    circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
    nqubits = len(device.wires)
    print(kwargs)
    nsteps_optimizer = kwargs.get('nsteps', 40)
    assert isinstance(layers, list), f'layers must be a list, received type {type(layers)}'
    assert all(
        isinstance(l, LieLayer) for l in layers), 'all layers must be instances of `LieLayer`'
    assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
        f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
    eta = kwargs.get('eta', 0.1)
    assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
        f'`eta` must be an float between 0 and 1, received {eta}'
    tol = kwargs.get('tol', 1e-3)
    if tol is not None:
        assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
            f'`tol` must be an float between 0 and infinity, received {tol}'

    escape_tol = kwargs.get('escape_tol', 1e-3)
    assert (isinstance(escape_tol, float)), \
        f'`escape_tol` must be a boolean, received {escape_tol}'
    return_state = kwargs.get('return_state', False)
    assert (isinstance(return_state, bool)), \
        f'`return_state` must be a boolean, received {return_state}'
    return_omegas = kwargs.get('return_omegas', False)
    assert (isinstance(return_omegas, bool)), \
        f'`return_state` must be a boolean, received {return_omegas}'
    return_unitary = kwargs.get('return_unitary', False)
    assert (isinstance(return_unitary, bool)), \
        f'`return_unitary` must be a boolean, received {return_unitary}'
    return_gates = kwargs.get('return_gates', False)
    assert (isinstance(return_gates, bool)), \
        f'`return_gates` must be a boolean, received {return_gates}'
    perturb = kwargs.get('perturb', False)
    assert (isinstance(perturb, bool)), \
        f'`perturb` must be a boolean, received {perturb}'
    return_perturbations = kwargs.get('return_perturbations', False)
    assert (isinstance(return_perturbations, bool)), \
        f'`return_perturbations` must be a boolean, received {return_perturbations}'

    lie_layers = it.cycle(layers)

    # get circuit as numpy array
    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    # initialize returnables
    cost_exact = []
    states = []
    gates = []
    omegas = []
    unitaries = []
    perturbations = []

    # Initialize qnodes
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
    circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)

    # Print layer
    print(f"Approximate Lie Optimizer - nqubits = {nqubits}")
    print("Returnables: cost_exact: True, state: {}, omegas: {}, unitary: {}, gates: {}] ".format(
        return_state, return_omegas, return_unitary, return_gates))
    print('-' * 80)
    print('|', ('{:^25}|' * 3).format('name', 'stride', 'Trotterize'))
    for l in layers:
        print('-' * 80)
        print(l)
        print('-' * 80)

    if return_state:
        states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
    if return_unitary:
        unitaries.append(circuit_unitary)
    # initialize cost
    cost_exact.append(0)
    for obs in observables:
        cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
                                                               observable=obs)
    for step in range(nsteps_optimizer):
        cost_exact.append(0)
        layer = next(lie_layers)
        if isinstance(layer, AdaptVQELayer):
            circuit_unitary, params = layer(circuit_unitary, optimizer=parameter_shift_optimizer,
                                            params=params, **kwargs)
        else:
            circuit_unitary = layer(circuit_unitary, **kwargs)

        for o in observables:
            cost_exact[step + 1] += circuit_observable_from_unitary_qnode(
                unitary=circuit_unitary,
                observable=o)

        # get returnables
        if return_state:
            states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
        if return_omegas:
            omegas.append(layer.get_lie_algebra_directions(circuit_unitary))
        if return_unitary:
            unitaries.append(circuit_unitary)

        # early stopping
        if (tol is not None) & (step > 8):
            if np.isclose(cost_exact[-1], cost_exact[-6], atol=tol):
                if perturb:
                    perturbations.append(step)
                    print(f'Cost difference between steps < {tol}, perturbing at step {step}...')
                    random_matrix = np.random.randn(2 ** nqubits, 2 ** nqubits) * 0.1
                    random_matrix = 0.5 * (random_matrix - random_matrix.T)
                    H = -1j * random_matrix
                    # avoid matrix exponential by diagonalizing
                    S, V = np.linalg.eigh(H)
                    U_riemann_exact = (V @ np.diag(np.exp(-1j * eta / 2 * S)) @ V.conj().T)
                    # update the circuit unitary
                    circuit_unitary = U_riemann_exact @ circuit_unitary
                else:
                    print(
                        f'Cost difference between steps < {tol}, stopping early at step {step}...')
                    break
    print(f"Final cost = {cost_exact[-1]}")
    returnables = [cost_exact, states, np.array(omegas), unitaries, gates, perturbations]
    boolean_returnables = [True, return_state, return_omegas, return_unitary, return_gates,
                           return_perturbations]
    if sum(boolean_returnables[1:]) < 1:
        return cost_exact
    else:
        return tuple(
            returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])
#
#
# def local_su_2_lie_optimizer(circuit, params: List, observables: List, device: qml.Device,
#                              **kwargs) -> List[float]:
#     """
#     Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
#     the cost function onto SU(p)_loc_2 = (X) SU(2) by way of the matrix exponential. Not hardware
#     friendly.
#
#     Args:
#         circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
#         params: List of parameters for the circuit. If no parameters, should be empty list.
#         observables: List of PennyLane observables.
#         device: PennyLane device.
#         **kwargs: Possible optimizer arguments:
#             - nsteps: Maximum steps for the optimizer to take.
#             - eta: Learning rate.
#             - tol: Tolerance on the cost for early stopping.
#
#     Returns:
#         List of floats corresponding to the cost.
#     """
#     return local_custom_su_lie_optimizer(circuit, params, observables, device,
#                                          layer_pattern=[(1, 0)], **kwargs)
#
#
# def local_su_4_lie_optimizer(circuit, params: List, observables: List, device: qml.Device,
#                              **kwargs) -> List[float]:
#     """
#     Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
#     the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
#     friendly.
#
#     Args:
#         circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
#         params: List of parameters for the circuit. If no parameters, should be empty list.
#         observables: List of PennyLane observables.
#         device: PennyLane device.
#         **kwargs: Possible optimizer arguments:
#             - nsteps: Maximum steps for the optimizer to take.
#             - eta: Learning rate.
#             - tol: Tolerance on the cost for early stopping.
#
#     Returns:
#         List of floats corresponding to the cost.
#     """
#
#     return local_custom_su_lie_optimizer(circuit, params, observables, device,
#                                          layer_pattern=[(2, 0)], **kwargs)
#
#
# def local_custom_su_lie_optimizer(circuit, params: List, observables: List, device: qml.Device,
#                                   layer_pattern: Any, **kwargs):
#     """
#     Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
#     the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
#     friendly.
#
#     Args:
#         circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
#         params: List of parameters for the circuit. If no parameters, should be empty list.
#         observables: List of PennyLane observables.
#         device: PennyLane device.
#         layer_pattern: Pattern of the gates that should be applied
#         **kwargs: Possible optimizer arguments:
#             - nsteps: Maximum steps for the optimizer to take.
#             - eta: Learning rate.
#             - tol: Tolerance on the cost for early stopping.
#
#     Returns:
#         List of floats corresponding to the cost.
#     """
#     # assert all(isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ)) for o in observables), \
#     #     f"Only Pauli Observables are supported, received " \
#     #     f"{[o for o in observables if not isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ))]}"
#     if hasattr(circuit(params), 'return_type'):
#         assert circuit(
#             params).return_type.name == 'State', f"`circuit` must return a state, received" \
#                                                  f" {circuit(params).return_type}"
#     else:
#         raise AssertionError(f"`circuit` must return a state, "
#                              f"received {type(circuit(params))}")
#     # assert all(len(obs.wires) == 1 for obs in
#     #            observables), 'Only single qubit observables are implemented currently'
#     circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
#     nqubits = len(device.wires)
#
#     assert isinstance(layer_pattern, Iterable) & all(isinstance(i, tuple) for i in layer_pattern), \
#         f'`layer_pattern` must be an iterable of tuples, received {layer_pattern}'
#     assert all(len(i) == 2 for i in layer_pattern), \
#         f'`layer_pattern` must be an iterable of tuples of length 2, received {layer_pattern}'
#     assert all(i[0] in [1, 2] for i in
#                layer_pattern), 'The tuples in `layer_patern` must have as first entry integers ' \
#                                'in [1, 2] that indicate the whether we apply a SU(2) or SU(4) ' \
#                                f'layer, received {layer_pattern}'
#     assert all(i[1] in [0, 1] for i in
#                layer_pattern), 'The tuples in `layer_patern` must have as first entry integers ' \
#                                'in [0, ] that indicate the stride of the layer, ' \
#                                f'received {layer_pattern}'
#
#     nsteps_optimizer = kwargs.get('nsteps', 40)
#     assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
#         f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
#     eta = kwargs.get('eta', 0.1)
#     assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
#         f'`eta` must be an float between 0 and 1, received {eta}'
#     tol = kwargs.get('tol', 1e-3)
#     assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
#         f'`tol` must be an float between 0 and infinity, received {tol}'
#     return_state = kwargs.get('return_state', False)
#     assert (isinstance(return_state, bool)), \
#         f'`return_state` must be a boolean, received {return_state}'
#     return_omegas = kwargs.get('return_omegas', False)
#     assert (isinstance(return_omegas, bool)), \
#         f'`return_state` must be a boolean, received {return_omegas}'
#     directions = kwargs.pop('directions', None)
#     if directions is not None:
#         assert isinstance(directions,
#                           Iterable), f'`directions` must be an iterable, received {directions}'
#         assert all((all(isinstance(ds, str) for ds in d)) for d in directions), \
#             f'`directions` must be an iterable of iterables of strings, received {directions}'
#         assert len(directions) == len(layer_pattern), '`directions` must have equal length to ' \
#                                                       '`layer_pattern`'
#     adaptive = kwargs.pop('adaptive', False)
#     assert (isinstance(adaptive, bool)), \
#         f'`adaptive` must be a boolean, received {adaptive}'
#     return_unitaries = kwargs.get('return_unitaries', False)
#     assert (isinstance(return_unitaries, bool)), \
#         f'`return_unitaries` must be a boolean, received {return_unitaries}'
#
#     print(f"-------------------------------------------------------------------")
#     print(f"- Riemannian optimization on custom SU(p) with matrix exponential -")
#     print(f"-------------------------------------------------------------------")
#     print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
#     print(f"-------------------------------------------------------------------")
#
#     circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
#     for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
#         circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary
#
#     cost_exact = []
#     states = []
#     omegas = []
#     unitaries = []
#     circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
#     circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
#
#     lie_layers = []
#     for i, locstr in enumerate(layer_pattern):
#         locality, stride = locstr
#         kwargs['stride'] = stride
#         if directions is not None:
#             lie_layers.append(
#                 LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, locality,
#                                      nqubits,
#                                      directions=directions[i], **kwargs))
#         else:
#             lie_layers.append(
#                 LocalLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, locality,
#                                      nqubits,
#                                      **kwargs))
#
#     print(f"Lie Layer model - nqubits = {nqubits}")
#     print('-' * 80)
#     print('|', ('{:^25}|' * 3).format('name', 'stride', 'Trotterize'))
#     print('-' * 80)
#     for layer in lie_layers:
#         print(layer)
#     print('-' * 80)
#
#     lie_layers = it.cycle(lie_layers)
#
#     if return_state:
#         states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#     cost_exact.append(0)
#     for obs in observables:
#         cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                observable=obs)
#     if return_unitaries:
#         unitaries.append(circuit_unitary)
#     for step in range(nsteps_optimizer):
#         cost_exact.append(0)
#         layer = next(lie_layers)
#         if adaptive:
#             print("Adaptive step")
#             adaptive_costs = []
#             circuit_unitary = layer(circuit_unitary)
#             adaptive_costs.append(0)
#             for o in observables:
#                 adaptive_costs[0] += circuit_observable_from_unitary_qnode(
#                     unitary=circuit_unitary,
#                     observable=o)
#             adaptive_step = 1
#             while True:
#                 circuit_unitary = layer(circuit_unitary)
#                 adaptive_costs.append(0)
#                 for o in observables:
#                     adaptive_costs[adaptive_step] += circuit_observable_from_unitary_qnode(
#                         unitary=circuit_unitary,
#                         observable=o)
#                 print(adaptive_costs[-1])
#                 adaptive_step += 1
#                 if np.isclose(adaptive_costs[-1], adaptive_costs[-2], atol=tol):
#                     break
#             cost_exact[step + 1] = np.copy(adaptive_costs[-1])
#         else:
#             circuit_unitary = layer(circuit_unitary)
#             for o in observables:
#                 cost_exact[step + 1] += circuit_observable_from_unitary_qnode(
#                     unitary=circuit_unitary,
#                     observable=o)
#         if return_state:
#             states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#         if return_omegas:
#             omegas.append(layer.get_lie_algebra_directions(circuit_unitary))
#         if return_unitaries:
#             unitaries.append(circuit_unitary)
#         if step > 8:
#             if np.isclose(cost_exact[-1], cost_exact[-6], atol=tol):
#                 print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
#                 break
#     print(f"Final cost = {cost_exact[-1]}")
#     returnables = [cost_exact, states, np.array(omegas), unitaries]
#     boolean_returnables = [True, return_state, return_omegas, return_unitaries]
#     if sum(boolean_returnables[1:]) < 1:
#         return cost_exact
#     else:
#         return tuple(
#             returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])
#
#
# def algebra_custom_su_lie_optimizer(circuit, params: List, observables: List, device: qml.Device,
#                                     directions: Any, **kwargs):
#     """
#     Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
#     the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
#     friendly.
#
#     Args:
#         circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
#         params: List of parameters for the circuit. If no parameters, should be empty list.
#         observables: List of PennyLane observables.
#         device: PennyLane device.
#         directions: List of strings indicating the Lie algebra directions. If None, take all
#         single and double (odd+even) directions on su(2) and su(4) respectively.
#
#         **kwargs: Possible optimizer arguments:
#             - nsteps: Maximum steps for the optimizer to take.
#             - eta: Learning rate.
#             - tol: Tolerance on the cost for early stopping.
#
#     Returns:
#         List of floats corresponding to the cost.
#     """
#
#     if hasattr(circuit(params), 'return_type'):
#         assert circuit(
#             params).return_type.name == 'State', f"`circuit` must return a state, received" \
#                                                  f" {circuit(params).return_type}"
#     else:
#         raise AssertionError(f"`circuit` must return a state, "
#                              f"received {type(circuit(params))}")
#
#     circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
#     nqubits = len(device.wires)
#
#     nsteps_optimizer = kwargs.get('nsteps', 40)
#     assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
#         f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
#     eta = kwargs.get('eta', 0.1)
#     assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
#         f'`eta` must be an float between 0 and 1, received {eta}'
#     tol = kwargs.get('tol', 1e-3)
#     assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
#         f'`tol` must be an float between 0 and infinity, received {tol}'
#     return_state = kwargs.get('return_state', False)
#     assert (isinstance(return_state, bool)), \
#         f'`return_state` must be a boolean, received {return_state}'
#     return_omegas = kwargs.get('return_omegas', False)
#     assert (isinstance(return_omegas, bool)), \
#         f'`return_state` must be a boolean, received {return_omegas}'
#     return_unitaries = kwargs.get('return_unitaries', False)
#     assert (isinstance(return_unitaries, bool)), \
#         f'`return_unitaries` must be a boolean, received {return_unitaries}'
#     if directions is not None:
#         assert isinstance(directions,
#                           Iterable), f'`directions` must be an iterable, received {directions}'
#         assert all(isinstance(d, str) for d in directions), \
#             f'`directions` must be an iterable of iterables of strings, received {directions}'
#
#     adaptive = kwargs.pop('adaptive', False)
#     assert (isinstance(adaptive, bool)), \
#         f'`adaptive` must be a boolean, received {adaptive}'
#
#     print(f"-------------------------------------------------------------------")
#     print(f"- Riemannian optimization on custom SU(p) with matrix exponential -")
#     print(f"-------------------------------------------------------------------")
#     print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
#     print(f"-------------------------------------------------------------------")
#
#     circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
#     for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
#         circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary
#
#     cost_exact = []
#     states = []
#     omegas = []
#     unitaries = []
#
#     circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
#     circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
#
#     layer = CustomDirectionLieAlgebraLayer(circuit_state_from_unitary_qnode, observables,
#                                            directions,
#                                            nqubits, **kwargs)
#
#     print(f"Lie Layer model - nqubits = {nqubits}")
#     print('-' * 80)
#     print('|', ('{:^25}|' * 3).format('name', 'stride', 'Trotterize'))
#     print('-' * 80)
#     print(layer)
#     print('-' * 80)
#
#     if return_state:
#         states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#     cost_exact.append(0)
#     for obs in observables:
#         cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                observable=obs)
#     if return_unitaries:
#         unitaries.append(circuit_unitary)
#
#     for step in range(nsteps_optimizer):
#         cost_exact.append(0)
#         if adaptive:
#             print("Adaptive step")
#             adaptive_costs = []
#             circuit_unitary = layer(circuit_unitary)
#             adaptive_costs.append(0)
#             for o in observables:
#                 adaptive_costs[0] += circuit_observable_from_unitary_qnode(
#                     unitary=circuit_unitary,
#                     observable=o)
#             adaptive_step = 1
#             while True:
#                 circuit_unitary = layer(circuit_unitary)
#                 adaptive_costs.append(0)
#                 for o in observables:
#                     adaptive_costs[adaptive_step] += circuit_observable_from_unitary_qnode(
#                         unitary=circuit_unitary,
#                         observable=o)
#                 print(adaptive_costs[-1])
#                 adaptive_step += 1
#                 if np.isclose(adaptive_costs[-1], adaptive_costs[-2], atol=tol):
#                     break
#             cost_exact[step + 1] = np.copy(adaptive_costs[-1])
#         else:
#             circuit_unitary = layer(circuit_unitary)
#             for o in observables:
#                 cost_exact[step + 1] += circuit_observable_from_unitary_qnode(
#                     unitary=circuit_unitary,
#                     observable=o)
#         if return_unitaries:
#             unitaries.append(circuit_unitary)
#         if return_state:
#             states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#         if return_omegas:
#             omegas.append(layer.get_lie_algebra_directions(circuit_unitary))
#
#         if step > 8:
#             if np.isclose(cost_exact[-1], cost_exact[-6], atol=tol):
#                 print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
#                 break
#     print(f"Final cost = {cost_exact[-1]}")
#     returnables = [cost_exact, states, np.array(omegas), unitaries]
#     boolean_returnables = [True, return_state, return_omegas, return_unitaries]
#     if sum(boolean_returnables[1:]) < 1:
#         return cost_exact
#     else:
#         return tuple(
#             returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])
#
#
# def algebra_stochastic_su_lie_optimizer(circuit, params: List, observables: List,
#                                         device: qml.Device, **kwargs):
#     """
#     Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
#     the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
#     friendly.
#
#     Args:
#         circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
#         params: List of parameters for the circuit. If no parameters, should be empty list.
#         observables: List of PennyLane observables.
#         device: PennyLane device.
#         directions: List of strings indicating the Lie algebra directions. If None, take all
#         single and double (odd+even) directions on su(2) and su(4) respectively.
#
#         **kwargs: Possible optimizer arguments:
#             - nsteps: Maximum steps for the optimizer to take.
#             - eta: Learning rate.
#             - tol: Tolerance on the cost for early stopping.
#
#     Returns:
#         List of floats corresponding to the cost.
#     """
#
#     if hasattr(circuit(params), 'return_type'):
#         assert circuit(
#             params).return_type.name == 'State', f"`circuit` must return a state, received" \
#                                                  f" {circuit(params).return_type}"
#     else:
#         raise AssertionError(f"`circuit` must return a state, "
#                              f"received {type(circuit(params))}")
#
#     circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
#     nqubits = len(device.wires)
#
#     nsteps_optimizer = kwargs.get('nsteps', 40)
#     assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
#         f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
#     eta = kwargs.get('eta', 0.1)
#     assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
#         f'`eta` must be an float between 0 and 1, received {eta}'
#     tol = kwargs.get('tol', 1e-3)
#     assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
#         f'`tol` must be an float between 0 and infinity, received {tol}'
#     return_state = kwargs.get('return_state', False)
#     assert (isinstance(return_state, bool)), \
#         f'`return_state` must be a boolean, received {return_state}'
#     return_omegas = kwargs.get('return_omegas', False)
#     assert (isinstance(return_omegas, bool)), \
#         f'`return_state` must be a boolean, received {return_omegas}'
#     return_unitary = kwargs.get('return_unitary', False)
#     assert (isinstance(return_unitary, bool)), \
#         f'`return_unitary` must be a boolean, received {return_unitary}'
#
#     print(f"-------------------------------------------------------------------------")
#     print(f"- Riemannian optimization on stochastic SU_4(p) with matrix exponential -")
#     print(f"-------------------------------------------------------------------------")
#     print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
#     print(f"-------------------------------------------------------------------------")
#
#     circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
#     for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
#         circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary
#
#     cost_exact = []
#     states = []
#     omegas = []
#     unitaries = []
#
#     circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
#     circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
#
#     layer = StochasticLieAlgebraLayer(circuit_state_from_unitary_qnode, observables, nqubits,
#                                       **kwargs)
#
#     print(f"Stochastic Lie Layer model - nqubits = {nqubits}")
#     print('-' * 80)
#     print('|', ('{:^25}|' * 3).format('name', 'stride', 'Trotterize'))
#     print('-' * 80)
#     print(layer)
#     print('-' * 80)
#
#     if return_state:
#         states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#     cost_exact.append(0)
#     for obs in observables:
#         cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                observable=obs)
#
#     p_perturb_final = 0.00
#     p_perturb_annealed = 1.0
#     for step in range(nsteps_optimizer):
#         cost_exact.append(0)
#         p_perturb_annealed *= 0.99
#         p_perturb = p_perturb_final + (1 - p_perturb_final) * p_perturb_annealed
#         perturb = np.random.choice((True, False), p=[p_perturb, 1 - p_perturb])
#         # perturb=False
#         circuit_unitary = layer(circuit_unitary, new_direction=True, perturb=perturb)
#         print(step, ' - ', layer.current_direction, layer.current_qubits)
#         adaptive_costs = []
#         adaptive_costs.append(0)
#         for o in observables:
#             adaptive_costs[0] += circuit_observable_from_unitary_qnode(
#                 unitary=circuit_unitary,
#                 observable=o)
#         adaptive_step = 1
#         while True:
#             circuit_unitary = layer(circuit_unitary, new_direction=False, perturb=perturb)
#             adaptive_costs.append(0)
#             for o in observables:
#                 adaptive_costs[adaptive_step] += circuit_observable_from_unitary_qnode(
#                     unitary=circuit_unitary,
#                     observable=o)
#             adaptive_step += 1
#             if perturb:
#                 print(
#                     f'Perturbed, cost start = {adaptive_costs[0]}, cost stop = {adaptive_costs[-1]}')
#                 break
#             if np.isclose(adaptive_costs[-1], adaptive_costs[-2], atol=tol):
#                 print(
#                     f'Stopped after {adaptive_step}, cost start = {adaptive_costs[0]}, cost stop = {adaptive_costs[-1]}')
#                 break
#         cost_exact[step + 1] = np.copy(adaptive_costs[-1])
#
#         if return_state:
#             states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#         if return_omegas:
#             omegas.append(layer.get_lie_algebra_directions(circuit_unitary))
#         if return_unitary:
#             unitaries.append(circuit_unitary)
#         if step > 30:
#             if np.isclose(cost_exact[-1], cost_exact[-30], atol=tol):
#                 print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
#                 break
#     print(f"Final cost = {cost_exact[-1]}")
#     returnables = [cost_exact, states, np.array(omegas), unitaries]
#     boolean_returnables = [True, return_state, return_omegas, return_unitary]
#     if sum(boolean_returnables[1:]) < 1:
#         return cost_exact
#     else:
#         return tuple(
#             returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])
#
#
# def algebra_squared_su_lie_optimizer(circuit, params: List, observables: List,
#                                      device: qml.Device, **kwargs):
#     """
#     Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
#     the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
#     friendly.
#
#     Args:
#         circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
#         params: List of parameters for the circuit. If no parameters, should be empty list.
#         observables: List of PennyLane observables.
#         device: PennyLane device.
#         directions: List of strings indicating the Lie algebra directions. If None, take all
#         single and double (odd+even) directions on su(2) and su(4) respectively.
#
#         **kwargs: Possible optimizer arguments:
#             - nsteps: Maximum steps for the optimizer to take.
#             - eta: Learning rate.
#             - tol: Tolerance on the cost for early stopping.
#
#     Returns:
#         List of floats corresponding to the cost.
#     """
#
#     if hasattr(circuit(params), 'return_type'):
#         assert circuit(
#             params).return_type.name == 'State', f"`circuit` must return a state, received" \
#                                                  f" {circuit(params).return_type}"
#     else:
#         raise AssertionError(f"`circuit` must return a state, "
#                              f"received {type(circuit(params))}")
#
#     circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
#     nqubits = len(device.wires)
#     print(kwargs)
#     nsteps_optimizer = kwargs.get('nsteps', 40)
#
#     assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
#         f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
#     eta = kwargs.get('eta', 0.1)
#     assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
#         f'`eta` must be an float between 0 and 1, received {eta}'
#     tol = kwargs.get('tol', 1e-3)
#     assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
#         f'`tol` must be an float between 0 and infinity, received {tol}'
#     escape_tol = kwargs.get('escape_tol', 1e-3)
#     assert (isinstance(escape_tol, float)), \
#         f'`escape_tol` must be a boolean, received {escape_tol}'
#     return_state = kwargs.get('return_state', False)
#     assert (isinstance(return_state, bool)), \
#         f'`return_state` must be a boolean, received {return_state}'
#     return_omegas = kwargs.get('return_omegas', False)
#     assert (isinstance(return_omegas, bool)), \
#         f'`return_state` must be a boolean, received {return_omegas}'
#     return_unitary = kwargs.get('return_unitary', False)
#     assert (isinstance(return_unitary, bool)), \
#         f'`return_unitary` must be a boolean, received {return_unitary}'
#     return_gates = kwargs.get('return_gates', False)
#     assert (isinstance(return_gates, bool)), \
#         f'`return_gates` must be a boolean, received {return_gates}'
#
#     print(f"-------------------------------------------------------------------------")
#     print(f"- Riemannian optimization on stochastic SU_4(p) with matrix exponential -")
#     print(f"-------------------------------------------------------------------------")
#     print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
#     print(f"-------------------------------------------------------------------------")
#
#     circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
#     for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
#         circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary
#
#     cost_exact = []
#     states = []
#     omegas = []
#     unitaries = []
#
#     circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
#     circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
#
#     layer = SquaredLieAlgebraLayer(circuit_state_from_unitary_qnode,
#                                    circuit_observable_from_unitary_qnode, observables, nqubits,
#                                    **kwargs)
#
#     print(f"Square Lie Layer model - nqubits = {nqubits}")
#     print('-' * 80)
#     print('|', ('{:^25}|' * 3).format('name', 'stride', 'Trotterize'))
#     print('-' * 80)
#     print(layer)
#     print('-' * 80)
#
#     if return_state:
#         states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#     cost_exact.append(0)
#     for obs in observables:
#         cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                observable=obs)
#     gates = []
#     print(nsteps_optimizer)
#     for step in range(nsteps_optimizer):
#         cost_exact.append(0)
#
#         # circuit_unitary = layer(circuit_unitary, new_direction=True)
#         print(step, ' - ', layer.current_pauli)
#         # adaptive_step = 1
#         circuit_unitary, gate = layer(circuit_unitary, new_direction=False,
#                                       optimizer=parameter_shift_optimizer, **kwargs)
#         gates.append(gate)
#         for obs in observables:
#             cost_exact[-1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                     observable=obs)
#         # print(adaptive_costs[-1])
#         # while True:
#         #     circuit_unitary = layer(circuit_unitary, new_direction=False)
#         #     adaptive_costs.append(0)
#         #     for o in observables:
#         #         adaptive_costs[adaptive_step] += circuit_observable_from_unitary_qnode(
#         #             unitary=circuit_unitary,
#         #             observable=o)
#         #     adaptive_step += 1
#         #     if np.isclose(adaptive_costs[-1], adaptive_costs[-2], atol=tol):
#         #         print(
#         #             f'Stopped after {adaptive_step}, cost start = {adaptive_costs[0]}, cost stop = {adaptive_costs[-1]}')
#         #         break
#         # cost_exact[step + 1] = np.copy(adaptive_costs[-1])
#         if np.isclose(cost_exact[-1], cost_exact[-2], atol=escape_tol):
#             print('Optimization stuck, attemping escape...')
#             circuit_unitary = layer(circuit_unitary, escape=True)
#         if return_state:
#             states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#         if return_omegas:
#             omegas.append(layer.get_lie_algebra_directions(circuit_unitary))
#         if return_unitary:
#             unitaries.append(circuit_unitary)
#         if step > 30:
#             if np.isclose(cost_exact[-1], cost_exact[-30], atol=tol):
#                 print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
#                 break
#     print(f"Final cost = {cost_exact[-1]}")
#     returnables = [cost_exact, states, np.array(omegas), unitaries, gates]
#     boolean_returnables = [True, return_state, return_omegas, return_unitary, return_gates]
#     if sum(boolean_returnables[1:]) < 1:
#         return cost_exact
#     else:
#         return tuple(
#             returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])
#
#
# def adaptive_vqe_su_lie_optimizer(circuit, params: List, observables: List,
#                                   device: qml.Device, **kwargs):
#     """
#     Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
#     the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
#     friendly.
#
#     Args:
#         circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
#         params: List of parameters for the circuit. If no parameters, should be empty list.
#         observables: List of PennyLane observables.
#         device: PennyLane device.
#         directions: List of strings indicating the Lie algebra directions. If None, take all
#         single and double (odd+even) directions on su(2) and su(4) respectively.
#
#         **kwargs: Possible optimizer arguments:
#             - nsteps: Maximum steps for the optimizer to take.
#             - eta: Learning rate.
#             - tol: Tolerance on the cost for early stopping.
#
#     Returns:
#         List of floats corresponding to the cost.
#     """
#
#     if hasattr(circuit(params), 'return_type'):
#         assert circuit(
#             params).return_type.name == 'State', f"`circuit` must return a state, received" \
#                                                  f" {circuit(params).return_type}"
#     else:
#         raise AssertionError(f"`circuit` must return a state, "
#                              f"received {type(circuit(params))}")
#
#     circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
#     nqubits = len(device.wires)
#     print(kwargs)
#     nsteps_optimizer = kwargs.get('nsteps', 40)
#
#     assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
#         f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
#     eta = kwargs.get('eta', 0.1)
#     assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
#         f'`eta` must be an float between 0 and 1, received {eta}'
#     tol = kwargs.get('tol', 1e-3)
#     assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
#         f'`tol` must be an float between 0 and infinity, received {tol}'
#     escape_tol = kwargs.get('escape_tol', 1e-3)
#     assert (isinstance(escape_tol, float)), \
#         f'`escape_tol` must be a boolean, received {escape_tol}'
#     return_state = kwargs.get('return_state', False)
#     assert (isinstance(return_state, bool)), \
#         f'`return_state` must be a boolean, received {return_state}'
#     return_omegas = kwargs.get('return_omegas', False)
#     assert (isinstance(return_omegas, bool)), \
#         f'`return_state` must be a boolean, received {return_omegas}'
#     return_unitary = kwargs.get('return_unitary', False)
#     assert (isinstance(return_unitary, bool)), \
#         f'`return_unitary` must be a boolean, received {return_unitary}'
#     return_gates = kwargs.get('return_gates', False)
#     assert (isinstance(return_gates, bool)), \
#         f'`return_gates` must be a boolean, received {return_gates}'
#
#     print(f"-------------------------------------------------------------------------")
#     print(f"- Riemannian optimization on stochastic SU_4(p) with matrix exponential -")
#     print(f"-------------------------------------------------------------------------")
#     print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
#     print(f"-------------------------------------------------------------------------")
#
#     circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
#     for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
#         circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary
#
#     cost_exact = []
#     states = []
#     omegas = []
#     unitaries = []
#
#     circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
#     circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
#
#     layer = AdaptVQELayer(device, observables, **kwargs)
#
#     print(f"Square Lie Layer model - nqubits = {nqubits}")
#     print('-' * 80)
#     print('|', ('{:^25}|' * 3).format('name', 'stride', 'Trotterize'))
#     print('-' * 80)
#     print(layer)
#     print('-' * 80)
#
#     if return_state:
#         states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#     cost_exact.append(0)
#     for obs in observables:
#         cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                observable=obs)
#     gates = []
#     print(nsteps_optimizer)
#
#     for step in range(nsteps_optimizer):
#         cost_exact.append(0)
#
#         # circuit_unitary = layer(circuit_unitary, new_direction=True)
#         print(step, ' - ', layer.previous_direction)
#         # adaptive_step = 1
#         circuit_unitary, params = layer(circuit_unitary, new_direction=False, circuit=circuit,
#                                         optimizer=parameter_shift_optimizer, params=params,
#                                         **kwargs)
#         print(params)
#         for obs in observables:
#             cost_exact[-1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                     observable=obs)
#         if return_state:
#             states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#         if return_omegas:
#             omegas.append(layer.get_lie_algebra_directions(circuit_unitary))
#         if return_unitary:
#             unitaries.append(circuit_unitary)
#         if step > 30:
#             if np.isclose(cost_exact[-1], cost_exact[-30], atol=tol):
#                 print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
#                 break
#     print(f"Final cost = {cost_exact[-1]}")
#     returnables = [cost_exact, states, np.array(omegas), unitaries, gates]
#     boolean_returnables = [True, return_state, return_omegas, return_unitary, return_gates]
#     if sum(boolean_returnables[1:]) < 1:
#         return cost_exact
#     else:
#         return tuple(
#             returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])
#
#
# def su8_lie_optimizer(circuit, params: List, observables: List,
#                       device: qml.Device, **kwargs):
#     """
#     Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
#     the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
#     friendly.
#
#     Args:
#         circuit: Function with signature (params, **kwargs) that returns a PennyLane state or observable.
#         params: List of parameters for the circuit. If no parameters, should be empty list.
#         observables: List of PennyLane observables.
#         device: PennyLane device.
#         directions: List of strings indicating the Lie algebra directions. If None, take all
#         single and double (odd+even) directions on su(2) and su(4) respectively.
#
#         **kwargs: Possible optimizer arguments:
#             - nsteps: Maximum steps for the optimizer to take.
#             - eta: Learning rate.
#             - tol: Tolerance on the cost for early stopping.
#
#     Returns:
#         List of floats corresponding to the cost.
#     """
#
#     if hasattr(circuit(params), 'return_type'):
#         assert circuit(
#             params).return_type.name == 'State', f"`circuit` must return a state, received" \
#                                                  f" {circuit(params).return_type}"
#     else:
#         raise AssertionError(f"`circuit` must return a state, "
#                              f"received {type(circuit(params))}")
#
#     circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
#     nqubits = len(device.wires)
#     print(kwargs)
#     nsteps_optimizer = kwargs.get('nsteps', 40)
#
#     assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
#         f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
#     eta = kwargs.get('eta', 0.1)
#     assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
#         f'`eta` must be an float between 0 and 1, received {eta}'
#     tol = kwargs.get('tol', 1e-3)
#     assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
#         f'`tol` must be an float between 0 and infinity, received {tol}'
#     escape_tol = kwargs.get('escape_tol', 1e-3)
#     assert (isinstance(escape_tol, float)), \
#         f'`escape_tol` must be a boolean, received {escape_tol}'
#     return_state = kwargs.get('return_state', False)
#     assert (isinstance(return_state, bool)), \
#         f'`return_state` must be a boolean, received {return_state}'
#     return_omegas = kwargs.get('return_omegas', False)
#     assert (isinstance(return_omegas, bool)), \
#         f'`return_state` must be a boolean, received {return_omegas}'
#     return_unitary = kwargs.get('return_unitary', False)
#     assert (isinstance(return_unitary, bool)), \
#         f'`return_unitary` must be a boolean, received {return_unitary}'
#     return_gates = kwargs.get('return_gates', False)
#     assert (isinstance(return_gates, bool)), \
#         f'`return_gates` must be a boolean, received {return_gates}'
#
#     print(f"-------------------------------------------------------------------------")
#     print(f"- Riemannian optimization on stochastic SU_8(p) with matrix exponential -")
#     print(f"-------------------------------------------------------------------------")
#     print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
#     print(f"-------------------------------------------------------------------------")
#
#     circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
#     for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
#         circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary
#
#     cost_exact = []
#     states = []
#     omegas = []
#     unitaries = []
#
#     circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
#     circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
#
#     layer = SU8_AlgebraLayer(circuit_state_from_unitary_qnode,
#                              circuit_observable_from_unitary_qnode, observables, nqubits,
#                              **kwargs)
#
#     print(f"SU(8) Lie Layer model - nqubits = {nqubits}")
#     print('-' * 80)
#     print('|', ('{:^25}|' * 3).format('name', 'stride', 'Trotterize'))
#     print('-' * 80)
#     print(layer)
#     print('-' * 80)
#
#     if return_state:
#         states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#     cost_exact.append(0)
#     for obs in observables:
#         cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                observable=obs)
#     gates = []
#     print(nsteps_optimizer)
#
#     for step in range(nsteps_optimizer):
#         cost_exact.append(0)
#
#         # circuit_unitary = layer(circuit_unitary, new_direction=True)
#         print(step, ' - ', layer.previous_pauli)
#         # adaptive_step = 1
#         circuit_unitary, params = layer(circuit_unitary, new_direction=False, circuit=circuit,
#                                         optimizer=parameter_shift_optimizer, params=params,
#                                         **kwargs)
#         for obs in observables:
#             cost_exact[-1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
#                                                                     observable=obs)
#         if return_state:
#             states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
#         if return_omegas:
#             omegas.append(layer.get_lie_algebra_directions(circuit_unitary))
#         if return_unitary:
#             unitaries.append(circuit_unitary)
#         if step > 30:
#             if np.isclose(cost_exact[-1], cost_exact[-30], atol=tol):
#                 print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
#                 break
#     print(f"Final cost = {cost_exact[-1]}")
#     returnables = [cost_exact, states, np.array(omegas), unitaries, gates]
#     boolean_returnables = [True, return_state, return_omegas, return_unitary, return_gates]
#     if sum(boolean_returnables[1:]) < 1:
#         return cost_exact
#     else:
#         return tuple(
#             returnable for r, returnable in enumerate(returnables) if boolean_returnables[r])
