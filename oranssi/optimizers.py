import pennylane as qml
import numpy as np
import scipy.linalg as ssla

from typing import List, Any
from collections.abc import Iterable
from itertools import cycle

from oranssi.circuit_tools import get_full_operator, get_ops_from_qnode, circuit_state_from_unitary, \
    circuit_observable_from_unitary, param_shift_comm
from oranssi.utils import get_su_2_operators, get_su_4_operators


def exact_lie_optimizer(circuit, params: List, observables: List, device: qml.Device, **kwargs) -> \
        List[float]:
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
        if step > 2:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
                break
    print(f"Final cost = {cost_exact[-1]}")
    if return_state:
        return cost_exact, states
    else:
        return cost_exact


def local_su_2_lie_optimizer(circuit, params: List, observables: List, device: qml.Device,
                             **kwargs) -> List[float]:
    """
    Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
    the cost function onto SU(p)_loc_2 = (X) SU(2) by way of the matrix exponential. Not hardware
    friendly.

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
    return local_custom_su_lie_optimizer(circuit, params, observables, device,
                                         layer_pattern=[(1, 0)], **kwargs)


def local_su_4_lie_optimizer(circuit, params: List, observables: List, device: qml.Device,
                             **kwargs) -> List[float]:
    """
    Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
    the cost function onto SU(p)_loc_4 = (X) SU(4) by way of the matrix exponential. Not hardware
    friendly.

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

    return local_custom_su_lie_optimizer(circuit, params, observables, device,
                                         layer_pattern=[(2, 0)], **kwargs)


def local_custom_su_lie_optimizer(circuit, params: List, observables: List, device: qml.Device,
                                  layer_pattern: Any, **kwargs):
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
    # assert all(len(obs.wires) == 1 for obs in
    #            observables), 'Only single qubit observables are implemented currently'
    circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
    nqubits = len(device.wires)

    assert isinstance(layer_pattern, Iterable) & all(isinstance(i, tuple) for i in layer_pattern), \
        f'`layer_pattern` must be an iterable of tuples, received {layer_pattern}'
    assert all(len(i) == 2 for i in layer_pattern), \
        f'`layer_pattern` must be an iterable of tuples of length 2, received {layer_pattern}'
    assert all(i[0] in [1, 2] for i in
               layer_pattern), 'The tuples in `layer_patern` must have as first entry integers ' \
                               'in [1, 2] that indicate the whether we apply a SU(2) or SU(4) ' \
                               f'layer, received {layer_pattern}'
    assert all(i[1] in [0, 1] for i in
               layer_pattern), 'The tuples in `layer_patern` must have as first entry integers ' \
                               'in [0, ] that indicate the stride of the layer, ' \
                               f'received {layer_pattern}'

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
    return_omegas = kwargs.get('return_omegas', False)
    assert (isinstance(return_omegas, bool)), \
        f'`return_state` must be a boolean, received {return_omegas}'
    directions = kwargs.pop('directions', None)

    if directions is not None:
        assert isinstance(directions,
                          Iterable), f'`directions` must be an iterable, received {directions}'
        assert all((all(isinstance(ds,str) for ds in d))for d in directions), \
            f'`directions` must be an iterable of iterables of strings, received {directions}'
        assert len(directions) == len(layer_pattern), '`directions` must have equal length to ' \
                                                      '`layer_pattern`'
    print(f"-------------------------------------------------------------------")
    print(f"- Riemannian optimization on custom SU(p) with matrix exponential -")
    print(f"-------------------------------------------------------------------")
    print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
    print(f"-------------------------------------------------------------------")

    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    cost_exact = []
    states = []
    omegas = []

    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
    circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)

    lie_layers = []
    for i, locstr in enumerate(layer_pattern):
        locality, stride = locstr
        kwargs['stride'] = stride
        if directions is not None:
            lie_layers.append(
                LocalLieLayer(circuit_state_from_unitary_qnode, observables, locality, nqubits,
                              directions=directions[i],**kwargs))
        else:

            lie_layers.append(
                LocalLieLayer(circuit_state_from_unitary_qnode, observables, locality, nqubits,
                              **kwargs))

    print(f"Lie Layer model - nqubits = {nqubits}")
    print('-' * 50)
    print('|', ('{:^15}|' * 3).format('name', 'stride', 'Trotterize'))
    print('-' * 50)
    for layer in lie_layers:
        print(layer)
    print('-' * 50)

    lie_layers = cycle(lie_layers)

    if return_state:
        states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
    cost_exact.append(0)
    for obs in observables:
        cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
                                                               observable=obs)
    for step in range(nsteps_optimizer):
        cost_exact.append(0)
        layer = next(lie_layers)
        circuit_unitary = layer(circuit_unitary)
        if return_state:
            states.append(circuit_state_from_unitary_qnode(unitary=circuit_unitary))
        if return_omegas:
            omegas.append(layer.get_lie_algebra_directions(circuit_unitary))
        for o in observables:
            cost_exact[step + 1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
                                                                          observable=o)
        if step > 2:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
                break
    print(f"Final cost = {cost_exact[-1]}")
    if return_state:
        return cost_exact, states
    if return_omegas:
        return cost_exact, np.array(omegas)
    if (return_state & return_omegas):
        return cost_exact, states, np.array(omegas)
    else:
        return cost_exact


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
    if (return_state & return_params):
        return cost_exact, states, params_per_step
    elif return_params:
        return cost_exact, params_per_step
    elif return_state:
        return cost_exact, states
    else:
        return cost_exact


class LocalLieLayer(object):
    def __init__(self, state_qnode, observables: List, locality: int, nqubits: int, **kwargs):
        """
        Class that applies a Riemannian optimization step on a SU(2) or SU(4) local manifold.
        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            state_qnode: QNode of a circuit that takes a unitary and returns a state.
            observables: List of single qubit Pauli observables.
            locality: Either 1 or 2, indicating SU(2) or SU(4) local.
            nqubits: The number of qubits in the circuit.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - stride (SU(4) only) : Integer that indicates wether to start on qubit 0 or 1 with applying the
                SU(4) operators
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
                -
        """
        self.state_qnode = state_qnode
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        assert locality in [1, 2], f'Only SU(2) and SU(4) local are supported with `locality` in ' \
                                   f'[1,2] respectively, received `locality` = {locality}'
        self.locality = locality

        if locality == 2:
            assert (nqubits / 2 == nqubits // 2), f"`nqubits` must be even, received {nqubits}"
            self.paulis, self.directions = get_su_4_operators(return_names=True)

            self.stride = kwargs.get('stride', 0)
            assert self.stride in [0, 1], f'`stride` must be in [0,1], received {self.stride}'
        else:
            self.paulis, self.directions = get_su_2_operators(return_names=True)
            self.stride = 0
        directions = kwargs.get('directions', None)
        if directions is not None:
            assert all(d in self.directions for d in directions), \
                f'Supplied Lie algebra directions are invalid, ' \
                f'expected {self.directions}, received {directions}'
            new_directions = []
            new_paulis = []
            for pauli, d in zip(self.paulis, self.directions):
                if d in directions:
                    new_directions.append(d)
                    new_paulis.append(pauli)
            self.paulis = new_paulis
            self.directions = new_directions

        self.nqubits = nqubits
        self.observables = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                            observables]

        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'
        # depending on the locality, create the full pauli matrices required to calculate the commutators
        self.full_paulis = []
        if self.locality == 1:
            for i in range(self.nqubits):
                self.full_paulis.append(
                    [get_full_operator(p, (i,), self.nqubits) for p in self.paulis])
        elif self.locality == 2:
            if self.stride == 0:
                for i in range(0, nqubits, 2):
                    self.full_paulis.append(
                        [get_full_operator(p, (i, i + 1), self.nqubits) for p in self.paulis])
            else:
                for i in range(self.stride, nqubits - 1, 2):
                    self.full_paulis.append(
                        [get_full_operator(p, (i, i + 1), self.nqubits) for p in self.paulis])
                self.full_paulis.append(
                    [get_full_operator(p, (nqubits - 1, 0), self.nqubits) for p in self.paulis])
        self.op = np.zeros((2 ** self.nqubits, 2 ** self.nqubits), dtype=complex)

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]

        for obs in self.observables:
            for full_paulis in self.full_paulis:
                self.op.fill(0)
                omegas = []
                if self.trotterize:
                    for j, pauli in enumerate(full_paulis):
                        omega = phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi
                        self.op = omega * pauli
                        U_riemann_approx = ssla.expm(- self.eta / 2 ** self.nqubits * self.op)
                        if (self.unitary_error_check) and (self._is_unitary(U_riemann_approx)):
                            U_riemann_approx = self._project_onto_unitary(U_riemann_approx)
                        circuit_unitary = U_riemann_approx @ circuit_unitary
                        self.op.fill(0)
                else:
                    for j, pauli in enumerate(full_paulis):
                        omegas.append(phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi)
                    self.op += sum(omegas[i] * pauli for i, pauli in enumerate(full_paulis))
                    U_riemann_approx = ssla.expm(- self.eta / 2 ** self.nqubits * self.op)
                    if (self.unitary_error_check) and (self._is_unitary(U_riemann_approx)):
                        U_riemann_approx = self._project_onto_unitary(U_riemann_approx)
                    circuit_unitary = U_riemann_approx @ circuit_unitary

        return circuit_unitary

    def _is_unitary(self, U):
        """
        Check that the matrix U is unitary.

        Args:
            U: M x M complex matrix.

        Returns:
            Boolean that indicates whether U is unitary
        """
        unitary_error = np.max(
            np.abs(U @ U.conj().T - np.eye(2 ** self.nqubits, 2 ** self.nqubits)))
        if unitary_error > 1e-8:
            print(
                f'WARNING: Unitary error = {unitary_error}, projecting onto unitary manifold by SVD')
            return False
        else:
            return True

    def _project_onto_unitary(self, U):
        """
        Use singular value decomposition to project the matrix U onto the Unitary manifold.

        Args:
            U: M x M complex matrix.

        Returns:
            M x M complex unitary matrix.
        """
        P, _, Q = np.linalg.svd(U)
        return P @ Q

    def __repr__(self):
        row_format = "{:^15}|" * 3
        return "| " + \
               row_format.format(f'Lie Layer SU({2 ** self.locality})', self.stride,
                                 self.trotterize) + " directions -> " + ", ".join(self.directions)

    def get_lie_algebra_directions(self, circuit_unitary):

        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        omegas = np.zeros((len(self.observables), len(self.full_paulis), len(self.full_paulis[0])))
        for o, obs in enumerate(self.observables):
            for p, full_paulis in enumerate(self.full_paulis):
                self.op.fill(0)
                for j, pauli in enumerate(full_paulis):
                    omegas[o, p, j] = (phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi)[0, 0].imag
        return omegas

    def get_lie_algebra_directions_strings(self):
        return self.directions
