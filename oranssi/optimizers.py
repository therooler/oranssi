import pennylane as qml
import numpy as np
from typing import List
import scipy.linalg as ssla
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
    assert all(isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ)) for o in observables), \
        f"Only Pauli Observables are supported, received " \
        f"{[o for o in observables if not isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ))]}"
    if hasattr(circuit(params), 'return_type'):
        assert circuit(
            params).return_type.name == 'State', f"`circuit` must return a state, received" \
                                                 f" {circuit(params).return_type}"
    else:
        raise AssertionError(f"`circuit` must return a state, received"
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

    print(f"------------------------------------------------------------")
    print(f"- Riemannian optimization on SU(p) with matrix exponential -")
    print(f"------------------------------------------------------------")
    print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
    print(f"---------------------------------------------------")
    # convert the circuit to a single unitary
    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary
    # Initialize qnodes
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
    circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
    # initializze cost
    cost_exact = []
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
            # update the circuit unitary :TODO is this correct? Do we split on observables?
            circuit_unitary = U_riemann_exact @ circuit_unitary
            # update cost
            cost_exact[step + 1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
                                                                          observable=o)
        # check early stopping.
        if step > 2:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
                break
    print(f"Final cost = {cost_exact[-1]}")
    return cost_exact


def local_su_2_lie_optimizer(circuit, params: List, observables: List, device: qml.Device, **kwargs) -> List[float]:
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
    assert all(isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ)) for o in observables), \
        f"Only Pauli Observables are supported, received " \
        f"{[o for o in observables if not isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ))]}"
    if hasattr(circuit(params), 'return_type'):
        assert circuit(
            params).return_type.name == 'State', f"`circuit` must return a state, received" \
                                                 f" {circuit(params).return_type}"
    else:
        raise AssertionError(f"`circuit` must return a state, received"
                             f"received {type(circuit(params))}")
    assert all(len(obs.wires) == 1 for obs in
               observables), 'Only single qubit observables are implemented currently'
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

    print(f"------------------------------------------------------------------")
    print(f"- Riemannian optimization on SU(p)_loc_2 with matrix exponential -")
    print(f"------------------------------------------------------------------")
    print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
    print(f"----------------------------------------------------------------")

    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    cost_exact = []

    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
    circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
    lie_layer = LocalLieLayer(circuit_state_from_unitary_qnode, observables, 1, nqubits, eta=eta)

    cost_exact.append(0)
    for obs in observables:
        cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
                                                               observable=obs)
    for step in range(nsteps_optimizer):
        cost_exact.append(0)
        circuit_unitary = lie_layer(circuit_unitary)
        for o in observables:
            cost_exact[step + 1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary, observable=o)
        if step > 2:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
                break
    print(f"Final cost = {cost_exact[-1]}")
    return cost_exact


def local_su_4_lie_optimizer(circuit, params: List, observables: List, device: qml.Device, **kwargs) -> List[float]:
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
    assert all(isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ)) for o in observables), \
        f"Only Pauli Observables are supported, received " \
        f"{[o for o in observables if not isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ))]}"
    if hasattr(circuit(params), 'return_type'):
        assert circuit(
            params).return_type.name == 'State', f"`circuit` must return a state, received" \
                                                 f" {circuit(params).return_type}"
    else:
        raise AssertionError(f"`circuit` must return a state, received"
                             f"received {type(circuit(params))}")
    assert all(len(obs.wires) == 1 for obs in
               observables), 'Only single qubit observables are implemented currently'
    circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, device)
    nqubits = len(device.wires)
    assert nqubits / 2 == nqubits // 2, f"`nqubits` must be even, received {nqubits}"
    nsteps_optimizer = kwargs.get('nsteps', 40)
    assert (isinstance(nsteps_optimizer, int) & (1 <= nsteps_optimizer <= np.inf)), \
        f'`nsteps` must be an integer between 0 and infinity, received {nsteps_optimizer}'
    eta = kwargs.get('eta', 0.1)
    assert (isinstance(eta, float) & (0. <= eta <= 1.)), \
        f'`eta` must be an float between 0 and 1, received {eta}'
    tol = kwargs.get('tol', 1e-3)
    assert (isinstance(tol, float) & (0. <= tol <= np.inf)), \
        f'`tol` must be an float between 0 and infinity, received {tol}'

    print(f"------------------------------------------------------------------")
    print(f"- Riemannian optimization on SU(p)_loc_4 with matrix exponential -")
    print(f"------------------------------------------------------------------")
    print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
    print(f"----------------------------------------------------------------")

    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    cost_exact = []

    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
    circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
    lie_layer = LocalLieLayer(circuit_state_from_unitary_qnode, observables, 2, nqubits, eta=eta)

    cost_exact.append(0)
    for obs in observables:
        cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary,
                                                               observable=obs)
    for step in range(nsteps_optimizer):
        cost_exact.append(0)
        circuit_unitary = lie_layer(circuit_unitary)
        for o in observables:
            cost_exact[step + 1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary, observable=o)
        if step > 2:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
                break
    print(f"Final cost = {cost_exact[-1]}")
    return cost_exact


class LocalLieLayer(object):
    def __init__(self, state_qnode, observables, locality: int, nqubits: int, **kwargs):
        self.state_qnode = state_qnode
        assert all(isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ)) for o in observables), \
            f"Only Pauli Observables are supported, received " \
            f"{[o for o in observables if not isinstance(o, (qml.PauliX, qml.PauliY, qml.PauliZ))]}"
        assert all(len(obs.wires) == 1 for obs in
                   observables), 'Only single qubit observables are implemented currently'
        self.observables = observables

        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        assert locality in [1, 2], f'Only SU(2) and SU(4) local are supported with `locality` in ' \
                                   f'[0,1] respectively, received `locality` = {locality}'
        self.locality = locality
        if locality == 2:
            assert (nqubits / 2 == nqubits // 2), f"`nqubits` must be even, received {nqubits}"
            self.paulis = get_su_4_operators()
            self.stride = kwargs.get('stride', 0)
        else:
            self.paulis = get_su_2_operators()
            self.stride = 0
        self.nqubits = nqubits
        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        self.full_paulis = []
        if self.locality == 1:
            for i in range(self.nqubits):
                self.full_paulis.append([get_full_operator(p, (i,), self.nqubits) for p in self.paulis])
        elif self.locality == 2:
            if self.stride == 0:
                for i in range(0, nqubits, 2):
                    self.full_paulis.append([get_full_operator(p, (i, i + 1), self.nqubits) for p in self.paulis])
            else:
                for i in range(1, nqubits, 2):
                    self.full_paulis.append([get_full_operator(p, (i, i + 1), self.nqubits) for p in self.paulis])
                self.full_paulis.append([get_full_operator(p, (nqubits - 1, 0), self.nqubits) for p in self.paulis])
        self.op = np.zeros((2 ** self.nqubits, 2 ** self.nqubits), dtype=complex)

    def __call__(self, circuit_unitary, *args, **kwargs):
        for full_paulis in self.full_paulis:
            self.op.fill(0)
            for obs in self.observables:
                omegas = []
                full_obs = get_full_operator(obs.matrix, obs.wires, self.nqubits)

                phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
                for j, pauli in enumerate(full_paulis):
                    omegas.append(phi.conj().T @ (pauli @ full_obs - full_obs @ pauli) @ phi)
                self.op += sum(omegas[i] * pauli for i, pauli in enumerate(full_paulis))
            U_riemann_approx = ssla.expm(- self.eta / 2 ** self.nqubits * self.op)
            if self.unitary_error_check:
                unitary_error = np.max(
                    np.abs(U_riemann_approx @ U_riemann_approx.conj().T - np.eye(2 ** self.nqubits, **self.nqubits)))
                if unitary_error > 1e-8:
                    print(f'WARNING: Unitary error = {unitary_error}, projecting onto unitary manifold by SVD')
                    P, _, Q = np.linalg.svd(U_riemann_approx)
                    U_riemann_approx = P @ Q
            circuit_unitary = U_riemann_approx @ circuit_unitary

        return circuit_unitary
