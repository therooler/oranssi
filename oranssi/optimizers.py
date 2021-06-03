import pennylane as qml
import numpy as np
from typing import List, Tuple, Dict
import scipy.linalg as ssla
import matplotlib.pyplot as plt
from circuit_tools import get_full_operator, get_ops_from_qnode, circuit_state_from_unitary, \
    circuit_observable_from_unitary, param_shift_comm
from utils import get_su_2_operators, get_su_4_operators


def exact_lie_optimizer(circuit, params: List, observables: List, device: qml.Device, **kwargs) -> List[float]:
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
        assert circuit(params).return_type.name == 'State', f"`circuit` must return a state, received" \
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

    print(f"---------------------------------------------------")
    print(f"- Riemannian optimization with matrix exponential -")
    print(f"---------------------------------------------------")
    print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
    print(f"---------------------------------------------------")

    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    cost_exact = []
    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
    circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
    cost_exact.append(0)
    for o in observables:
        cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary, observable=o)

    for step in range(nsteps_optimizer):
        cost_exact.append(0)
        for o in observables:
            # Exact exponential
            phi = circuit_state_from_unitary_qnode(unitary=circuit_unitary)
            rho = np.outer(phi, phi.conj().T)
            H = param_shift_comm(rho, lambda t: ssla.expm(-1j * t / 2 * get_full_operator(o.matrix, o.wires, nqubits)))
            S, V = np.linalg.eigh(H)
            U_riemann_exact = (V @ np.diag(np.exp(-1j * eta / 2 * S)) @ V.conj().T)

            circuit_unitary = U_riemann_exact @ circuit_unitary

            cost_exact[step + 1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary, observable=o)
        if step > 2:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
                break
    print(f"Final cost = {cost_exact[-1]}")
    return cost_exact


def local_su_2_lie_optimizer(circuit, params: List, observables: List, device: qml.Device, **kwargs):
    """
    Riemannian gradient flow on the local unitary group. Implements U_{k+1} = exp(-ia [rho, O]) U_k by projecting
    the cost function onto SU(p)_loc = (X) SU(2) by way of the matrix exponential. Hardware friendly.

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
        assert circuit(params).return_type.name == 'State', f"`circuit` must return a state, received" \
                                                            f" {circuit(params).return_type}"
    else:
        raise AssertionError(f"`circuit` must return a state, received"
                             f"received {type(circuit(params))}")
    assert all(len(obs.wires) == 1 for obs in observables), 'Only single qubit observables are implemented currently'
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

    print(f"----------------------------------------------------------------")
    print(f"- Riemannian optimization on SU(2)_loc with matrix exponential -")
    print(f"----------------------------------------------------------------")
    print(f"nqubits = {nqubits} \nlearning rate = {eta} \nconvergence tolerance {tol}")
    print(f"----------------------------------------------------------------")

    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    cost_exact = []
    paulis = get_su_2_operators()

    circuit_state_from_unitary_qnode = qml.QNode(circuit_state_from_unitary, device)
    circuit_observable_from_unitary_qnode = qml.QNode(circuit_observable_from_unitary, device)
    cost_exact.append(0)
    for obs in observables:
        cost_exact[0] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary, observable=obs)
    for step in range(nsteps_optimizer):
        cost_exact.append(0)
        for i in range(nqubits):
            op = np.zeros((2 ** nqubits, 2 ** nqubits), dtype=complex)
            for obs in observables:
                omegas = []
                full_obs = get_full_operator(obs.matrix, obs.wires, nqubits)
                full_paulis = [get_full_operator(p, (i,), nqubits) for p in paulis]
                phi = circuit_state_from_unitary_qnode(unitary=circuit_unitary)[:, np.newaxis]
                for j, p in enumerate(full_paulis):
                    omegas.append(phi.conj().T @ (p @ full_obs - full_obs @ p) @ phi)
                op += omegas[0] * full_paulis[0] + omegas[1] * full_paulis[1] + omegas[2] * full_paulis[2]
            U_riemann_approx = ssla.expm(- eta / 2 ** nqubits * op)
            unitary_error = np.max(np.abs(U_riemann_approx @ U_riemann_approx.conj().T - np.eye(4, 4)))
            if unitary_error > 1e-8:
                print(f'WARNING: Unitary error = {unitary_error}, projecting onto unitary manifold by SVD')
                P, _, Q = np.linalg.svd(U_riemann_approx)
                U_riemann_approx = P @ Q
            circuit_unitary = U_riemann_approx @ circuit_unitary
        for o in observables:
            cost_exact[step + 1] += circuit_observable_from_unitary_qnode(unitary=circuit_unitary, observable=o)
        if step > 2:
            if np.isclose(cost_exact[-1], cost_exact[-2], atol=tol):
                print(f'Cost difference between steps < {tol}, stopping early at step {step}...')
                break
    print(f"Final cost = {cost_exact[-1]}")
    return cost_exact


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

    observables = [qml.PauliX(1)]
    params = [0.1, 1.2]
    costs_exact = exact_lie_optimizer(circuit, params, observables, dev, eta=0.2)
    costs_local = local_su_2_lie_optimizer(circuit, params, observables, dev, eta=0.2)
    cmap = plt.cm.get_cmap('Set1')
    plt.plot(costs_local, label=r'Lie $SU_{loc}(2^n)$', color=cmap(0.3))
    plt.plot(costs_exact, label=r'Lie $SU(2^n)$', color=cmap(0.2))
    plt.plot(range(20), [-1.0 for _ in range(20)], label='Min.', color='gray', linestyle='--')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(r'$\langle X_2 \rangle$')
    plt.title(fr'Different optimizers for $\eta = {0.2}$')
    plt.savefig('../data/figures' + f'/local_lie_optimizers_nq_{nqubits}_{0.2:1.3f}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
