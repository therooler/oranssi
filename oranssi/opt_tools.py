import copy

import numpy as np
import scipy.linalg as ssla
import pennylane as qml

from typing import List
import itertools as it

from oranssi.circuit_tools import get_full_operator, get_commuting_set, get_ops_from_qnode, \
    circuit_state_from_unitary, circuit_observable_from_unitary, get_all_su_n_directions
from oranssi.utils import get_su_2_operators, get_su_4_operators, get_su_n_operators
from oranssi.pauli import Pauli, PauliMonomial, pauli_idx_str2int_map, pauli_idx_int2str_map
import sympy as sp
from sympy.utilities.lambdify import lambdify

from scipy.optimize import minimize

class LieLayer(object):
    def __init__(self, device, observables: List, nqubits: int):
        """
        Abstract class that contains some basic functionality for LieLayers

        Args:
            device: PennyLane device.
            observables: List of single qubit Pauli observables.
            nqubits: The number of qubits in the circuit.
                -
        """
        self.device = device
        self.state_qnode = qml.QNode(circuit_state_from_unitary, device)
        self.obs_qnode = qml.QNode(circuit_observable_from_unitary, device)
        self.su_state = None
        self.nqubits = nqubits
        self.observables = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                            observables]

    def __call__(self, circuit_unitary, *args, **kwargs):
        raise NotImplementedError

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
        """
        String representation.
        """
        raise NotImplementedError

    def set_eta(self, eta):
        """
        Set the learning rate

        Args:
            eta: float that sets the learning rate.

        """
        self.eta = np.copy(eta)

    def get_lie_algebra_directions(self, circuit_unitary):
        """
        Return the Riemanian gradient in all directions of the Lie algebra given a unitary.

        Args:
            circuit_unitary: Matrix corresponiding to the unitary of the circuit.

        Returns:
            Dictionary containing the directions and corresponding SU(2^N) directions.
        """
        return get_all_su_n_directions(circuit_unitary, self.observables, self.device)


class LocalLieAlgebraLayer(LieLayer):
    def __init__(self, device, observables: List, locality: int, **kwargs):
        """
        Class that applies a Riemannian optimization step on a SU(2) or SU(4) local manifold.

        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            state_qnode: QNode of a circuit that takes a unitary and returns a state.
            observables: List of single qubit Pauli observables.
            locality: Either 1 or 2, indicating SU(2) or SU(4) local approximation.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - stride (SU(4) only) : Integer that indicates wether to start on qubit 0 or 1 with applying the
                SU(4) operators
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
        """
        self.nqubits = len(device.wires)
        super().__init__(device, observables, self.nqubits)
        self.eta = kwargs.get('eta', 0.1)

        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        assert locality in [1, 2], f'Only SU(2) and SU(4) local are supported with `locality` in ' \
                                   f'[1,2] respectively, received `locality` = {locality}'
        self.locality = locality

        if locality == 2:
            assert ((self.nqubits / 2) == (
                    self.nqubits // 2)), f"`nqubits` must be even, received {self.nqubits}"
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
            for comb in it.combinations(range(self.nqubits), r=2):
                self.full_paulis.append(
                    [get_full_operator(p, (comb[0], comb[1]), self.nqubits) for p in self.paulis])
        self.op = np.zeros((2 ** self.nqubits, 2 ** self.nqubits), dtype=complex)

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        for oi, obs in enumerate(self.observables):
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
                    # omegas = np.array(omegas) /( self.eta/ 2 ** self.nqubits +1e-9)
                    self.op = sum(omegas[i] * full_paulis[i] for i in range(len(full_paulis)))
                    U_riemann_approx = ssla.expm(-self.eta / 2 ** self.nqubits * self.op)
                    if (self.unitary_error_check) and (self._is_unitary(U_riemann_approx)):
                        U_riemann_approx = self._project_onto_unitary(U_riemann_approx)
                    circuit_unitary = U_riemann_approx @ circuit_unitary

        return circuit_unitary

    def __repr__(self):
        row_format = "{:^25}|" * 3
        return "| " + \
               row_format.format(f'Local Lie Algebra Layer SU({2 ** self.locality})', self.stride,
                                 self.trotterize) + " directions -> " + ", ".join(self.directions)

    def get_lie_algebra_directions_strings(self):
        return self.directions


class CustomDirectionLieAlgebraLayer(LieLayer):
    def __init__(self, device, observables: List, directions: List[str], **kwargs):
        """
        Class that applies a Riemannian optimization step on pre-specified Lie algebra.

        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            device: PennyLane device.
            observables: List of single qubit Pauli observables.
            directions: List of strings containing the allowed directions.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
                -
        """
        self.nqubits = len(device.wires)
        super().__init__(device, observables, self.nqubits)

        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'

        self.lastate = AlgebraSU4(self.nqubits, add_su2=True)
        self.lastate.get_all_directions_and_qubits()
        self.directions =list(self.lastate.full_paulis.keys())
        self.paulis = self.lastate.full_paulis
        if directions is not None:
            assert all(d in self.lastate.directions for d in directions), \
                f'Supplied Lie algebra directions are invalid, ' \
                f'expected {self.lastate.directions}, received {directions}'
            new_directions = []
            new_paulis = []
            for d, pauli in self.paulis.items():
                print(d)
                if d[0] in directions:
                    new_directions.append(d)
                    new_paulis.append(pauli)
            self.paulis = new_paulis
            self.directions = new_directions
        self.full_paulis = dict(zip(self.directions, self.paulis))
        self.observables = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                            observables]

        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'

        self.op = np.zeros((2 ** self.nqubits, 2 ** self.nqubits), dtype=complex)

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        for oi, obs in enumerate(self.observables):
            if self.trotterize:
                for k, pauli in self.full_paulis.items():
                    self.op.fill(0)
                    # phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
                    omega = phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi
                    self.op = omega * pauli
                    U_riemann_approx = ssla.expm(- self.eta / 2 ** self.nqubits * self.op)
                    if (self.unitary_error_check) and (self._is_unitary(U_riemann_approx)):
                        U_riemann_approx = self._project_onto_unitary(U_riemann_approx)
                    circuit_unitary = U_riemann_approx @ circuit_unitary
                    self.op.fill(0)
            else:
                omegas = []
                for k, pauli in self.full_paulis.items():
                    omegas.append(phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi)
                    print(k, omegas[-1])
                self.op = sum(
                    omegas[i] * pauli for i, pauli in enumerate(self.full_paulis.values()))
                U_riemann_approx = ssla.expm(-self.eta / 2 ** self.nqubits * self.op)
                if (self.unitary_error_check) and (self._is_unitary(U_riemann_approx)):
                    U_riemann_approx = self._project_onto_unitary(U_riemann_approx)
                circuit_unitary = U_riemann_approx @ circuit_unitary

        return circuit_unitary

    def __repr__(self):
        row_format = "{:^25}|" * 3
        return "| " + \
               row_format.format(f'Lie Algebra Layer SU({2 ** self.nqubits})', 'NaN',
                                 self.trotterize) + " directions -> " + ", ".join([d[0] for d in self.directions])

    def get_lie_algebra_directions_strings(self):
        return self.directions


class StochasticLieAlgebraLayer(LieLayer):
    def __init__(self, device, observables: List, **kwargs):
        """
        Class that applies a Riemannian optimization step by randomly selecting a direction on the
        local SU(2)+SU(4) Lie algebra.

        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            device: PennyLane device
            observables: List of single qubit Pauli observables.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
                -
        """
        self.nqubits = len(device.wires)
        super().__init__(device, observables, self.nqubits)

        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.lastate = AlgebraSU4(self.nqubits, add_su2=True)

        self.observables = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                            observables]
        # print(np.linalg.eigvalsh(np.sum(self.observables, axis=0)))

        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'
        # initialize
        self.current_direction, self.current_qubits = None, None

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]

        direction, qubits = self.lastate.get_random_direction_and_qubits()
        self.current_direction = direction
        self.current_qubits = qubits

        for oi, obs in enumerate(self.observables):
            pauli = self.lastate.full_paulis[(self.current_direction, *self.current_qubits)]
            omega = phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi
            U_riemann_approx = ssla.expm(- self.eta / 2 ** self.nqubits * omega * pauli)
            if (self.unitary_error_check) and (self._is_unitary(U_riemann_approx)):
                U_riemann_approx = self._project_onto_unitary(U_riemann_approx)
            circuit_unitary = U_riemann_approx @ circuit_unitary

        return circuit_unitary

    def __repr__(self):
        row_format = "{:^25}|" * 3
        return "| " + \
               row_format.format(f'Stoch. Algebra Layer SU({2 ** self.nqubits})', 'NaN',
                                 self.trotterize) + " directions -> " + ", ".join(
            self.lastate.directions)

    def get_lie_algebra_directions_strings(self):
        return self.lastate.directions


class SquaredLieAlgebraLayer(LieLayer):
    def __init__(self, device, observables: List, **kwargs):
        """
        Class that applies a Riemannian optimization step by calculating all allowed Lie gradients
        and uses Rotosolve to find the optimal stepsze.

        Setting commute = True creates a layer of commuting SU(2) and SU(4) directions that is
        optimized with VQE

        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            device: PennyLane device
            observables: List of single qubit Pauli observables.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
                -
        """
        self.nqubits = len(device.wires)
        super().__init__(device, observables, self.nqubits)

        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.lastate = AlgebraSU4(self.nqubits, add_su2=True)
        self.observables = observables
        self.observables_full = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                                 observables]
        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'
        # initialize
        self.current_pauli, self.previous_pauli = None, ('Null', 0)
        self.lastate.get_all_directions_and_qubits()
        self.dev = qml.device('default.qubit', wires=self.nqubits)

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        commute = kwargs.get('commute', False)

        omegas = {}
        for k, pauli in self.lastate.full_paulis.items():
            if k != self.previous_pauli:
                omegas[k] = 0
                for oi, obs in enumerate(self.observables_full):
                    omegas[k] += float(
                        (phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])
                omegas[k] = abs(omegas[k])
        if commute:
            max_value = max(omegas.values())
            # get the maximum gradients
            max_omegas = [k for k, v in omegas.items() if np.isclose(v, max_value, atol=1e-3)]

            # select arbitrary direction from among them
            direction = max_omegas[0][0]
            max_omega_edge_list = [tuple(om[1:]) for om in max_omegas if om[0] == direction]
            filtered_edges = get_commuting_set(max_omega_edge_list)

            def circuit_lie(params, **kwargs):
                qml.QubitUnitary(circuit_unitary, wires=list(range(self.nqubits)))
                for i, edge in enumerate(filtered_edges):
                    qml.PauliRot(params[i], direction, wires=edge)

            params = [0.1 for _ in filtered_edges]
            parameter_shift_optimizer = kwargs.get('optimizer', None)
            costs, params = parameter_shift_optimizer(circuit_lie, params, self.observables,
                                                      device=self.dev, return_params=True,
                                                      eta=0.05, nsteps=200, tol=1e-5)
            for i, edge in enumerate(filtered_edges):
                U_riemann_approx = ssla.expm(
                    -1j * params[-1][i] * self.lastate.full_paulis[(direction, *edge)] / 2)
                circuit_unitary = U_riemann_approx @ circuit_unitary
            return circuit_unitary, [((direction, *edge), params[-1][i]) for i, edge in
                                     enumerate(filtered_edges)]
        else:
            k_max = max(omegas, key=omegas.get)
            print(k_max)
            self.current_pauli = k_max
            self.previous_pauli = k_max
            circuit_unitary, eta = rotosolve(self.lastate.full_paulis[self.current_pauli],
                                             self.observables, self.obs_qnode, circuit_unitary,
                                             len(k_max[0]))

            return circuit_unitary

    def __repr__(self):
        row_format = "{:^25}|" * 3
        return "| " + \
               row_format.format(f'Stoch. Algebra Layer SU({2 ** self.nqubits})', 'NaN',
                                 self.trotterize) + " directions -> " + ", ".join(
            self.lastate.directions)

    def get_lie_algebra_directions_strings(self):
        return self.lastate.directions


class AdaptVQELayer(LieLayer):
    def __init__(self, device, observables: List, **kwargs):
        """
        Implementation of adapt-VQE.

        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            state_qnode: QNode of a circuit that takes a unitary and returns a state.
            observables: List of single qubit Pauli observables.
            nqubits: The number of qubits in the circuit.
            directions: List of strings containing the allowed directions.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
                -
        """
        self.nqubits = len(device.wires)
        super().__init__(device, observables, self.nqubits)

        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.lastate = AlgebraSU4(self.nqubits, add_su2=True)
        self.observables = observables
        self.observables_full = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                                 observables]
        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'
        # initialize
        self.current_pauli, self.previous_direction = None, ('Null', 0)
        self.lastate.get_all_directions_and_qubits()
        self.dev = qml.device('default.qubit', wires=self.nqubits)
        self.layers = []

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        params_ext = kwargs.get('params')
        ps_optimizer = kwargs.get('optimizer')
        omegas = {}
        layer_omegas = {}
        for k, pauli in self.lastate.full_paulis.items():
            omegas[k] = 0
            for oi, obs in enumerate(self.observables_full):
                omegas[k] += float((phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])
            omegas[k] = abs(omegas[k])

        for direction in self.lastate.directions:
            if direction != self.previous_direction:
                if len(direction) == 1:
                    layer_omegas[(direction, 'single')] = sum(
                        [omegas[(direction, i)] for i in range(self.nqubits)])
                else:
                    layer_omegas[(direction, 'even')] = sum(
                        [omegas[(direction, i, i + 1)] for i in range(0, self.nqubits - 1, 2)])
                    layer_omegas[(direction, 'odd')] = sum(
                        [omegas[(direction, i, i + 1)] for i in range(1, self.nqubits - 1, 2)]) + \
                                                       omegas[(direction, 0, self.nqubits - 1)]
        k_max = max(layer_omegas, key=layer_omegas.get)
        print(k_max)
        self.layers.append(k_max)
        self.previous_direction = k_max[0]
        single_layer = {'X': qml.RX, 'Y': qml.RY, 'Z': qml.RZ}

        def circuit_vqe(params, **kwargs):
            for j in range(self.nqubits):
                qml.Hadamard(wires=j)
            for j, layer in enumerate(self.layers):
                if len(layer[0]) == 1:
                    for i in range(0, self.nqubits):
                        single_layer[layer[0]](params[j][i], wires=i)

                else:
                    if layer[1] == 'even':
                        for idx, i in enumerate(range(0, self.nqubits - 1, 2)):
                            qml.PauliRot(params[j][idx], layer[0], wires=(i, i + 1))
                    else:
                        for idx, i in enumerate(range(1, self.nqubits - 1, 2)):
                            qml.PauliRot(params[j][idx], layer[0], wires=(i, i + 1))
                        qml.PauliRot(params[j][-1], layer[0], wires=(0, self.nqubits - 1))

        params_ext.append([0.1 for _ in range(self.nqubits)])

        costs, params_ext = ps_optimizer(circuit_vqe, params_ext, self.observables, self.dev,
                                         return_params=True, eta=0.01, tol=1e-5)
        circuit_unitary = np.eye(2 ** self.nqubits, 2 ** self.nqubits, dtype=complex)

        def circuit_vqe_state(params, **kwargs):
            circuit_vqe(params, **kwargs)
            return qml.state()

        circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit_vqe_state,
                                                                          params_ext[-1].tolist(),
                                                                          self.dev)

        for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
            circuit_unitary = get_full_operator(op, wires, self.nqubits) @ circuit_unitary

        return circuit_unitary, params_ext[-1].tolist()

    def __repr__(self):
        row_format = "{:^25}|" * 3
        return "| " + \
               row_format.format(f'Adapt VQE Layer', 'NaN',
                                 self.trotterize) + " directions -> " + ", ".join(
            self.lastate.directions)

    def get_lie_algebra_directions_strings(self):
        return self.lastate.directions


class SU8_AlgebraLayer(LieLayer):
    def __init__(self, device, observables: List, **kwargs):
        """
        Class that applies a Riemannian optimization step on a SU(2) or SU(4) and SU(8) local manifold.

        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            state_qnode: QNode of a circuit that takes a unitary and returns a state.
            observables: List of single qubit Pauli observables.
            nqubits: The number of qubits in the circuit.
            directions: List of strings containing the allowed directions.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
                -
        """
        self.nqubits = len(device.wires)
        super().__init__(device, observables, self.nqubits)

        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        assert self.nqubits >= 3, '`nqubits` must be larger than 3'
        self.lastate = AlgebraSU8(self.nqubits,
                                  add_su2=kwargs.get('add_su2', True),
                                  add_su4=kwargs.get('add_su4', True))
        self.observables = observables
        self.observables_full = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                                 observables]
        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'
        # initialize
        self.current_pauli, self.previous_pauli = None, ('Null', 0)
        self.lastate.get_all_directions_and_qubits()
        self.dev = qml.device('default.qubit', wires=self.nqubits)
        self.layers = []

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        omegas = {}
        for k, pauli in self.lastate.full_paulis.items():
            omegas[k] = 0
            for oi, obs in enumerate(self.observables_full):
                omegas[k] += float((phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])
            omegas[k] = abs(omegas[k])

        k_max = max(omegas, key=omegas.get)

        # select arbitrary direction from among them
        self.current_pauli = k_max
        self.previous_pauli = k_max
        circuit_unitary, eta = rotosolve(self.lastate.full_paulis[self.current_pauli],
                                         self.observables, self.obs_qnode, circuit_unitary,
                                         len(k_max[0]))

        return circuit_unitary

    def __repr__(self):
        row_format = "{:^25}|" * 3
        return "| " + \
               row_format.format(f'SU(8) Layer SU({2 ** self.nqubits})', 'NaN',
                                 self.trotterize) + " directions -> " + ", ".join(
            self.lastate.directions)

    def get_lie_algebra_directions_strings(self):
        return self.lastate.directions


class ZassenhausLayer(LieLayer):
    def __init__(self, device, observables: List, **kwargs):
        """
        Class that applies a Riemannian optimization step on a SU(2) or SU(4) and SU(8) local manifold.

        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            state_qnode: QNode of a circuit that takes a unitary and returns a state.
            observables: List of single qubit Pauli observables.
            nqubits: The number of qubits in the circuit.
            directions: List of strings containing the allowed directions.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
                -
        """
        self.nqubits = len(device.wires)
        super().__init__(device, observables, self.nqubits)

        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        assert self.nqubits >= 3, '`nqubits` must be larger than 3'
        self.lastates = [AlgebraSU2(self.nqubits),
                         AlgebraSU4(self.nqubits,
                                    add_su2=kwargs.get('add_su2', False)),
                         AlgebraSU8(self.nqubits,
                                    add_su2=kwargs.get('add_su2', False),
                                    add_su4=kwargs.get('add_su4', False))]
        self.observables = observables
        self.observables_full = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                                 observables]
        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'
        # initialize
        self.current_pauli, self.previous_pauli = None, ('Null', 0)
        for lastate in self.lastates:
            lastate.get_all_directions_and_qubits()
        self.dev = qml.device('default.qubit', wires=self.nqubits)
        self.ratio = kwargs.get('ratio', (1, 1))
        # self.layers = []
        self.counter = 0
        self.su_state = 'su4'
        # self.lastate_idx = 0
        # self.previous_eta = 0
        # self.palgebra = {('X', 'X'): (1, 'I'), ('X', 'Y'): (1, 'Z'), ('X', 'Z'): (-1, 'Y'),
        #                  ('Y', 'X'): (-1, 'Z'), ('Y', 'Y'): (1, 'I'), ('Y', 'Z'): (1, 'X'),
        #                  ('Z', 'X'): (1, 'Y'), ('Z', 'Y'): (-1, 'X'), ('Z', 'Z'): (1, 'I'), }

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]

        if self.su_state == 'su4':
            print('state: ', self.su_state)
            omegas = {}
            for k, pauli in self.lastates[1].full_paulis.items():
                omegas[k] = 0
                for oi, obs in enumerate(self.observables_full):
                    omegas[k] += float(
                        (phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])
                omegas[k] = abs(omegas[k])

            k_max = max(omegas, key=omegas.get)
            # omegas_sorted = sorted(omegas, key=omegas.get)
            # probs = np.array([omegas[om] for om in omegas_sorted])
            # probs /= np.sum(probs)
            # k_max = omegas_sorted[np.random.choice(list(range(len(omegas_sorted))), p=probs)]

            circuit_unitary, eta = rotosolve(
                self.lastates[1].full_paulis[k_max],
                self.observables, self.obs_qnode, circuit_unitary,
                len(k_max[0]))
            self.current_pauli = k_max
            self.previous_eta = eta

            self.counter += 1
            if self.counter >= self.ratio[0]:
                self.counter = 0
                self.su_state = 'su8'
        else:
            print('state: ', self.su_state)
            omegas_su8 = {}
            omegas_su4 = {}
            omegas_su8_k = {}
            omegas_su8_signs = {}
            pauli = self.lastates[1].full_paulis[self.current_pauli]
            omegas_A = self.previous_eta
            # for oi, obs in enumerate(self.observables_full):
            #     omegas_A += float((phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])

            for k, pauli in self.lastates[1].full_paulis.items():
                if sum(ki in self.current_pauli[1:] for ki in k[1:]) == 1:
                    p_a = PauliMonomial([Pauli(loc, pauli) for pauli, loc in
                                         zip(self.current_pauli[0], self.current_pauli[1:])])
                    p_b = PauliMonomial([Pauli(loc, pauli) for pauli, loc in zip(k[0], k[1:])])
                    p_ab = p_a.commutator(p_b)
                    AB = (
                        ''.join(pauli_idx_int2str_map[p.gamma] for p in p_ab),
                        *[p.alpha for p in p_ab])
                else:
                    continue
                if 'I' in AB[0]:
                    continue
                omegas_su8[AB] = 0
                pauli = self.lastates[2].full_paulis[AB]
                for oi, obs in enumerate(self.observables_full):
                    omegas_su8[AB] += float(
                        (phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])
                omegas_su8[AB] = abs(omegas_su8[AB])
                omegas_su8_k[AB] = copy.copy(k)
                omegas_su8_signs[AB] = p_ab.coeff.imag

                omegas_su4[k] = 0
                pauli = self.lastates[1].full_paulis[k]
                for oi, obs in enumerate(self.observables_full):
                    omegas_su4[k] += float(
                        (phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])

            print(self.current_pauli)
            print(omegas_su8)
            print(omegas_su4)
            for k,v in omegas_su8.items():
                if abs(v)>1e-3:
                    print(k,v)
                    if abs(omegas_su4[omegas_su8_k[k]])>0:
                        print("Gate A:",self.current_pauli, omegas_A)
                        print("SU(8) candidate found:",k, omegas_su8[k])
                        print("Gate B SU(4) candidate:",omegas_su8_k[k], omegas_su4[omegas_su8_k[k]])
                        B = omegas_su8_k[k]
                        k_max = k
                        break
            # k_max = max(omegas_su8, key=omegas_su8.get)
            # omegas_sorted = sorted(omegas_su8, key=omegas_su8.get)
            # probs = np.array([omegas_su8[om] for om in omegas_sorted])
            # probs /= np.sum(probs)
            # k_max = omegas_sorted[np.random.choice(list(range(len(omegas_sorted))), p=probs)]
            omegas_AB = 0
            # B = omegas_su8_k[k_max]
            pauli = self.lastates[2].full_paulis[k_max]
            for oi, obs in enumerate(self.observables_full):
                omegas_AB += float((phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])
            omegas_B = -((omegas_AB * 2) / omegas_A)  / omegas_su8_signs[k_max]
            if np.sign(omegas_B)==np.sign(omegas_su4[omegas_su8_k[k]]):
                print("SAME SIGN")
            else:
                print('DIFFERENT SIGN')
            U_riemann_approx = ssla.expm(
                # -1j * self.eta * omegas_su4[B] * self.lastates[1].full_paulis[B])
                -1j * self.eta * omegas_B / 2**self.nqubits * self.lastates[1].full_paulis[B])
            circuit_unitary = U_riemann_approx @ circuit_unitary
            print(omegas_A, omegas_B, omegas_AB)
            # omegas_B = 0
            # pauli = self.lastates[1].full_paulis[B]
            # for oi, obs in enumerate(self.observables_full):
            #     omegas_B += float((phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0, 0])
            # print(omegas_B)
            # if abs(omegas_B)>1e-3:
            #     omegas_B = ((omegas_AB * 2) / omegas_A)%(2*np.pi) * omegas_su8_signs[k_max]
            #     print(omegas_A, omegas_B, omegas_AB)
            #     U_riemann_approx = ssla.expm(-1j * self.eta  * omegas_B * self.lastates[1].full_paulis[B])
            #     circuit_unitary = U_riemann_approx @ circuit_unitary
            #
            # grad = []
            # pauli = self.lastates[2].full_paulis[k_max]
            # for i,t in enumerate(np.linspace(0, 2*np.pi,100)):
            #     U_riemann_approx = ssla.expm(-1j * t * pauli)
            #     grad.append(0)
            #     for oi, obs in enumerate(self.observables_full):\
            #         grad[i] += float(
            #         # (phi.conj().T @ U_riemann_approx.conj().T@(pauli @ obs - obs @ pauli) @ U_riemann_approx@phi).imag[0, 0])
            #         (phi.conj().T @ U_riemann_approx.conj().T@ obs @ U_riemann_approx@phi).imag[0, 0])
            # print(grad)
            # B = omegas_su8_k[k_max]
            # eta = np.linspace(0, 2 * np.pi, 100)[np.argmin(grad)]
            # U_riemann_approx = ssla.expm(-1j * eta * self.lastates[1].full_paulis[B])
            # circuit_unitary = U_riemann_approx @ circuit_unitary
            #
            # circuit_unitary, eta = rotosolve(
            #     self.lastates[1].full_paulis[B],
            #     self.observables, self.obs_qnode, circuit_unitary,
            #     len(k_max[0]))

            self.current_pauli = B
            # self.previous_eta = eta
            self.counter += 1
            if self.counter >= self.ratio[1]:
                self.counter = 0
                self.su_state = 'su4'

        return circuit_unitary

    def __repr__(self):
        row_format = "{:^25}|" * 3
        return "| " + \
               row_format.format(f'Zassenhaus Layer SU({2 ** self.nqubits})', 'NaN',
                                 self.trotterize) + " directions -> " + "SU(2), SU(4), SU(8) by Zassenhaus "

    def get_lie_algebra_directions_strings(self):
        return [lastate.directions for lastate in self.lastates]


class ZassenhausLayerV2(LieLayer):
    def __init__(self, device, observables: List, **kwargs):
        """
        Class that applies a Riemannian optimization step on a SU(2) or SU(4) and SU(8) local manifold.

        Uses the matrix exponential to calculate the exact operator of the commutator.
        Trotterization applies only on the level of observables, NOT on the level of individual SU(p) terms.

        Args:
            state_qnode: QNode of a circuit that takes a unitary and returns a state.
            observables: List of single qubit Pauli observables.
            nqubits: The number of qubits in the circuit.
            directions: List of strings containing the allowed directions.
            **kwargs: Additional keyword arguments are
                - eta: the stepsize
                - unitary_error_check: Boolean that flags whether to check if the resulting unitary is a valid
                unitary operator.
                -
        """
        self.nqubits = len(device.wires)
        super().__init__(device, observables, self.nqubits)

        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        # assert self.nqubits >= 3, '`nqubits` must be larger than 3'
        self.lastates = [AlgebraSU2(self.nqubits),
                         AlgebraSU4(self.nqubits,
                                    add_su2=kwargs.get('add_su2', False)),
                         AlgebraSU8(self.nqubits,
                                    add_su2=kwargs.get('add_su2', False),
                                    add_su4=kwargs.get('add_su4', False))]
        self.observables = observables
        self.observables_full = [get_full_operator(obs.matrix, obs.wires, self.nqubits) for obs in
                                 observables]
        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'
        # initialize
        self.current_pauli, self.previous_pauli = None, ('Null', 0)
        for lastate in self.lastates:
            lastate.get_all_directions_and_qubits()
        self.dev = qml.device('default.qubit', wires=self.nqubits)
        self.ratio = kwargs.get('ratio', (1, 1))
        self.params = np.array([0.1,1.2,0.6])
        self.symbols = sp.symbols('x0,x1,z12')
        print(self.symbols)


        circuit = [PauliMonomial([Pauli(0, 'X', coeff=-1j * self.symbols[0]), Pauli(1, 'I')]),
                   PauliMonomial([Pauli(0, 'I'), Pauli(1, 'X', coeff=-1j * self.symbols[1])]),
                   PauliMonomial([Pauli(0, 'Z', coeff=-1j * self.symbols[2]), Pauli(1, 'Z')])]
        bch_terms = dict(zip(range(len(circuit)), [set() for _ in range(len(circuit))]))
        for i in range(len(circuit) - 1):
            bch_terms[i] |= set([circuit[i]])
            bch_terms[i + 1] |= set([circuit[i + 1]])
            new_bch_terms = set()
            for term_i in bch_terms[i]:
                new_bch_terms.add(term_i)
                for term_i_plus_one in bch_terms[i + 1]:
                    comm = term_i.commutator(term_i_plus_one)
                    comm.coeff *= 0.5
                    if comm.coeff == 0:
                        continue
                    elif not np.isclose(float(comm.coeff.args[0]), 0.):
                        new_bch_terms.add(comm)
            bch_terms[i + 1] |= new_bch_terms
        self.lie_algebra_vector = []
        self.lie_algebra_polynomial = []
        for t in bch_terms[2]:
            print(t.to_oranssi(), t.coeff)
            self.lie_algebra_vector.append(t.to_oranssi())
            self.lie_algebra_polynomial.append(t.coeff)
        print(self.lie_algebra_vector)
    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        omegas = []
        for i, p in enumerate(self.lie_algebra_vector):
            omegas.append(0)
            if len(p[0])==1:
                pauli = self.lastates[0].full_paulis[p]
            elif len(p[0]) == 2:
                pauli = self.lastates[1].full_paulis[p]
            else:
                pauli = self.lastates[2].full_paulis[p]
            for oi, obs in enumerate(self.observables_full):
                omegas[i] += float((phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi).imag[0,0])
        print(omegas)
        lie_algebra_polynomial = [self.lie_algebra_polynomial[i] - self.eta * 1j*omegas[i] for i in
                                  range(len(self.lie_algebra_vector))]
        # print(lie_algebra_polynomial)
        lie_algebra_polynomial_python = [lambdify(self.symbols, f) for f in lie_algebra_polynomial]

        def cost(x):
            return np.mean([np.square(f(*x).imag) for f in lie_algebra_polynomial_python])

        result = minimize(cost, x0=[0.1,0.2,0.3], method='Nelder-Mead')
        dev = qml.device('default.qubit', wires=2)
        self.params += np.copy(result.x)
        print(self.params)
        def circuit(params, **kwargs):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(params[2], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, result.x, dev)
        circuit_unitary = np.eye(2 ** self.nqubits, 2 ** self.nqubits, dtype=complex)
        for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
            circuit_unitary = get_full_operator(op, wires, self.nqubits) @ circuit_unitary

        return circuit_unitary

    def __repr__(self):
        row_format = "{:^25}|" * 3
        return "| " + \
               row_format.format(f'Zassenhaus Layer SU({2 ** self.nqubits})', 'NaN',
                                 self.trotterize) + " directions -> " + "SU(2), SU(4), SU(8) by Zassenhaus "

    def get_lie_algebra_directions_strings(self):
        return [lastate.directions for lastate in self.lastates]
class AlgebraSU2():

    def __init__(self, nqubits, add_su2: bool = False):
        """
        Abstract SU(4) algebra class.

        Args:
            nqubits: Integer number of qubits.
            add_su2: Booolean that indicates if we add SU(2)
        """
        self.nqubits = nqubits
        self.add_su2 = add_su2
        self.paulis, self.directions = get_su_2_operators(return_names=True)

        self.reset_directions()

        self.paulis = dict(zip(self.directions, self.paulis))

        self.directions_left = copy.copy(self.directions)
        self.qubits_left = np.zeros(self.nqubits, dtype=int)
        self.full_paulis = {}

        self.FIRST_CALL = True

    def get_random_direction_and_qubits(self):
        """
        Get a random SU(4) direction.

        Returns:

        """
        raise NotImplementedError
        idx = np.where(self.qubits_left == 0)[0].tolist()
        # allow SU(4) samples
        if not idx:
            self.reset_qubits()
            idx = np.where(self.qubits_left == 0)[0].tolist()
        if not self.directions_left:
            self.reset_directions()
        if len(idx) > 1:
            direction = np.random.choice(self.directions_left, size=1)[0]
            if len(direction) == 1:
                qubits = np.random.permutation(idx)[:1]
            else:
                qubits = np.random.permutation(idx)[:2]
        else:
            if any(len(d) == 1 for d in self.directions_left):
                direction = np.random.choice([d for d in self.directions_left if len(d) == 1],
                                             size=1)[0]
            else:
                self.reset_directions(su2_only=True)
                direction = np.random.choice([d for d in self.directions_left if len(d) == 1],
                                             size=1)[0]

            qubits = np.random.permutation(idx)[:1]
        self.qubits_left[qubits] = 1
        self.add_pauli(direction, qubits)
        self.directions_left.remove(direction)
        return direction, qubits

    def get_all_directions_and_qubits(self):
        """
        Construct all

        Returns:

        """
        idx = np.where(self.qubits_left == 0)[0].tolist()
        # allow SU(4) samples
        if not idx:
            self.reset_qubits()
            idx = np.where(self.qubits_left == 0)[0].tolist()

        if len(idx) > 1:
            qubits = [(q,) for q in idx] + list(it.combinations(idx, r=2))
        else:
            qubits = [(q,) for q in idx]
        for direction in self.directions:
            for q in qubits:
                if len(q) == len(direction):
                    self.add_pauli(direction, q)

    def add_pauli(self, direction, qubits):
        if (direction, *qubits) not in self.full_paulis.keys():
            self.full_paulis[(direction, *qubits)] = get_full_operator(self.paulis[direction],
                                                                       qubits, self.nqubits)

    def reset_directions(self):
        self.directions_left = copy.copy(self.directions)

    def reset_qubits(self):
        self.qubits_left = np.zeros(self.nqubits, dtype=int)


class AlgebraSU4():

    def __init__(self, nqubits, add_su2: bool = False):
        """
        Abstract SU(4) algebra class.

        Args:
            nqubits: Integer number of qubits.
            add_su2: Booolean that indicates if we add SU(2)
        """
        self.nqubits = nqubits
        self.add_su2 = add_su2
        if add_su2:
            self.paulis = get_su_2_operators(return_names=False) + \
                          get_su_4_operators(return_names=False)
            self.directions = get_su_2_operators(return_names=True)[1] + \
                              get_su_4_operators(return_names=True)[1]
        else:
            self.paulis = get_su_4_operators(return_names=False)
            self.directions = get_su_4_operators(return_names=True)[1]

        self.reset_directions()

        self.paulis = dict(zip(self.directions, self.paulis))

        self.directions_left = copy.copy(self.directions)
        self.qubits_left = np.zeros(self.nqubits, dtype=int)
        self.full_paulis = {}

        self.FIRST_CALL = True

    def get_random_direction_and_qubits(self):
        """
        Get a random SU(4) direction.

        Returns:

        """
        raise NotImplementedError
        idx = np.where(self.qubits_left == 0)[0].tolist()
        # allow SU(4) samples
        if not idx:
            self.reset_qubits()
            idx = np.where(self.qubits_left == 0)[0].tolist()
        if not self.directions_left:
            self.reset_directions()
        if len(idx) > 1:
            direction = np.random.choice(self.directions_left, size=1)[0]
            if len(direction) == 1:
                qubits = np.random.permutation(idx)[:1]
            else:
                qubits = np.random.permutation(idx)[:2]
        else:
            if any(len(d) == 1 for d in self.directions_left):
                direction = np.random.choice([d for d in self.directions_left if len(d) == 1],
                                             size=1)[0]
            else:
                self.reset_directions(su2_only=True)
                direction = np.random.choice([d for d in self.directions_left if len(d) == 1],
                                             size=1)[0]

            qubits = np.random.permutation(idx)[:1]
        self.qubits_left[qubits] = 1
        self.add_pauli(direction, qubits)
        self.directions_left.remove(direction)
        return direction, qubits

    def get_all_directions_and_qubits(self):
        """
        Construct all

        Returns:

        """
        idx = np.where(self.qubits_left == 0)[0].tolist()
        # allow SU(4) samples
        if not idx:
            self.reset_qubits()
            idx = np.where(self.qubits_left == 0)[0].tolist()

        if len(idx) > 1:
            qubits = [(q,) for q in idx] + list(it.combinations(idx, r=2))
        else:
            qubits = [(q,) for q in idx]
        for direction in self.directions:
            for q in qubits:
                if len(q) == len(direction):
                    self.add_pauli(direction, q)

    def add_pauli(self, direction, qubits):
        if (direction, *qubits) not in self.full_paulis.keys():
            self.full_paulis[(direction, *qubits)] = get_full_operator(self.paulis[direction],
                                                                       qubits, self.nqubits)

    def reset_directions(self, su2_only=False):
        if su2_only:
            self.directions_left = self.directions_left + get_su_2_operators(return_names=True)[1]
        else:
            self.directions_left = copy.copy(self.directions)

    def reset_qubits(self):
        self.qubits_left = np.zeros(self.nqubits, dtype=int)


class AlgebraSU8():
    def __init__(self, nqubits, add_su2: bool = False, add_su4: bool = False):
        """
        Abstract SU(8) algebra class.

        Args:
            nqubits: Integer number of qubits.
            add_su2: Booolean that indicates if we add SU(2)
            add_su4: Booolean that indicates if we add SU(2)
        """
        self.nqubits = nqubits
        self.add_su2 = add_su2
        self.add_su4 = add_su4
        to_add_bool = [add_su2, add_su4, True]
        to_add_paulis = [lambda: get_su_2_operators(return_names=True),
                         lambda: get_su_4_operators(return_names=True),
                         lambda: get_su_n_operators(8, return_names=True)]
        p_d_list = [to_add_paulis[i]() for i, check in enumerate(to_add_bool) if check]

        self.paulis = [item for sublist in p_d_list for item in sublist[0]]
        self.directions = [item for sublist in p_d_list for item in sublist[1]]
        self.qubits = {3: list(it.combinations(range(self.nqubits), r=3))}
        if add_su2:
            self.qubits[1] = list((i,) for i in range(self.nqubits))
        if add_su4:
            self.qubits[2] = list(it.combinations(range(self.nqubits), r=2))

        self.paulis = dict(zip(self.directions, self.paulis))

        self.full_paulis = {}

    def get_random_direction_and_qubits(self):
        raise NotImplementedError

    def get_all_directions_and_qubits(self):

        for direction in self.directions:
            for q in self.qubits[len(direction)]:
                self.add_pauli(direction, q)

    def add_pauli(self, direction, qubits):
        if (direction, *qubits) not in self.full_paulis.keys():
            self.full_paulis[(direction, *qubits)] = get_full_operator(self.paulis[direction],
                                                                       qubits, self.nqubits)


def rotosolve(pauli, observables, obs_qnode, circuit_unitary, nqubits):
    """
    Implementation of Rotosolve algorithm to find the optimal stepsize.

    Args:
        pauli: The full pauli that we wwant to append to the circuit
        observables: List of PennyLane observables
        obs_qnode: observable Qnode
        circuit_unitary: Matrix corresponding to the circuit unitary.
        nqubits: number of qubits.

    Returns:
        Circuit unitary with optimal angle.
    """
    H_0 = 0.
    for o in observables:
        H_0 += obs_qnode(
            unitary=circuit_unitary,
            observable=o)
    H_p = 0.
    U_riemann_approx = ssla.expm(-1j * np.pi / 2 * pauli / 2 ** nqubits)
    for o in observables:
        H_p += obs_qnode(
            unitary=U_riemann_approx @ circuit_unitary,
            observable=o)
    H_m = 0.
    U_riemann_approx = ssla.expm(1j * np.pi / 2 * pauli / 2 ** nqubits)
    for o in observables:
        H_m += obs_qnode(
            unitary=U_riemann_approx @ circuit_unitary,
            observable=o)
    eta = -np.pi / 2 - np.arctan2(2 * H_0 - H_p - H_m, H_p - H_m)
    U_riemann_approx = ssla.expm(-1j * eta * pauli / 2 ** nqubits)
    return U_riemann_approx @ circuit_unitary, eta
