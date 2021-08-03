import copy

import numpy as np
import scipy.linalg as ssla

from typing import List
import itertools as it

from oranssi.circuit_tools import get_full_operator
from oranssi.utils import get_su_2_operators, get_su_4_operators


class LieLayer(object):
    def __init__(self, state_qnode, observables: List, nqubits: int):
        """
        Class that applies a Riemannian optimization step on a submanifold of the Lie group.

        Args:
            state_qnode: QNode of a circuit that takes a unitary and returns a state.
            observables: List of single qubit Pauli observables.
            locality: Either 1 or 2, indicating SU(2) or SU(4) local.
            nqubits: The number of qubits in the circuit.
                -
        """
        self.state_qnode = state_qnode
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
        raise NotImplementedError

    def get_lie_algebra_directions(self, circuit_unitary):

        raise NotImplementedError

    def get_lie_algebra_directions_strings(self):
        raise NotImplementedError

    def set_eta(self, eta):
        self.eta = np.copy(eta)


class LocalLieAlgebraLayer(LieLayer):
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
        super().__init__(state_qnode, observables, nqubits)
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
        # print(np.linalg.eigvalsh(np.sum(self.observables, axis=0)))

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
            # if self.stride == 0:
            #     for i in range(0, nqubits, 2):
            #         self.full_paulis.append(
            #             [get_full_operator(p, (i, i + 1), self.nqubits) for p in self.paulis])
            # else:
            #     for i in range(self.stride, nqubits - 1, 2):
            #         self.full_paulis.append(
            #             [get_full_operator(p, (i, i + 1), self.nqubits) for p in self.paulis])
            #     self.full_paulis.append(
            #         [get_full_operator(p, (nqubits - 1, 0), self.nqubits) for p in self.paulis])
            for comb in it.combinations(range(nqubits), r=2):
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

    def get_lie_algebra_directions(self, circuit_unitary):

        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        omegas = np.zeros((len(self.observables), len(self.full_paulis), len(self.full_paulis[0])))
        for o, obs in enumerate(self.observables):
            for p, full_paulis in enumerate(self.full_paulis):
                self.op.fill(0)
                for j, pauli in enumerate(full_paulis):
                    omegas[o, p, j] = (phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi)[0, 0].imag
        return omegas


class LieAlgebraLayer(LieLayer):
    def __init__(self, state_qnode, observables: List, directions: List[str], nqubits: int,
                 **kwargs):
        """
        Class that applies a Riemannian optimization step on pre-specified Lie algebra.
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
        super().__init__(state_qnode, observables, nqubits)

        self.state_qnode = state_qnode
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'

        self.paulis = get_su_2_operators(return_names=False) + get_su_4_operators(
            return_names=False)
        self.directions = get_su_2_operators(return_names=True)[1] + \
                          get_su_4_operators(return_names=True)[1]
        print(directions)
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
        # print(np.linalg.eigvalsh(np.sum(self.observables, axis=0)))

        self.unitary_error_check = kwargs.get('unitary_error_check', False)
        assert isinstance(self.unitary_error_check,
                          bool), f'`unitary_error_check` must be a boolean, ' \
                                 f'received {type(self.unitary_error_check)}'
        self.trotterize = kwargs.get('trotterize', False)
        assert isinstance(self.trotterize, bool), f'`trotterize` must be a boolean, ' \
                                                  f'received {type(self.trotterize)}'
        # depending on the locality, create the full pauli matrices required to calculate the commutators
        self.full_paulis = []
        for d, pauli in zip(self.directions, self.paulis):
            if len(d) == 1:
                for i in range(self.nqubits):
                    self.full_paulis.append(
                        [get_full_operator(p, (i,), self.nqubits) for p in self.paulis if
                         p.shape[0] == 2])
            elif len(d) == 2:
                for i in range(0, nqubits, 2):
                    self.full_paulis.append(
                        [get_full_operator(p, (i, i + 1), self.nqubits) for p in self.paulis if
                         p.shape[0] == 4])
                for i in range(1, nqubits - 1, 2):
                    self.full_paulis.append(
                        [get_full_operator(p, (i, i + 1), self.nqubits) for p in self.paulis if
                         p.shape[0] == 4])
                self.full_paulis.append(
                    [get_full_operator(p, (nqubits - 1, 0), self.nqubits) for p in self.paulis if
                     p.shape[0] == 4])
        self.op = np.zeros((2 ** self.nqubits, 2 ** self.nqubits), dtype=complex)

    def __call__(self, circuit_unitary, *args, **kwargs):
        phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
        for oi, obs in enumerate(self.observables):
            for full_paulis in self.full_paulis:
                self.op.fill(0)
                omegas = []
                if self.trotterize:
                    for j, pauli in enumerate(full_paulis):
                        # phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
                        omega = phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi
                        self.op = omega * pauli
                        U_riemann_approx = ssla.expm(- self.eta / 2 ** self.nqubits * self.op)
                        if (self.unitary_error_check) and (self._is_unitary(U_riemann_approx)):
                            U_riemann_approx = self._project_onto_unitary(U_riemann_approx)
                        circuit_unitary = U_riemann_approx @ circuit_unitary
                        self.op.fill(0)
                else:
                    # phi = self.state_qnode(unitary=circuit_unitary)[:, np.newaxis]
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
               row_format.format(f'Lie Algebra Layer SU({2 ** self.nqubits})', 'NaN',
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


class StochasticLieAlgebraLayer(LieLayer):
    def __init__(self, state_qnode, observables: List, nqubits: int,
                 **kwargs):
        """
        Class that applies a Riemannian optimization step on pre-specified Lie algebra.
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
        super().__init__(state_qnode, observables, nqubits)

        self.state_qnode = state_qnode
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.nqubits = nqubits
        self.lastate = AlgebraSU4(nqubits, add_su2=True)

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
        new_direction = kwargs.get('new_direction', False)
        perturb = kwargs.get('perturb', False)

        if new_direction:
            direction, qubits = self.lastate.get_direction_and_qubits()
            self.current_direction = direction
            self.current_qubits = qubits

        for oi, obs in enumerate(self.observables):
            pauli = self.lastate.full_paulis[(self.current_direction, *self.current_qubits)]
            omega = phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi
            if perturb:
                omega = 1j * np.sign(omega.real)

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

    def get_lie_algebra_directions(self, circuit_unitary):

        raise NotImplementedError

    def get_lie_algebra_directions_strings(self):
        return self.lastate.directions

    # def reset_directions(self):
    #     # self.directions = get_su_2_operators(return_names=True)[1] +\
    #     #                   get_su_4_operators(return_names=True)[1]
    #     self.directions = get_su_4_operators(return_names=True)[1]
    #     self.directions_left = copy.copy(self.directions)
    #     print('reset directions')
    #
    # def reset_qubits(self):
    #     # self.combinations_1 = list((q,) for q in range(nqubits))
    #     # self.combinations_2 = list(it.combinations(range(self.nqubits), r=2))
    #     self.qubits_left = np.zeros(self.nqubits, dtype=int)
    #     print('reset qubits')


class SquaredLieAlgebraLayer(LieLayer):
    def __init__(self, state_qnode, observables: List, nqubits: int,
                 **kwargs):
        """
        Class that applies a Riemannian optimization step on pre-specified Lie algebra.
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
        super().__init__(state_qnode, observables, nqubits)

        self.state_qnode = state_qnode
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.eta = kwargs.get('eta', 0.1)
        assert (isinstance(self.eta, float) & (0. <= self.eta <= 1.)), \
            f'`eta` must be an float between 0 and 1, received {self.eta}'
        self.nqubits = nqubits
        self.lastate = AlgebraSU4(nqubits, add_su2=True)

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
        new_direction = kwargs.get('new_direction', False)
        perturb = kwargs.get('perturb', False)

        if new_direction:
            direction, qubits = self.lastate.get_direction_and_qubits()
            self.current_direction = direction
            self.current_qubits = qubits

        for oi, obs in enumerate(self.observables):
            pauli = self.lastate.full_paulis[(self.current_direction, *self.current_qubits)]
            omega = phi.conj().T @ (pauli @ obs - obs @ pauli) @ phi
            if perturb:
                omega = 1j * np.sign(omega.real)

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

    def get_lie_algebra_directions(self, circuit_unitary):

        raise NotImplementedError

    def get_lie_algebra_directions_strings(self):
        return self.lastate.directions

    # def reset_directions(self):
    #     # self.directions = get_su_2_operators(return_names=True)[1] +\
    #     #                   get_su_4_operators(return_names=True)[1]
    #     self.directions = get_su_4_operators(return_names=True)[1]
    #     self.directions_left = copy.copy(self.directions)
    #     print('reset directions')
    #
    # def reset_qubits(self):
    #     # self.combinations_1 = list((q,) for q in range(nqubits))
    #     # self.combinations_2 = list(it.combinations(range(self.nqubits), r=2))
    #     self.qubits_left = np.zeros(self.nqubits, dtype=int)
    #     print('reset qubits')


class AlgebraSU4():
    def __init__(self, nqubits, add_su2: bool = False):
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

    def get_direction_and_qubits(self):
        print(self.directions_left, ' - ', self.qubits_left)
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

    def add_pauli(self, direction, qubits):
        self.full_paulis[(direction, *qubits)] = get_full_operator(self.paulis[direction],
                                                                   qubits, self.nqubits)

    def reset_directions(self, su2_only=False):
        if su2_only:
            self.directions_left = self.directions_left + get_su_2_operators(return_names=True)[1]
        else:
            self.directions_left = copy.copy(self.directions)

    def reset_qubits(self):
        self.qubits_left = np.zeros(self.nqubits, dtype=int)