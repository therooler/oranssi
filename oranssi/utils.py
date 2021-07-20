import numpy as np
import itertools as it
import os


pauli_int_to_str_id = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
pauli_int_to_str = {0: 'X', 1: 'Y', 2: 'Z'}


def get_su_2_operators(identity: bool = False, return_names: bool = False):
    """
    Get the 2x2 SU(2) operators. The dimension of the group is n^2-1, so we have 3 operators.

    Args:
        identity: Boolean that flags whether we add the identity to the operators or not

    Returns:
        List of 2x2 numpy complex arrays
    """
    I = np.eye(2, 2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1., 0], [0j, -1.]], complex)
    if identity:
        paulis=  [I, X, Y, Z]
        if return_names:
            return paulis, ['I', 'X', 'Y', 'Z']
        else:
            return paulis
    else:
        paulis = [X, Y, Z]
        if return_names:
            return paulis, ['X', 'Y', 'Z']
        else:
            return paulis


def get_su_4_operators(identity: bool = False, return_names: bool = False):
    """
    Get the 4x4 SU(2) operators. The dimension of the group is n^2-1, so we have 15 operators.

    Args:
        identity: Boolean that flags whether we add the identity to the operators or not

    Returns:
        List of 2x2 numpy complex arrays
    """
    I = np.eye(2, 2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1., 0], [0, -1.]], complex)
    if identity:
        paulis = [I, X, Y, Z]
    else:
        paulis = [X, Y, Z]
    operators = []
    names = []
    for comb in it.product(list(range(len(paulis))), repeat=2):
        operators.append(np.kron(paulis[comb[0]], paulis[comb[1]]))
        names.append(comb)

    if return_names:
        if identity:
            return operators, [''.join([pauli_int_to_str_id[i] for i in n]) for n in names]
        else:
            return operators, [''.join([pauli_int_to_str[i] for i in n])  for n in names]
    else:
        return operators


def operator_2_norm(R: np.ndarray) -> float:
    """
    Calculate the operator two norm sqrt{tr{R^dag R}

    Args:
        R:

    Returns:
        Scalar corresponding to the norm
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R)).real


def save_path_creator(path, experiment_name):
    save_path = os.path.join(path, experiment_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path
