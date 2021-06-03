import numpy as np
from typing import List
import itertools as it


def get_su_2_operators(identity: bool = False) -> List[np.ndarray]:
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
        return [I, X, Y, Z]
    else:
        return [X, Y, Z]


def get_su_4_operators(identity: bool = False) -> List[np.ndarray]:
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
    Z = np.array([[1., 0], [0j, -1.]], complex)
    if identity:
        paulis = [I, X, Y, Z]
    else:
        paulis = [X, Y, Z]
    operators = []
    for comb in it.product(list(range(4)), repeat=2):
        operators.append(np.kron(paulis[comb[0]], paulis[comb[1]]))
    return operators


def operator_2_norm(R: np.ndarray) -> float:
    """
    Calculate the operator two norm sqrt{tr{R^dag R}

    Args:
        R:

    Returns:
        Scalar corresponding to the norm
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R))
