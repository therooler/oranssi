import numpy as np
import pytest
from oranssi.utils import get_su_2_operators, get_su_4_operators, get_su_n_operators
import itertools as it


@pytest.mark.parametrize('identity', [True, False])
def test_su_2_operators(identity):
    I = np.eye(2, 2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1., 0], [0j, -1.]], complex)
    paulis_true = [I,X,Y,Z]
    paulis = get_su_2_operators(identity=identity)
    for p in paulis:
        assert any([np.allclose(p, pt) for pt in paulis_true])

@pytest.mark.parametrize('identity', [True, False])
def test_su_4_operators(identity):
    paulis = get_su_2_operators(identity)
    paulis_true = []
    for comb in it.product(list(range(len(paulis))), repeat=2):
        paulis_true.append(np.kron(paulis[comb[0]], paulis[comb[1]]))
    paulis_su_4 = get_su_4_operators(identity=identity)

    for p in paulis_su_4:
        assert any([np.allclose(p, pt) for pt in paulis_true])

@pytest.mark.parametrize('identity, N', [[True, 2], [False, 2], [True, 4], [False, 4]])
def test_su_n_operators(identity, N):
    if N==2:
        paulis_true = get_su_2_operators(identity)
    else:
        paulis_true = get_su_4_operators(identity)
    for p in get_su_n_operators(N, identity=identity):
        assert any([np.allclose(p, pt) for pt in paulis_true])

@pytest.mark.parametrize('N', [2,4,8,16])
def test_su_n_operators_dim(N):
    paulis =  get_su_n_operators(N, identity=True)
    assert len(paulis) == (N**2)