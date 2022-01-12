import numpy as np
from oranssi.pauli import Pauli, PauliMonomial
import sympy as sp
from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify
import pennylane as qml
from oranssi.circuit_tools import get_ops_from_qnode, get_full_operator
import scipy.sparse.linalg as ssla


def closeness_1():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)

    def circuit(params, **kwargs):
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[2], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    params = [0.2, 0.4, 0.3]

    circuit_as_numpy_ops, circuit_as_numpy_wires = get_ops_from_qnode(circuit, params, dev)
    circuit_unitary = np.eye(2 ** nqubits, 2 ** nqubits, dtype=complex)
    for op, wires in zip(circuit_as_numpy_ops, circuit_as_numpy_wires):
        circuit_unitary = get_full_operator(op, wires, nqubits) @ circuit_unitary

    np.random.seed(234234)
    symbols = sp.symbols('x0,x1,z12')
    circuit = [PauliMonomial([Pauli(0, 'X', coeff=-1j * symbols[0]), Pauli(1, 'I')]),
               PauliMonomial([Pauli(0, 'I'), Pauli(1, 'X', coeff=-1j * symbols[1])]),
               PauliMonomial([Pauli(0, 'Z', coeff=-1j * symbols[2]), Pauli(1, 'Z')])]

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
    lie_algebra_vector = []
    lie_algebra_polynomial = []
    for t in bch_terms[2]:
        lie_algebra_vector.append(t.to_oranssi())
        lie_algebra_polynomial.append(t.coeff)
    print(lie_algebra_vector)
    print(lie_algebra_polynomial)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ZZ = np.kron(Z, Z)
    ZY = np.kron(Z, Y)
    YZ = np.kron(Y, Z)
    circuit_unitary = lambda x0, x1, z12: ssla.expm(-1j * (z12 * ZZ)) @ \
                                          ssla.expm(-1j * (x1 * np.kron(np.eye(2), X))) @ \
                                          ssla.expm(-1j * (x0 * np.kron(X, np.eye(2))))
    polynomial_unitary = lambda x0, x1, z12: ssla.expm(
        -1j * z12 * ZZ - 1j * x0 * np.kron(X, np.eye(2)) + 1j * x1 * np.kron(np.eye(2),
                                                                             X) + 1j * x1 * z12 * ZY + 1j * x0 * z12 * YZ
    )
    # + 1j * x1 * z12 * ZY - 1j * x0 * z12 * YZ
    print(circuit_unitary(*(params)) - polynomial_unitary(*(params)))
    # gradf = 1j * np.random.randn(len(lie_algebra_vector))
    # eta = 0.1
    # lie_algebra_polynomial = [lie_algebra_polynomial[i] - eta * gradf[i] for i in
    #                           range(len(lie_algebra_vector))]
    # print(lie_algebra_polynomial)
    #
    # lie_algebra_polynomial_python = [lambdify(symbols, f) for f in lie_algebra_polynomial]
    #
    # def cost(x):
    #     return np.mean([np.abs(f(*x)) for f in lie_algebra_polynomial_python])
    #
    # result = minimize(cost, x0=[0.1, 0.1, 0.1], method='Nelder-Mead')


def closeness_2():
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ZZ = np.kron(Z, Z)
    ZY = np.kron(Z, Y)
    YZ = np.kron(Y, Z)
    I = np.eye(2)
    un1 = ssla.expm(-1j * np.kron(X, I) - 1j * np.kron(I, X))
    un2 = ssla.expm(-1j * np.kron(X, I)) @ ssla.expm(-1j * np.kron(I, X))
    print(np.allclose(un1, un2))
    p1 = PauliMonomial([Pauli(0, 'Z'), Pauli(1, 'Z'), Pauli(2, 'I')])
    p2 = PauliMonomial([Pauli(0, 'I'), Pauli(1, 'X'), Pauli(2, 'Z')])
    # print(((p1.commutator(p2)).commutator(p1)).commutator())
    print((p1.commutator(p2)))
    print((p1.commutator(p2)).commutator(p2))
    print((p1.commutator(p2)).commutator(p1))
    A = 1j*np.kron(X, np.kron(Z, I))
    B = 1j*np.kron(I, np.kron(X, Z))
    A_comm_B = A @ B - B @ A
    print( A @ A_comm_B - A_comm_B@A )
    print(np.allclose(np.zeros((8,8)), A @ A_comm_B -A_comm_B@A ))
    print(np.allclose(np.zeros((8,8)), B @ A_comm_B -A_comm_B@B ))

    un1 = ssla.expm(A + B + 0.5 *A_comm_B)
    un2 = ssla.expm(A) @ ssla.expm(B)
    print(un1)
    print(un2)
    print(np.allclose(un1, un2))

def bch_recusion(X:PauliMonomial, Y:PauliMonomial, terms):
    terms.append([])
    n = len(terms)
    print(n)
    for t in terms[-2]:
        terms[-1].append(X.commutator(t))
        if terms[-1][-1].coeff ==0:
            terms[-1].pop()
        else:
            terms[-1][-1].coeff /= np.math.factorial(n)
        terms[-1].append(Y.commutator(t))
        if terms[-1][-1].coeff ==0:
            terms[-1].pop()
        else:
            terms[-1][-1].coeff /= -1*np.math.factorial(n)

def closeness_3():
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ZZ = np.kron(Z, Z)
    ZY = np.kron(Z, Y)
    YZ = np.kron(Y, Z)
    I = np.eye(2)
    un1 = ssla.expm(-1j * np.kron(X, I) - 1j * np.kron(I, X))
    un2 = ssla.expm(-1j * np.kron(X, I)) @ ssla.expm(-1j * np.kron(I, X))
    print(np.allclose(un1, un2))
    p1 = PauliMonomial([Pauli(0, 'Z', coeff=sp.Symbol('x')), Pauli(1, 'Z'), Pauli(2, 'I')])
    p2 = PauliMonomial([Pauli(0, 'I', coeff=sp.Symbol('y')), Pauli(1, 'X'), Pauli(2, 'Z')])
    terms = [[p1, p2]]
    for i in range(3):
        bch_recusion(p1, p2, terms)

    for t in terms:
        print(t)
    # print(p1.commutator(p2))
    # print((p1.commutator(p2)).commutator(p1))
    # print((p1.commutator(p2)).commutator(p2))
    # print(((p1.commutator(p2)).commutator(p2)).commutator(p1))
    # print(((p1.commutator(p2)).commutator(p2)).commutator(p2))
    # print(((p1.commutator(p2)).commutator(p1)).commutator(p1))
    # print(((p1.commutator(p2)).commutator(p1)).commutator(p2))
def main():
    nqubits = 2
    dev = qml.device('default.qubit', wires=nqubits)

    np.random.seed(234234)
    symbols = sp.symbols('x0,x1,z12')
    print(symbols)
    circuit = [PauliMonomial([Pauli(0, 'X', coeff=-1j * symbols[0]), Pauli(1, 'I')]),
               PauliMonomial([Pauli(0, 'I'), Pauli(1, 'X', coeff=-1j * symbols[1])]),
               PauliMonomial([Pauli(0, 'Z', coeff=-1j * symbols[2]), Pauli(1, 'Z')])]
    print(circuit[0])
    print(circuit[2])
    print(circuit[0].commutator(circuit[2]))
    bch_terms = dict(zip(range(len(circuit)), [set() for _ in range(len(circuit))]))
    for i in range(len(circuit) - 1):
        bch_terms[i] |= set([circuit[i]])
        bch_terms[i + 1] |= set([circuit[i + 1]])
        print(bch_terms)

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
        print(new_bch_terms, 'new')
        bch_terms[i + 1] |= new_bch_terms
    print(bch_terms)
    lie_algebra_vector = []
    lie_algebra_polynomial = []
    for t in bch_terms[2]:
        print(t.to_oranssi(), t.coeff)
        lie_algebra_vector.append(t.to_oranssi())
        lie_algebra_polynomial.append(t.coeff)
    gradf = 1j * np.random.randn(len(lie_algebra_vector))
    eta = 0.1
    lie_algebra_polynomial = [lie_algebra_polynomial[i] - eta * gradf[i] for i in
                              range(len(lie_algebra_vector))]
    print(lie_algebra_polynomial)

    lie_algebra_polynomial_python = [lambdify(symbols, f) for f in lie_algebra_polynomial]

    def cost(x):
        return np.mean([np.abs(f(*x)) for f in lie_algebra_polynomial_python])

    result = minimize(cost, x0=[0.1, 0.1, 0.1], method='Nelder-Mead')
    print(result.x)


if __name__ == '__main__':
    # closeness_2()
    closeness_3()
    # main()
