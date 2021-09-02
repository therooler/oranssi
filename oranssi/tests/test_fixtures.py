import pytest
import pennylane as qml
import numpy as np


@pytest.fixture
def circuit_1():
    device = qml.device('default.qubit', wires=2)

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.state()

    param_shape = (2,)
    return circuit, device, param_shape


@pytest.fixture
def circuit_2():
    device = qml.device('default.qubit', wires=3)

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[0], wires=1)
        qml.RZ(params[0], wires=2)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RZ(params[1], wires=0)
        qml.RZ(params[1], wires=1)
        qml.RZ(params[1], wires=2)
        return qml.state()

    param_shape = (2,)
    return circuit, device, param_shape


@pytest.fixture
def circuit_3():
    device = qml.device('default.qubit', wires=4)

    def circuit(params, **kwargs):
        for n in range(4):
            qml.Hadamard(wires=n)
            qml.RZ(params[0], wires=n)
        for n in range(3):
            qml.CNOT(wires=[n, n + 1])
        qml.CNOT(wires=[3, 0])
        for n in range(4):
            qml.RY(params[1], wires=n)
        return qml.state()

    param_shape = (2,)
    return circuit, device, param_shape

@pytest.fixture
def circuit_4():
    device = qml.device('default.qubit', wires=2)

    def circuit(params, **kwargs):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        return qml.state()

    param_shape = (0,)
    return circuit, device, param_shape

@pytest.fixture
def circuit_1_bad_return_types(request):
    if request.param == 'float':
        def circuit():
            device = qml.device('default.qubit', wires=2)

            def circuit(params, **kwargs):
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
                qml.RZ(params[0], wires=0)
                qml.RZ(params[0], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RZ(params[1], wires=0)
                qml.RZ(params[1], wires=1)
                return 1.0

            param_shape = (2,)
            return circuit, device, param_shape

        return circuit()
    elif request.param == 'observable':
        def circuit():
            device = qml.device('default.qubit', wires=2)

            def circuit(params, **kwargs):
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
                qml.RZ(params[0], wires=0)
                qml.RZ(params[0], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RZ(params[1], wires=0)
                qml.RZ(params[1], wires=1)
                return qml.expval(qml.PauliX(0))

            param_shape = (2,)
            return circuit, device, param_shape

        return circuit()


@pytest.fixture()
def circuit_4_state_obs(request):
    nqubits = request.param
    device = qml.device('default.qubit', wires=nqubits)
    param_shape = (2,)


    def circuit(params, **kwargs):
        observable = kwargs.get('observable')
        for n in range(nqubits):
            qml.Hadamard(wires=n)
            qml.RZ(params[0], wires=n)
        for n in range(nqubits - 1):
            qml.CNOT(wires=[n, n + 1])
        qml.CNOT(wires=[nqubits - 1, 0])
        for n in range(nqubits):
            qml.RY(params[1], wires=n)
        return qml.expval(observable)

    def circuit_state(params, **kwargs):
        for n in range(nqubits):
            qml.Hadamard(wires=n)
            qml.RZ(params[0], wires=n)
        for n in range(nqubits - 1):
            qml.CNOT(wires=[n, n + 1])
        qml.CNOT(wires=[nqubits - 1, 0])
        for n in range(nqubits):
            qml.RY(params[1], wires=n)
        return qml.state()

    return circuit, circuit_state, device, param_shape