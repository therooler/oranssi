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
