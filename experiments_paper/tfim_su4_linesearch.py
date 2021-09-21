import pennylane as qml
from oranssi.optimizers import parameter_shift_optimizer, approximate_lie_optimizer
from oranssi.plot_utils import change_label_fontsize, LABELSIZE, LINEWIDTH, reds, blues, \
    plot_su16_directions
from oranssi.circuit_tools import get_all_su_n_directions, get_hamiltonian_matrix
from oranssi.opt_tools import SquaredLieAlgebraLayer
import matplotlib.pyplot as plt
import numpy as np
import os

nqubits = 4
device = qml.device('default.qubit', wires=nqubits)
init_params = [0.1, 0.1, 0.1, 0.1]
plt.rc('font', family='serif')


def circuit(params, **kwargs):
    for n in range(nqubits):
        qml.Hadamard(wires=n)
    for d in range(nqubits // 2):
        for n in range(0, nqubits - 1, 2):
            qml.CNOT(wires=[n, n + 1])
            qml.RZ(params[0 + d * 2], wires=n + 1)
            qml.CNOT(wires=[n, n + 1])
        for n in range(1, nqubits - 1, 2):
            qml.CNOT(wires=[n, n + 1])
            qml.RZ(params[0 + d * 2], wires=n + 1)
            qml.CNOT(wires=[n, n + 1])
        qml.CNOT(wires=[nqubits - 1, 0])
        qml.RZ(params[0 + d * 2], wires=0)
        qml.CNOT(wires=[nqubits - 1, 0])
        for n in range(nqubits):
            qml.RX(params[1 + d * 2], wires=n)


observables = [qml.PauliX(i) for i in range(nqubits)] + \
              [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
              [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]
costs, params_out = parameter_shift_optimizer(circuit, params=init_params, observables=observables,
                                              device=device, eta=0.01, nsteps=500, tol=1e-5,
                                              return_params=True)


def circuit(params, **kwargs):
    for n in range(nqubits):
        qml.Hadamard(wires=n)

    return qml.state()


H = get_hamiltonian_matrix(nqubits, observables)
eigvals = np.linalg.eigvalsh(H)

costs_exact, unitaries = approximate_lie_optimizer(circuit, params=init_params, layers=[SquaredLieAlgebraLayer(device, observables)],
                                                          observables=observables,
                                                          device=device, eta=.1, nsteps=50,
                                                          tol=1e-6, return_unitary=True)

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
axs.plot(costs, color=reds(0.7), label='VQE Opt.', linewidth=LINEWIDTH)
axs.plot(costs_exact, color=blues(0.7), label='Riemann Opt.', linewidth=LINEWIDTH)
axs.set_xlabel('Step')
axs.set_ylabel(r'$\langle H \rangle$')
axs.plot(range(max(len(costs), len(costs_exact))),
         [-5.226251859505502 for _ in range(max(len(costs), len(costs_exact)))],
         label='Minimum',
         color='gray', linestyle='--', zorder=-1)
axs.legend()
change_label_fontsize(axs, LABELSIZE)
fig.savefig('./figures/tfim_su4_linesearch_comparison.pdf')

omegas = []
if not os.path.exists('./data/tfim_su4'):
    os.makedirs('./data/tfim_su4')

for i, uni in enumerate(unitaries):
    np.save(f'./data/tfim_su4/uni_{i}', uni)
    omegas.append(get_all_su_n_directions(uni, observables, device))

fig1, ax = plt.subplots(1, 1)
fig1.set_size_inches(8, 8)
cmap = plt.get_cmap('Reds')

ax.plot(costs_exact, label=r'Lie $SU(2^n)$ Stoch.', color=cmap(0.3), zorder=-1)
ax.plot(range(len(costs_exact)), [np.min(eigvals) for _ in range(len(costs_exact))],
        label='Min.',
        color='gray', linestyle='--', zorder=-1)

fig, axs = plot_su16_directions(nqubits, unitaries, observables, device)
fig.savefig('./figures' + f'/su4_linesearch_optimizer_tfim_{nqubits}.pdf')
plt.show()
