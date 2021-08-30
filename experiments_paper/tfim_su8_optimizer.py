import matplotlib.pyplot as plt
import pennylane as qml
from oranssi.optimizers import su8_lie_optimizer
import os
from oranssi.circuit_tools import get_all_su_n_directions, get_hamiltonian_matrix
import numpy as np
from oranssi.plot_utils import change_label_fontsize, plot_su16_directions

np.set_printoptions(precision=3)

plt.rc('font', family='serif')

nqubits = 8
dev = qml.device('default.qubit', wires=nqubits)
eta = 1.0


def circuit(params, **kwargs):
    # LOWEST EIGENSTATE
    for n in range(nqubits):
        qml.Hadamard(wires=n)
    #     qml.RX(0.1, wires=n)

    # PERTURBATION
    # qml.RY(params[0], wires=0)

    return qml.state()


observables = [qml.PauliX(i) for i in range(nqubits)] + \
              [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(nqubits - 1)] + \
              [qml.PauliZ(nqubits - 1) @ qml.PauliZ(0)]
H = get_hamiltonian_matrix(nqubits, observables)
eigvals = np.linalg.eigvalsh(H)
print(f'Spectrum: {eigvals}')
params = []
costs_exact, unitaries = su8_lie_optimizer(circuit, params, observables,
                                           dev, eta=eta,
                                           return_unitary=True,
                                           nsteps=30, escape_tol=1e-5,
                                           add_su2=True, add_su4=True)
if not os.path.exists('./data/tfim_su8'):
    os.makedirs('./data/tfim_su8')

omegas = []
for i, uni in enumerate(unitaries):
    np.save(f'./data/tfim_su8/uni_{i}', uni)
    omegas.append(get_all_su_n_directions(uni, observables, nqubits, dev))

fig1, ax = plt.subplots(1, 1)
fig1.set_size_inches(8, 8)
cmap = plt.get_cmap('Reds')
ax.plot(costs_exact, label=r'Lie $SU(2^n)$ Stoch.', color=cmap(0.3), zorder=-1)
# axs.plot(costs_exact_unp, label=r'Lie $SU(2^n)$', color=cmap(0.2), zorder=-1)
ax.plot(range(len(costs_exact)), [np.min(eigvals) for _ in range(len(costs_exact))],
        label='Min.',
        color='gray', linestyle='--', zorder=-1)

fig, axs = plot_su16_directions(nqubits, unitaries, observables, dev)
fig.savefig('./figures' + f'/su8_optimizer_tfim_{nqubits}.pdf')
plt.show()
