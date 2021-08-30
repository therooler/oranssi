import pennylane as qml
from oranssi.optimizers import parameter_shift_optimizer, local_custom_su_lie_optimizer, \
    local_su_4_lie_optimizer
from oranssi.plot_utils import change_label_fontsize, LABELSIZE, LINEWIDTH, reds, blues
from oranssi.circuit_tools import get_all_su_n_directions
import matplotlib.pyplot as plt
import numpy as np

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


costs_lie, unitaries = local_custom_su_lie_optimizer(circuit, params=init_params, layer_pattern=[(2, 0)],
                                                observables=observables,
                                                device=device, eta=.1, nsteps=150,
                                                tol=1e-6, trotterize=True,
                                                return_unitaries=True)

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
axs.plot(costs, color=reds(0.7), label='VQE Opt.', linewidth=LINEWIDTH)
axs.plot(costs_lie, color=blues(0.7), label='Riemann Opt.', linewidth=LINEWIDTH)
axs.set_xlabel('Step')
axs.set_ylabel(r'$\langle H \rangle$')
axs.plot(range(max(len(costs), len(costs_lie))),
         [-5.226251859505502 for _ in range(max(len(costs), len(costs_lie)))],
         label='Minimum',
         color='gray', linestyle='--', zorder=-1)
axs.legend()
change_label_fontsize(axs, LABELSIZE)
fig.savefig('./figures/tfim_ps_comparison.pdf')

omegas = []
for uni in unitaries:
    omegas.append(get_all_su_n_directions(uni, observables, nqubits, device))

stepstotal = len([om['XXXX'] for om in omegas])
omegas_length_m = {0: np.zeros(stepstotal), 1: np.zeros(stepstotal), 2: np.zeros(stepstotal),
                   3: np.zeros(stepstotal)}

for k in omegas[0].keys():
    if k.count('I') == 3:
        omegas_length_m[3] += np.abs([om[k] for om in omegas])
    if k.count('I') == 2:
        omegas_length_m[2] += np.abs([om[k] for om in omegas])
    if k.count('I') == 1:
        omegas_length_m[1] += np.abs([om[k] for om in omegas])
    if k.count('I') == 0:
        omegas_length_m[0] += np.abs([om[k] for om in omegas])

fig2, ax2 = plt.subplots(1, 1)
fig2.set_size_inches(6, 6)
ax2.plot([0 for _ in range(len(omegas_length_m[0]))], label='Min.', color='Black', linestyle='--')

cmap = plt.get_cmap('Set1')
for i in range(4):
    ax2.plot(omegas_length_m[i], label=rf'SU$({2 ** (4 - i)})$', color=cmap((i + 1) / 5),
             linewidth=LINEWIDTH)
    ax2.set_xlabel('Step')
    ax2.set_ylabel(r'$\sum_i |\omega_i|$')
ax2.legend()
change_label_fontsize(ax2, LABELSIZE)
fig2.tight_layout()
fig2.savefig('./figures' + f'/stochastic_su16_abs_optimizer_ps_comp.pdf')
plt.show()
