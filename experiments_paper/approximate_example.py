import pennylane as qml
from oranssi.optimizers import  approximate_lie_optimizer
from oranssi.plot_utils import change_label_fontsize, LABELSIZE, LINEWIDTH, reds, blues, \
    plot_su8_directions_individually
from oranssi.circuit_tools import get_all_su_n_directions, get_hamiltonian_matrix
from oranssi.opt_tools import CustomDirectionLieAlgebraLayer
import matplotlib.pyplot as plt
import numpy as np
import os

nqubits = 2
device = qml.device('default.qubit', wires=nqubits)
init_params = [0.1, 0.1, 0.1, 0.1]
plt.rc('font', family='serif')


def circuit(params, **kwargs):
    for n in range(nqubits):
        qml.Hadamard(wires=n)
    return qml.state()

observables=  [qml.PauliX(0), qml.PauliY(0)@qml.PauliZ(1)]

H = get_hamiltonian_matrix(nqubits, observables)
eigvals = np.linalg.eigvalsh(H)

costs_exact, unitaries = approximate_lie_optimizer(circuit, params=init_params, layers=[CustomDirectionLieAlgebraLayer(device, observables, directions=['YY', 'ZZ'])],
                                                          observables=observables,
                                                          device=device, eta=.1, nsteps=50,
                                                          tol=1e-6, return_unitary=True)


fig0, axs0 = plot_su8_directions_individually(unitaries, observables, device)
axs0.set_ylabel(r'$\omega_i$')
axs0.set_xlabel('Step')


left, bottom, width, height = [0.54, 0.15, 0.35, 0.3]
ax_inset = fig0.add_axes([left, bottom, width, height])
# axs.plot(np.abs(np.array(costs)-min(eigvals)), color=reds(0.7), label='VQE Opt.', linewidth=LINEWIDTH)
ax_inset.plot(np.abs(np.array(costs_exact) - min(eigvals)), color='black',
              label='Riemann Opt.', linewidth=LINEWIDTH)
# These are in unitless percentages of the figure size. (0,0 is bottom left)
# left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
# ax_inset = fig.add_axes([left, bottom, width, height])
# ax_inset.plot(range(10), color='red')
# ax_inset.plot(range(6)[::-1], color='green')
# ax_inset.set_xlabel('Step')
ax_inset.set_xticks([])
ax_inset.set_ylabel(r'$\epsilon_{res}$')
# ax_inset.plot(range(len(costs_exact)),
#          [min(eigvals) for _ in range( len(costs_exact))],
#          label='Minimum',
#          color='gray', linestyle='--', zorder=-1)
ax_inset.set_yscale('log')
# ax_inset.legend(prop={'size':7})
change_label_fontsize(ax_inset, LABELSIZE)
fig0.savefig('./figures/simple_approximate_example.pdf',bbox_inches="tight")
plt.show()
#
# omegas = []
# if not os.path.exists('./data/tfim_su4'):
#     os.makedirs('./data/tfim_su4')
#
# for i, uni in enumerate(unitaries):
#     np.save(f'./data/tfim_su4/uni_{i}', uni)
#     omegas.append(get_all_su_n_directions(uni, observables, device))
#
# fig1, ax = plt.subplots(1, 1)
# fig1.set_size_inches(6,6)
# cmap = plt.get_cmap('Reds')
#
# ax.plot(costs_exact, label=r'Lie $SU(2^n)$ Stoch.', color=cmap(0.3), zorder=-1)
# ax.plot(range(len(costs_exact)), [np.min(eigvals) for _ in range(len(costs_exact))],
#         label='Min.',
#         color='gray', linestyle='--', zorder=-1)
# plt.show()
# fig, axs = plot_su16_directions(nqubits, unitaries, observables, device)
# fig.savefig('./figures' + f'/su4_linesearch_optimizer_tfim_{nqubits}.pdf',bbox_inches="tight")
# plt.show()
