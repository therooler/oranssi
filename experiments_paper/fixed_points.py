import numpy as np
import matplotlib.pyplot as plt
from oranssi.circuit_tools import get_all_su_n_directions, get_hamiltonian_matrix
import pennylane as qml

nqubits = 1
observables = [qml.PauliX(0), qml.PauliY(0)]


def unitary(r):
    norm = np.linalg.norm(r) ** 2
    r3 = np.sqrt(1 - norm)
    U = np.array([[r[0] + 1j * r[1], -r[2] + 1j * r3],
                  [r[2] + 1j * r3, r[0] - 1j * r[1]]])
    return U, np.allclose(U @ U.conj().T, np.eye(2))


gran = 20
grid = np.linspace(-1., 1., gran)
rr = np.meshgrid(*[grid for _ in range(3)])
rr = [r.flatten() for r in rr]

hamiltonian = get_hamiltonian_matrix(nqubits, observables)
eigenvalues = np.linalg.eigvalsh(hamiltonian)
print(eigenvalues)
dev = qml.device('default.qubit', wires=nqubits)

gradients = np.zeros((gran ** 3, 3))
costs = np.zeros(gran ** 3)
rho_0 = np.zeros((2, 2), dtype=complex)
rho_0[1, 0] = 1.0

for i, r in enumerate(zip(*rr)):
    U, uni_check = unitary(np.array(r))
    if uni_check:
        directions = get_all_su_n_directions(U, observables, nqubits, dev)
        gradients[i, 0] = directions['X']
        gradients[i, 1] = directions['Y']
        gradients[i, 2] = directions['Z']
        costs[i] = np.trace(U @ rho_0 @ U.conj().T @ hamiltonian).real
    else:
        gradients[i, :] = 0.0
        costs[i] = 0.0
plt.hist(np.sort(gradients[:, 0][gradients[:, 0] != 0]/2),alpha=0.3)
plt.hist(np.sort(gradients[:, 1][gradients[:, 1] != 0]/2),alpha=0.3)
plt.hist(np.sort(gradients[:, 2][gradients[:, 2] != 0]/2),alpha=0.3)
plt.show()

gradients = gradients.reshape((gran, gran, gran, 3))
costs = costs.reshape((gran, gran, gran))
minimum_idx = np.unravel_index(np.argmin(costs), dims=(gran, gran, gran))
print([grid[i] for i in minimum_idx])
rows = 4
cols = int(np.ceil(gran / rows))
fig, axs = plt.subplots(rows, cols)
import matplotlib.colors

norm = matplotlib.colors.Normalize(vmin=min(eigenvalues), vmax=max(eigenvalues), clip=False)
cmap = matplotlib.colors.ListedColormap(plt.get_cmap('viridis')(np.linspace(0.0, 1, 25)), "name")
for i, ax in enumerate(axs.flatten()):
    if i > (gran - 1):
        break
    ax.quiver(grid, grid, gradients[:, i, :, 0], gradients[:, i, :, 1], costs[:, i, :], cmap=cmap)
# c = plt.get_cmap('Reds')(costs.flatten())
# ax = plt.figure().add_subplot(projection='3d')
# q = ax.quiver(grid, grid, grid, gradients[:,:,:,0],gradients[:,:,:,1],gradients[:,:,:,2],
#           cmap='Reds', lw=2, color=c, length=0.1)
plt.show()
