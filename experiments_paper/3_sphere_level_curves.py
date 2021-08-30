import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def level_surface(r3, r4, const):
    return (-const * r3 + const * r4 - np.sqrt(-(r3 + r4) ** 2 * (
            const ** 2 + 8 * r3 ** 4 - 8 * r4 ** 2 + 8 * r4 ** 4 + 8 * r3 ** 2 *
            (-1 + 2 * r4 ** 2)))) / (4 * (r3 ** 2 + r4 ** 2))


c = 1.0
gran = 50
r_grid = np.linspace(-1, 1, gran)
rr3, rr4 = np.meshgrid(*[r_grid, r_grid])
rr2 = level_surface(rr3.flatten(), rr4.flatten(), c)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(rr3.shape, rr4.shape, rr2.shape)
ax.plot_surface(rr3, rr4, rr2.reshape((gran, gran)))
plt.show()
