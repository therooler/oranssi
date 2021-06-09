![alt text](resources/oranssi-logo.png "Logo Title Text 1")
****************************************************
Oranssi: Riemannian optimization of quantum circuits.

The code base consists of the following modules:

- `oranssi.circuit_tools`: All the tools needed to handle PennyLane circuits and operators and convert them to
data types that the Oranssi optimizers can handle.
- `oranssi.optimizers` Different Riemannian optimizers, ranging from matrix exponential-based to full quantum circuit
simulations. At the moment, the optimizers rely on the `LocalLieLayer` class for the gradient flow.
- `oranssi.plot_utils`: Utility functions used for plotting.
- `oranssi.utils`: General utility functions

Developed on Python 3.8.8, see `requirements.txt` for the necessary packages. Library can be tested by running pytest
on the tests in `oranssi/tests/`.