import numpy as np
import pytest
from oranssi.plot_utils import state_to_spherical, spherical_to_state


@pytest.mark.parametrize('phi, theta', [[0, 0], [1, 1], [2.3, 3.1], [0, 1]])
def test_state_conversion(phi, theta):
    rho = spherical_to_state(phi,theta)
    phi_out, theta_out = state_to_spherical(rho)
    assert np.allclose([phi, theta], [phi_out, theta_out], atol=1e-7)