import numpy as np
from emp.geometry import get_rotation_matrix


def test_rotation_matrix_properties():
    """Test that rotation matrices have correct mathematical properties."""
    # Test with a non-trivial set of angles
    axis = np.asarray([0.1, 0.2, 0.3])
    theta = np.pi / 2
    R = get_rotation_matrix(theta, axis)

    # Check that it's a 3x3 matrix
    assert R.shape == (3, 3)

    # Check orthogonality: R^T * R = I
    np.testing.assert_allclose(
        R.T @ R, np.eye(3), rtol=1e-10, atol=1e-12,
        err_msg="Matrix is not orthogonal"
    )

    # Check that determinant is +1 (proper rotation, not reflection)
    det = np.linalg.det(R)
    np.testing.assert_allclose(
        det, 1.0, rtol=1e-10, atol=1e-12,
        err_msg="Determinant is not 1 - not a proper rotation matrix"
    )
