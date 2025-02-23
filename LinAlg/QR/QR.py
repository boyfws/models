from nptyping import NDArray, Shape, Float32, Int8
import numpy as np


def getQR(
        matrix: NDArray[Shape["Any, Any"], Float32]
        ) -> tuple[
    NDArray[Shape["Any, Any"], Float32],
    NDArray[Shape["Any, Any"], Float32]
    ]:
    """
    Here we use Householder method to find QR decomposition
    """
    m, n = matrix.shape

    assert m >= n

    Q = np.eye(m, dtype=np.float32)
    R = matrix.copy().astype(np.float32)

    e1 = np.zeros(m, dtype=np.int8)
    e1[0] = 1

    for k in range(n):
        x = R[k:m, k]

        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        v = x - alpha * e1[:m - k]
        v = v / np.linalg.norm(v)

        H = np.eye(m)
        H[k:m, k:m] -= 2 * np.outer(v, v)

        R = H @ R
        Q = Q @ H.T

    return Q, R