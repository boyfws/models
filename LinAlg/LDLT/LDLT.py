from nptyping import NDArray, Shape, Float32, Int8
import numpy as np


def getLDLT(
        matrix: NDArray[Shape["Any, Any"], Float32]\
        ) -> tuple[
    NDArray[Shape["Any, Any"], Float32],
    NDArray[Shape["Any, Any"], Float32]
    ]:
    assert matrix.shape[0] == matrix.shape[1]
    matrix = matrix.copy().astype(np.float32)

    n = matrix.shape[0]

    D = np.zeros((n,n), dtype=np.float32)
    L = np.eye(n, dtype=np.float32)

    for i in range(n):
        diag_values = np.diagonal(D)
        D[i, i] = matrix[i, i] - diag_values[:i] @ L[i, :i] ** 2

        #for j in range(i + 1, n):
        #    L[j, i] = ( matrix[j, i] - (L[j, :i] * L[i, :i]) @ diag_values[:i] ) / D[i, i]

        mult = L[i, :i] * diag_values[:i]

        L[i + 1:, i] = (
                               matrix[i + 1:, i] - np.dot(L[i + 1:, :i], mult)
                       ) / D[i, i]

    return L, D
