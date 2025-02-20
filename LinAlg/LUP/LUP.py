from nptyping import NDArray, Shape, Float32, Int8
import numpy as np


def findLUP(matrix: NDArray[Shape["Any, Any"], Float32]) -> tuple[
    NDArray[Shape["Any, Any"], Int8],
    NDArray[Shape["Any, Any"], Float32],
    NDArray[Shape["Any, Any"], Float32]
]:
    """
    It is Doolittle's algorithm
    """

    matrix = matrix.copy().astype(np.float32)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError

    P = np.arange(matrix.shape[0], dtype=np.int32)

    C = matrix

    for i in range(matrix.shape[0] - 1):
        max_el_index = np.abs(C[i:, i]).argmax() + i

        C[[i, max_el_index], :] = C[[max_el_index, i], :]
        P[i], P[max_el_index] = P[max_el_index], P[i]

        max_el = C[i, i]

        C[i + 1:, i] /= max_el

        C[i + 1:, i + 1:] -= C[i + 1:, i].reshape(-1, 1) @ C[i, i + 1:].reshape(1, -1)



    U = np.triu(C)
    L = np.tril(C)

    np.fill_diagonal(L, 1)

    P_ret = np.zeros(matrix.shape, dtype=np.int8)

    P_ret[
        np.arange(
            matrix.shape[0], dtype=np.int32
        ),
        P
    ] = 1

    return P_ret, L, U




