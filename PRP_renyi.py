import itertools
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from shared_functions import sorting_sizes


@sorting_sizes
def Renyi_PRP(sizes: List[int], freqs: List[float], c: float):
    n = len(sizes)
    S_X = np.array(sizes)
    P_X = np.array(freqs)

    P_YX = np.zeros((n, n))

    for i in range(n - 1, -1, -1):
        q = P_X[i]
        if q == 0:
            continue
        for j in range(i, -1, -1):
            if not S_X[j] <= S_X[i] <= c * S_X[j]:
                break
            P_YX[j, i] = min(q, P_X[j])
            P_X[j] -= P_YX[j, i]

    P_X = P_YX.sum(axis=1)
    P_Y_given_X = P_YX
    P_Y_given_X[P_X != 0, :] /= P_X[P_X != 0][:, None]
    return P_Y_given_X
