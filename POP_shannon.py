from typing import List, Tuple
import numpy as np
from shared_functions import sorting_sizes


@sorting_sizes
def Shannon_POP(sizes: List[int], freqs: List[float], c: float):
    '''
    DP optimal solution to the object padding problem.

    Explanation:
        The objective is F[0].
        F[i] = (info, j) means that the optimal solution for the
        subproblem whose input is restricted to objects [i..n)
        has shannon information "info" and the first group in the
        partition is [i..j).
    
    Input:
        - list of (unique) object sizes [s_1, s_2, ..., s_n]
        - list of object frequencies [f_1, f_2, ..., f_n]
        - constant c >= 1
    Output:
        - partition of the collection of objects into groups
        For any two objects (i, j) in a group, s_i <= c*s_j
        Each group is a set of objects that will be padded to the same size.
        - list of group frequencies (sum of frequencies in each group)
        - Shannon information of the random variable for the size of the padded output.
    '''
    n = len(sizes)

    xlogx = lambda x: x * np.log2(x)

    # Solution computation
    F: List[Tuple[float, int]]
    F = [(0, 0) for _ in range(n + 1)]
    F[n] = (0, n)
    for i in range(n - 1, -1, -1):
        freqs_i_j = 0
        F[i] = (float('inf'), 0)
        for j in range(i + 1, n + 1):
            if sizes[j - 1] > c * sizes[i]:
                break
            # freqs_i_j = sum of freqs in [i..j)
            freqs_i_j += freqs[j - 1]
            info = -xlogx(freqs_i_j) + F[j][0]
            F[i] = min(F[i], (info, j))

    # Solution reconstruction
    P_Y_given_X = np.zeros((n, n))
    i = 0
    while i != n:
        _, j = F[i]
        assert i < j
        P_Y_given_X[i:j, j - 1] = 1
        i = j
    return P_Y_given_X
