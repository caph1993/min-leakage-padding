from typing import List, Optional, Sequence, Tuple, TypeVar, Union
import numpy as np
from numpy.typing import NDArray
from shared_functions import sorting_sizes, xlog2x
from functools import lru_cache
from itertools import product


@sorting_sizes
def Renyi_POP_bruteforce_1(sizes: Sequence[int], freqs: Sequence[float],
                           c: float):
    'Can not handle more than 10'
    n = len(freqs)
    S_X = np.array(sizes)

    poss_Y_given_X = [
        [j for j in range(i, n) if S_X[j] <= S_X[i] * c] for i in range(n)
    ]

    plogp = lambda t: t * np.log2(t)

    def renyi_shannon(f):
        p = {}
        for x, y in enumerate(f):
            p[y] = p.get(y, [])
            p[y].append(freqs[x])
        renyi = sum(max(l) for l in p.values())
        shannon = -sum(plogp(sum(l)) for l in p.values())
        return renyi, shannon

    mapping = min(product(*poss_Y_given_X), key=lambda f: renyi_shannon(f))

    # Solution reconstruction
    P_Y_given_X = np.zeros((n, n))
    for i in range(n):
        P_Y_given_X[i, mapping[i]] = 1
    return P_Y_given_X


@sorting_sizes
def Renyi_POP_bruteforce_non_decreasing(sizes: Sequence[int],
                                        freqs: Sequence[float], c: float):
    '''
    Assumes that the mapping is non-decreasing.
    FAILS because of this
    '''
    n = len(freqs)
    S_X = np.array(sizes)

    poss_Y_given_X = [
        [j for j in range(i, n) if S_X[j] <= S_X[i] * c] for i in range(n)
    ]

    EMPTY = (0, 0, -1, -1)
    INF = (float('inf'), float('inf'), -1, -1)

    @lru_cache(maxsize=None)
    def backtrack(lo: int, min_j: int):
        if lo == n:
            return EMPTY
        ANS = INF
        for col in poss_Y_given_X[lo]:
            if col < min_j:
                continue
            col_renyi = 0
            col_shannon = 0
            for hi in range(lo + 1, n + 1):
                if hi - 1 > min_j:
                    break
                col_renyi = max(col_renyi, freqs[hi - 1])
                col_shannon += freqs[hi - 1]
                rest_renyi, rest_shannon, _, _ = backtrack(hi, col + 1)
                shannon = -plogp(col_shannon) + rest_shannon
                renyi = col_renyi + rest_renyi
                ans = (renyi, shannon, col, hi)
                ANS = min(ANS, ans)
        return ANS

    plogp = lambda t: t * np.log2(t)

    # Reconstruction
    P_Y_given_X: NDArray = np.zeros((n, n))
    Q: List[Tuple[int, int]] = [(0, 0)]
    while Q:
        lo, min_j = Q.pop()
        _, _, col, hi = backtrack(lo, min_j)
        P_Y_given_X[lo:hi, col] = 1
        if lo < n:
            Q.append((hi, col + 1))

    maps_to = np.argmax(P_Y_given_X, axis=1)
    assert np.all(np.diff(maps_to) >= 0), maps_to
    return P_Y_given_X
