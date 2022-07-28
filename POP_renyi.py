from typing import List, Optional, Sequence, Tuple, TypeVar, Union
import numpy as np
from numpy.typing import NDArray
from shared_functions import sorting_sizes
from functools import lru_cache

_T = TypeVar('_T', np.inexact, np.integer)
_NDArray = NDArray[_T]


@sorting_sizes
def Renyi_POP_basic(sizes: Sequence[int], freqs: Sequence[float], c: float):
    n = len(freqs)
    S_X = np.array(sizes)
    P_X = np.array(freqs)

    poss_Y_given_X = [
        [j for j in range(i, n) if S_X[j] <= S_X[i] * c] for i in range(n)
    ]
    poss_X_given_Y = [
        [i for i in range(0, j + 1) if S_X[j] <= S_X[i] * c] for j in range(n)
    ]
    min_poss_X_given_Y = [min(l) for l in poss_X_given_Y]
    max_poss_X_given_Y = [max(l) for l in poss_X_given_Y]

    @lru_cache(maxsize=None)
    def f(LO: int, HI: int) -> Tuple[float, int, int, int]:
        assert LO <= HI
        if LO == HI:
            return (0, n + 1, LO, HI)
        i_max = LO + np.argmax(P_X[LO:HI])
        ANS = (float('inf'), n + 1, n + 1, n + 1)
        for j_max in poss_Y_given_X[i_max]:
            lo = max(LO, min_poss_X_given_Y[j_max])
            hi = min(HI, max_poss_X_given_Y[j_max] + 1)
            assert LO <= lo <= i_max <= hi <= HI
            renyi = f(LO, lo)[0] + P_X[i_max] + f(hi, HI)[0]
            ans = (renyi, j_max, lo, hi)
            ANS = min(ANS, ans)
        return ANS

    # Reconstruction
    P_Y_given_X: NDArray = np.zeros((n, n))
    Q: List[Tuple[int, int]] = [(0, n)]
    while Q:
        LO, HI = Q.pop()
        _, j, lo, hi = f(LO, HI)
        P_Y_given_X[lo:hi, j] = 1
        if LO < lo:
            Q.append((LO, lo))
        if hi < HI:
            Q.append((hi, HI))
    return P_Y_given_X


@sorting_sizes
def Renyi_POP(sizes: List[int], freqs: List[float], c: float):
    n = len(freqs)

    # Assume sizes are sorted
    P_X = np.array(freqs)
    ref = Renyi_POP_basic(sizes, freqs, c)
    ref_XY = P_X[:, None] * ref

    ref_max = ref_XY.max(axis=0)
    X_is_pinned = ref_max > 0

    poss_Y_given_X = [[i] if X_is_pinned[i] else [
        j
        for j in range(i, n)
        if sizes[j] <= c * sizes[i] and P_X[i] <= ref_max[j]
    ]
                      for i in range(n)]
    poss = poss_Y_given_X  # Shortcut

    @lru_cache(maxsize=None)
    def f(LO: int, HI: int) -> Tuple[float, int, int, int]:
        '''
        Optimal assignment of the elements i such that
            - i is in [LO, HI)
            - poss_Y_given_X[i] is a subset of [LO, HI)
        Divide and conquer strategy:
            As many elements as possible are assigned to a single j,
            and for the remaining elements, recursion is used.
            Of course, always guaranteeing the initial Renyi leakage.
        Returns:
            - shannon: shannon leakage of the assignment (to be minimized)
            - indices j, LO, HI for reconstructing the channel
        '''
        assert LO <= HI
        if LO == HI:
            return (0, -1, LO, HI)
        elems = [
            i for i in range(LO, HI) if LO <= poss[i][0] <= poss[i][-1] < HI
        ]
        columns = sorted(set([j for i in elems for j in poss[i]]))
        if not columns:
            return (0, -1, LO, HI)
        ANS = (float('inf'), -1, -1, -1)
        for j in columns:
            # Greedy capture: all elems are mapped to j
            captured = [i for i in elems if poss[i][0] <= j <= poss[i][-1]]
            assert captured
            s = np.sum(P_X[captured])
            mid_shannon = -s * np.log2(s)
            # Subproblems
            lo, hi = j, j + 1
            left_shannon, *_ = f(LO, lo)
            right_shannon, *_ = f(hi, HI)
            shannon = left_shannon + mid_shannon + right_shannon
            ans = (shannon, j, lo, hi)
            ANS = min(ANS, ans)
        return ANS

    # Channel reconstruction
    P_Y_given_X = np.zeros((n, n))
    Q: List[Tuple[int, int]] = [(0, n)]
    while Q:
        LO, HI = Q.pop()
        _, j, lo, hi = f(LO, HI)
        if j == -1:
            continue
        elems = [
            i for i in range(LO, HI) if LO <= poss[i][0] <= poss[i][-1] < HI
        ]
        captured = [i for i in elems if poss[i][0] <= j <= poss[i][-1]]
        P_Y_given_X[captured, j] = 1
        if LO < lo:
            Q.append((LO, lo))
        if hi < HI:
            Q.append((hi, HI))
    return P_Y_given_X
