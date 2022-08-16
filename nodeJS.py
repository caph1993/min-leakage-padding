"""
Sparse reimplementation of all the algorithms
"""
from functools import lru_cache, wraps
import itertools
from pathlib import Path
import sys
import time
from typing import List, Optional, Sequence, Union
import numpy as np
from typing import Tuple
from scipy.sparse import dok_array
from tqdm import tqdm

IntArray = np.ndarray  # Just for reference
FloatArray = np.ndarray  # Just for reference

sys.setrecursionlimit(10**9)

# Utils


def max_Y_given_X(S_X: IntArray, c: float):
    '''
    i can only be padded to j in range(i, max_j[i]+1)
    '''
    n = len(S_X)
    max_j = np.array([n - 1] * n)
    for i in range(n):
        for j in range(i + 1, n):
            if S_X[j] > c * S_X[i]:
                max_j[i] = j - 1
                break
    return max_j


def min_X_given_Y(S_X: IntArray, c: float):
    '''
    j can only be output of i in range(min_i[j], j+1)
    '''
    n = len(S_X)
    min_i = np.array([0] * n)
    for j in range(n):
        for i in range(j - 1, -1, -1):
            if c * S_X[i] < S_X[j]:
                min_i[j] = i + 1
                break
    return min_i


def test_utils(S_X: IntArray, c):
    n = len(S_X)
    min_i = min_X_given_Y(S_X, c)
    max_j = max_Y_given_X(S_X, c)
    for i in range(n):
        for j in range(n):
            can1 = (S_X[i] <= S_X[j] <= c * S_X[i])
            can2 = min_i[j] <= i <= j
            can3 = i <= j <= max_j[j]
            assert can1 == can2, (i, j, can1, can2, can3, S_X[i], S_X[j], c)
    return


def file_parser_iterator(path: Union[str, Path, None]):
    if path is None:
        stdin = iter(x for l in sys.stdin for x in l.split())
    else:
        stdin = iter(open(path).read().split())
    n_test_cases = int(next(stdin))

    def _iterator():
        for tc in range(1, 1 + n_test_cases):
            n, c = int(next(stdin)), float(next(stdin))
            sizes = [int(next(stdin)) for _ in range(n)]
            freqs = [float(next(stdin)) for _ in range(n)]
            yield tc, n, c, sizes, freqs

    return n_test_cases, iter(_iterator())


# Solvers


def PRP_Renyi_only(S_X: IntArray, P_X: FloatArray, c: float):
    n = len(S_X)
    min_i = min_X_given_Y(S_X, c)

    B_X = P_X.copy()  # budget for each X

    P_XY = dok_array((n, n))
    for j in tqdm(range(n - 1, -1, -1)):
        if B_X[j] == 0:
            continue
        B_X_j = B_X[j]
        for i in range(j, min_i[j] - 1, -1):
            P_XY[i, j] = min(B_X_j, B_X[i])
            B_X[i] -= P_XY[i, j]

    P_Y_given_X = P_XY.multiply(1 / P_X[:, None]).asformat('dok')
    return P_Y_given_X, locals()


def PRP_Renyi_bandwidth(S_X: IntArray, P_X: FloatArray, c: float, pre=None):
    # Improve bandwidth
    n = len(S_X)
    P_Y_given_X, scope = pre or PRP_Renyi_only(S_X, P_X, c)
    P_XY, min_i = scope['P_XY'], scope['min_i']
    max_j = max_Y_given_X(S_X, c)

    P_XY_max = P_XY.asformat('coo').max(axis=0).toarray().ravel()

    # Find pinned coordinates
    argmax_i = {
        j: min_i[j] + np.argmax([P_XY[i, j] for i in range(min_i[j], j + 1)
                                ]) for j in range(n) if P_XY_max[j] > 0
    }
    pinned = {i: [] for i in range(n)}
    for j, i in argmax_i.items():
        pinned[i].append(j)

    # Pin them
    out_P_XY = dok_array((n, n))
    for i in pinned:
        for j in pinned[i]:
            out_P_XY[i, j] = P_XY[i, j]

    # Greedily assign the rest favoring the leftmost
    for i in range(n):
        budget = P_X[i] - sum(P_XY[i, j] for j in pinned[i])
        for j in range(i, max_j[i] + 1):
            if np.allclose(budget, 0):
                break
            if j in pinned[i]:
                continue
            db = min(budget, P_XY_max[j])
            out_P_XY[i, j] = db
            budget -= db
        assert np.allclose(budget, 0)

    P_XY = out_P_XY
    P_Y_given_X = P_XY.multiply(1 / P_X[:, None]).asformat('dok')
    return P_Y_given_X, locals()


def POP_Renyi_only(S_X: IntArray, P_X: FloatArray, c: float):
    # Assume S_X are sorted
    n = len(S_X)
    min_i = min_X_given_Y(S_X, c)
    max_j = max_Y_given_X(S_X, c)

    #progress = tqdm(total=int(np.sum(max_j - np.arange(n) + 1)))

    @lru_cache(maxsize=None)
    def f(LO: int, HI: int) -> Tuple[float, int, int, int]:
        #progress.update(1)
        assert LO <= HI
        if LO == HI:
            return (0, n + 1, LO, HI)
        i_max = LO + np.argmax(P_X[LO:HI])
        ANS = (float('inf'), n + 1, n + 1, n + 1)
        for j_max in range(i_max, max_j[i_max] + 1):
            lo = max(LO, min_i[j_max])
            hi = min(HI, j_max + 1)
            assert LO <= lo <= i_max <= hi <= HI
            renyi = f(LO, lo)[0] + P_X[i_max] + f(hi, HI)[0]
            ans = (renyi, j_max, lo, hi)
            ANS = min(ANS, ans)
        return ANS

    #progress.close()
    # Reconstruction
    Y_given_X = np.zeros(n, dtype=int)
    Q: List[Tuple[int, int]] = [(0, n)]
    while Q:
        LO, HI = Q.pop()
        _, j, lo, hi = f(LO, HI)
        Y_given_X[lo:hi] = j
        if LO < lo:
            Q.append((LO, lo))
        if hi < HI:
            Q.append((hi, HI))

    # Compute P_Y_given_X and P_XY

    P_Y_given_X = dok_array((n, n))
    P_Y_given_X[np.arange(n), Y_given_X] = 1

    return P_Y_given_X, locals()


def POP_Renyi_bandwidth(S_X: IntArray, P_X: FloatArray, c: float, pre=None):
    n = len(S_X)
    P_Y_given_X, scope = pre or POP_Renyi_only(S_X, P_X, c)
    Y_given_X = scope['Y_given_X']

    P_XY_max = P_Y_given_X.multiply(P_X[:, None]).max(axis=0).toarray().ravel()

    Y_given_X = np.array([
        next(
            (j for j in range(i, Y_given_X[i]) if P_X[i] <= P_XY_max[j]),
            Y_given_X[i],
        ) for i in range(n)
    ])
    P_Y_given_X = dok_array((n, n))
    P_Y_given_X[np.arange(n), Y_given_X] = 1
    return P_Y_given_X, locals()


def POP_Shannon_only(S_X: IntArray, P_X: FloatArray, c: float):
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
    n = len(P_X)

    max_j = max_Y_given_X(S_X, c)

    xlogx = lambda x: x * np.log2(x)

    # Solution computation
    F: List[Tuple[float, int]]
    F = [(0, 0) for _ in range(n + 1)]
    F[n] = (0, n)
    for i in range(n - 1, -1, -1):
        freqs_i_j = 0
        F[i] = (float('inf'), 0)
        for j in range(i, max_j[i] + 1):
            freqs_i_j += P_X[j]  # freqs_i_j = sum of freqs in [i..j]
            info = -xlogx(freqs_i_j) + F[j + 1][0]
            F[i] = min(F[i], (info, j))

    # Solution reconstruction
    Y_given_X = np.zeros(n, dtype=int)
    i = 0
    while i != n:
        _, j = F[i]
        assert i <= j
        Y_given_X[i:j + 1] = j
        i = j + 1

    P_Y_given_X = dok_array((n, n))
    P_Y_given_X[np.arange(n), Y_given_X] = 1
    return P_Y_given_X, locals()


def nodeJS():
    file = Path('paper-cases/npm.txt')
    if not file.exists():
        print('Creating dataset from original...')
        import pandas as pd
        df = pd.read_csv(
            './datasets/npm_no_scope_full_stats_nonzero_downloads.csv',
            names=['name', 'size', 'visits'],
        )
        df.sort_values(by='visits', ascending=False, inplace=True)
        df.sort_values(by='size', inplace=True)
        S_X: IntArray = df['size'].values  # type: ignore
        P_X: FloatArray = (df['visits'] / df['visits'].sum()).values
        del df
        c = 1.1
        with open(file, 'w') as f:
            f.write(f'1\n{len(S_X)} {c}\n')
            f.write(f'{" ".join(map(str, S_X))}\n')
            f.write(f'{" ".join(map(str, P_X))}\n')
        print(f'Saved as {file}.')
        print('Avg Bandwidth:', np.dot(S_X, P_X))
    _, _, _, sizes, freqs = next(file_parser_iterator(file)[1])
    S_X = np.array(sizes)
    P_X = np.array(freqs)
    return S_X, P_X


def measure(solver, S_X: IntArray, P_X: FloatArray, c: float, **kwargs):
    assert len(S_X) == len(P_X) and c >= 1
    assert np.all(S_X >= 0) and np.all(S_X[:-1] <= S_X[1:])
    assert np.all(P_X >= 0) and np.allclose(P_X.sum(), 1)

    def xlog2x_sparse(x: dok_array):
        out = x.copy()
        mask = x.nonzero()
        masked = x[mask].toarray()
        out[mask] = masked * np.log2(masked)
        return out

    def xlog2x_array(x: FloatArray):
        out = x.copy()
        mask = x.nonzero()
        out[mask] = x[mask] * np.log2(x[mask])
        return out

    assert np.allclose(np.sum(P_X), 1)
    print('Running...')
    start = time.time()
    P_Y_given_X: dok_array
    if c == 1.0:
        P_Y_given_X = dok_array((len(S_X), len(S_X)))
        P_Y_given_X.setdiag(1)
        scope = {}
    else:
        P_Y_given_X, scope = solver(S_X, P_X, c, **kwargs)
    end = time.time()
    elapsed = end - start

    print('Verifying solution...')
    assert np.allclose(P_Y_given_X.sum(axis=1), 1)

    print('Computing join matrix...')
    P_X = np.array(P_X)
    P_XY = P_Y_given_X.multiply(P_X[:, None]).asformat('dok')

    def leakages():
        P_Y = P_XY.sum(axis=0)
        Q_Y = P_XY.asformat('coo').max(axis=0).toarray()
        H_X_inf = -np.log2(np.max(P_X))
        H_Y_given_X_inf = -np.log2(np.sum(Q_Y))
        H_Y = -xlog2x_array(P_Y).sum()
        H_Y_given_X = -np.dot(P_X, xlog2x_sparse(P_Y_given_X).sum(axis=1))
        renyi = H_X_inf - H_Y_given_X_inf
        shannon = H_Y - H_Y_given_X
        return renyi, shannon

    def bandwidth_factor():
        ones = P_XY.copy()
        ones[ones.nonzero()] = 1
        in_size = ones * S_X[:, None]
        out_size = ones * S_X[None, :]
        used_bandwidth = (P_XY * (out_size - in_size)).sum()
        min_bandwidth = np.dot(P_X, S_X)
        return used_bandwidth / min_bandwidth

    # from shared_functions import leakage_renyi, leakage_shannon
    # print(leakage_renyi(P_Y_given_X.toarray(), P_X))
    # print(leakage_shannon(P_Y_given_X.toarray(), P_X))

    print('Computing leakages...')
    renyi, shannon = leakages()
    print('Computing bandwidths...')
    bandwidth = bandwidth_factor()
    print('Done.')
    measurements = {
        'renyi': renyi,
        'shannon': shannon,
        'elapsed': elapsed,
        'bandwidth': bandwidth,
    }
    alg_output = P_Y_given_X, scope
    return measurements, alg_output


def main():
    S_X, P_X = nodeJS()

    solvers = [
        ('PRP_Renyi_bandwidth', PRP_Renyi_bandwidth, 'PRP_Renyi_only'),
        ('POP_Renyi_bandwidth', POP_Renyi_bandwidth, 'POP_Renyi_only'),
        ('PRP_Renyi_only', PRP_Renyi_only, None),
        ('POP_Renyi_only', POP_Renyi_only, None),
        ('POP_Shannon_only', POP_Shannon_only, None),
    ]
    # Independent first order:
    solvers.sort(key=lambda x: x[2] != None)

    case = 'top-1000'

    if case == 'top-1000':
        idx = np.argsort(P_X)[-1000:]
        S_X = S_X[idx]
        P_X = P_X[idx]
        P_X /= np.sum(P_X)
        idx = np.argsort(S_X)
        S_X = S_X[idx]
        P_X = P_X[idx]
    else:
        case = 'all'

    with open(f'paper-cases/.{case}.txt', 'a') as f:
        for c in [1.30, 1, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
            Measurements = {}
            Outputs = {}
            for name, solver, dependency in solvers:
                print('-' * 30)
                print(name, c)

                # Inject pre-computed outputs:
                kwargs = {'pre': Outputs[dependency]} if dependency else {}
                # Run and measure
                measurements, output = measure(solver, S_X, P_X, c, **kwargs)
                # Fix time
                if dependency:
                    prev = Measurements[dependency]
                    measurements['elapsed'] += prev['elapsed']

                Outputs[name] = output
                Measurements[name] = measurements

                measurements = {'name': name, 'c': c, **measurements}
                f.write(f'{measurements}\n')
                f.flush()
                print(measurements)
    return


if __name__ == '__main__':
    main()