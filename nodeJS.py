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


# Sparse matrix


class CS_Matrix:
    '''
    Constrained sparse matrix.
    The matrix is restricred to [i, j] where j in [min_j[i]..max_j[i]]
    '''

    def __init__(self, min_j: IntArray, max_j: IntArray):
        assert np.all(min_j <= max_j)
        assert np.all(np.diff(min_j) >= 0)
        assert np.all(np.diff(max_j) >= 0)
        n = len(min_j)
        m = max_j[-1] + 1
        assert min_j[0] <= 0 <= max_j[0]
        assert min_j[n - 1] <= m - 1 <= max_j[n - 1]
        self.shape = n, m
        self.min_j = min_j
        self.max_j = max_j
        self.min_i, self.max_i = self.inverse_bounds()
        self.dok = dok_array((n, m))

    def total_entries(self):
        return np.sum(self.max_j - self.min_j + 1)  # type: ignore

    def inverse_bounds(self):
        'j can only be output of i in [min_i[j]..max_i[j]]'
        n, m = self.shape
        min_i = np.array([-1] * n)
        j = 0
        for i in range(n):
            while j <= self.max_j[i]:
                min_i[j] = i
                j += 1
        max_i = np.array([-1] * n)
        j = m - 1
        for i in range(n - 1, -1, -1):
            while j >= self.min_j[i]:
                max_i[j] = i
                j -= 1
        return min_i, max_i

    def new(self, dok=None):
        out = self.__new__(self.__class__)
        out.shape = self.shape
        out.min_j = self.min_j
        out.max_j = self.max_j
        if dok is None:
            out.dok = dok_array(self.shape)
        else:
            assert dok.shape == self.shape
            out.dok = dok
        return out

    # def old_init(self, n: int, m: int):
    #     self.shape = n, m
    #     self.dok = dok_array((n, m))

    def __getitem__(self, slice):
        return self.dok[slice]

    def __setitem__(self, slice, value):
        self.dok[slice] = value

    def __mul__(self, other):
        n, m = self.shape
        if isinstance(other, CS_Matrix):
            other = other.dok
        assert other.shape in [(n, m), (n, 1), (1, m)]
        dok = self.dok.multiply(other).asformat('dok')
        return self.new(dok)

    def eye(self):
        out = self.new()
        out[np.arange(self.shape[0]), out.min_j] = 1
        return out

    def sum(self, axis=None):
        return self.dok.sum(axis=axis)

    def max(self, axis=None) -> FloatArray:
        return self.dok.asformat('coo').max(axis=axis).toarray().ravel()

    def xlog2x(self):
        dok = self.dok
        out_dok = dok.copy()
        mask = dok.nonzero()
        masked = dok[mask].toarray()
        out_dok[mask] = masked * np.log2(masked)
        return self.new(out_dok)

    def __sub__(self, other):  # Subtraction
        return self.new(self.dok - other.dok)

    @classmethod
    def from_sizes(cls, S_X: IntArray, c: float):
        S_Y = np.unique(S_X)
        min_j, max_j = bounds_Y_given_X(S_X, S_Y, c)
        return cls(min_j, max_j), S_Y


def bounds_Y_given_X(S_X: IntArray, S_Y: IntArray, c: float):
    '''
    i can only be padded to j in range(min_j[i], max_j[i]+1)
    min_j[i] is not always i due to repeated values in S_X
    '''
    n = len(S_X)
    m = len(S_Y)
    min_j = np.array([-1] * n)
    j = 0
    for i in range(n):
        while j < m and S_X[i] > S_Y[j]:
            j += 1
        min_j[i] = j

    max_j = np.array([-1] * n)
    j = 0
    for i in range(n):
        while j < m and S_Y[j] <= c * S_X[i]:
            j += 1
        max_j[i] = j - 1
    return min_j, max_j


def test_bounds(S_X: IntArray, c):
    M, S_Y = CS_Matrix.from_sizes(S_X, c)
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            can1 = (S_X[i] <= S_Y[j] <= c * S_X[i])
            can2 = M.min_i[j] <= i <= M.max_i[j]
            can3 = M.min_j[i] <= j <= M.max_j[i]
            if not can1 == can2 == can3:
                print('ERROR')
                print(can1, can2, can3)
                print(f'{S_X[i]} <= {S_Y[j]} <= {c * S_X[i]} (c={c})')
                print(M.min_i[j], i, j, M.max_j[i])
                print(S_X[M.min_i[j]], S_X[i], S_Y[j], S_X[M.max_j[i]])
                sys.exit(1)
    return


# Solvers


def PRP_Renyi_only(M: CS_Matrix, P_X: FloatArray):
    n, m = M.shape
    min_i, max_i = M.min_i, M.max_i
    min_j, max_j = M.min_j, M.max_j

    B_X = P_X.copy()  # budget for each X

    P_XY = M.new()
    for j in tqdm(range(m - 1, -1, -1)):
        ii = max_i[j]
        while ii > 0 and min_j[ii] == j:
            ii -= 1
        greedy = max(B_X[ii:max_i[j] + 1])
        if greedy == 0:
            continue
        for i in range(min_i[j], max_i[j] + 1):
            P_XY[i, j] = min(greedy, B_X[i])
            B_X[i] -= P_XY[i, j]

    P_Y_given_X = P_XY * (1 / P_X[:, None])
    return P_Y_given_X, locals()


def PRP_Renyi_Bandwidth(M: CS_Matrix, P_X: FloatArray, pre=None):
    # Improve bandwidth
    n, m = M.shape
    P_Y_given_X, scope = pre or PRP_Renyi_only(M, P_X)
    P_XY: CS_Matrix = scope['P_XY']
    min_j, max_j = M.min_j, M.max_j
    min_i, max_i = M.min_i, M.max_i

    P_XY_max = P_XY.max(axis=0)

    # Find pinned coordinates
    argmax_i = {
        j: min_i[j] +
        np.argmax([P_XY[i, j] for i in range(min_i[j], max_i[j] + 1)])
        for j in tqdm(range(m))
        if P_XY_max[j] > 0
    }
    pinned = {i: [] for i in range(n)}
    for j, i in argmax_i.items():
        pinned[i].append(j)

    # Pin them
    out_P_XY = M.new()
    for i in pinned:
        for j in pinned[i]:
            out_P_XY[i, j] = P_XY[i, j]

    # Greedily assign the rest favoring the leftmost
    for i in tqdm(range(n)):
        budget = P_X[i] - sum(P_XY[i, j] for j in pinned[i])
        for j in range(min_j[i], max_j[i] + 1):
            if np.allclose(budget, 0):
                break
            if j in pinned[i]:
                continue
            db = min(budget, P_XY_max[j])
            out_P_XY[i, j] = db
            budget -= db
        assert np.allclose(budget, 0)

    P_XY = out_P_XY
    P_Y_given_X = P_XY * (1 / P_X[:, None])
    return P_Y_given_X, locals()


def POP_Renyi_only(M: CS_Matrix, P_X: FloatArray):
    # Assume S_X are sorted
    n, m = M.shape
    min_i, max_i = M.min_i, M.max_i
    min_j, max_j = M.min_j, M.max_j

    @lru_cache(maxsize=None)
    def f(LO: int, HI: int) -> Tuple[float, int, int, int]:
        progress.update(1)
        if (progress.total + 1) % 1000 == 0:
            sys.stderr.flush()
        assert LO <= HI
        if LO == HI:
            return (0, n + 1, LO, HI)
        i_max = LO + np.argmax(P_X[LO:HI])
        ANS = (float('inf'), n + 1, n + 1, n + 1)
        for j_max in range(min_j[i_max], max_j[i_max] + 1):
            lo = max(LO, min_i[j_max])
            hi = min(HI, max_i[j_max] + 1)
            assert LO <= lo <= i_max <= hi <= HI
            renyi = f(LO, lo)[0] + P_X[i_max] + f(hi, HI)[0]
            ans = (renyi, j_max, lo, hi)
            ANS = min(ANS, ans)
        return ANS

    with tqdm(total=M.total_entries()) as progress:
        f(0, n)

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
    del f

    # Compute P_Y_given_X and P_XY

    P_Y_given_X = M.new()
    P_Y_given_X[np.arange(n), Y_given_X] = 1

    return P_Y_given_X, locals()


def POP_Renyi_Bandwidth(M: CS_Matrix, P_X: FloatArray, pre=None):
    P_Y_given_X, scope = pre or POP_Renyi_only(M, P_X)
    n, m = M.shape
    Y_given_X = scope['Y_given_X']

    P_XY_max = (P_Y_given_X * P_X[:, None]).max(axis=0)

    Y_given_X = np.array([
        next(
            (j for j in range(i, Y_given_X[i]) if P_X[i] <= P_XY_max[j]),
            Y_given_X[i],
        ) for i in tqdm(range(n))
    ])
    P_Y_given_X = M.new()
    P_Y_given_X[np.arange(n), Y_given_X] = 1
    return P_Y_given_X, locals()


def POP_Renyi_Shannon(M: CS_Matrix, P_X: FloatArray, pre=None):
    P_Y_given_X, scope = pre or POP_Renyi_only(M, P_X)
    n, m = M.shape
    min_j, max_j = M.min_j, M.max_j
    min_i, max_i = M.min_i, M.max_i
    P_XY_max = (P_Y_given_X * P_X[:, None]).max(axis=0)

    poss = [  # Possibilities of j given i
        [j
         for j in range(min_j[i], max_j[i] + 1)
         if P_X[i] <= P_XY_max[j]]
        for i in tqdm(range(n))
    ]

    @lru_cache(maxsize=None)
    def f(LO: int, HI: int) -> Tuple[float, int, int, int]:
        '''
        Optimal assignment of the elements i such that
            - poss[i] is a subset of [LO, HI)
            - (as a consequence, also) i in [LO, HI)
        Divide and conquer strategy:
            As many elements as possible are assigned to a single j,
            and for the remaining elements, recursion is used.
            Of course, always guaranteeing the initial Renyi leakage.
        Returns:
            - shannon: shannon leakage of the assignment (to be minimized)
            - indices j, LO, HI for reconstructing the channel
        '''
        progress.update(1)
        if (progress.total + 1) % 1000 == 0:
            sys.stderr.flush()
        assert LO <= HI
        if LO == HI:
            return (0, -1, LO, HI)
        elems = [
            i for i in range(min_i[LO], max_i[HI - 1])
            if LO <= poss[i][0] <= poss[i][-1] < HI
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

    with tqdm(total=M.total_entries()) as progress:
        f(0, m)

    # Channel reconstruction
    Y_given_X = np.array([-1] * n)
    Q: List[Tuple[int, int]] = [(0, n)]
    while Q:
        LO, HI = Q.pop()
        _, j, lo, hi = f(LO, HI)
        if j == -1:
            continue
        elems = [
            i for i in range(min_i[LO], max_i[HI - 1])
            if LO <= poss[i][0] <= poss[i][-1] < HI
        ]
        captured = [i for i in elems if poss[i][0] <= j <= poss[i][-1]]
        Y_given_X[captured] = j
        if LO < lo:
            Q.append((LO, lo))
        if hi < HI:
            Q.append((hi, HI))

    P_Y_given_X = M.new()
    P_Y_given_X[np.arange(n), Y_given_X] = 1
    return P_Y_given_X, locals()


def POP_Shannon_only(M: CS_Matrix, P_X: FloatArray):
    '''
    DP optimal solution to the object padding problem.

    Explanation:
        The objective is F[0].
        F[i] = (info, next_i, j) means that the optimal solution for the
        subproblem whose input is restricted to objects [i..n)
        has shannon information "info" and the first group in the
        partition is [i..next_i) and is padded to column j
    
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
    n, m = M.shape
    min_j, max_j = M.min_j, M.max_j
    min_i, max_i = M.min_i, M.max_i

    xlogx = lambda x: x and (x * np.log2(x))

    # Solution computation
    F: List[Tuple[float, int, int]]
    F = [(0, 0, 0) for _ in range(n + 1)]
    F[n] = (0, n, m)
    for i in tqdm(range(n - 1, -1, -1)):
        F[i] = (float('inf'), 0, 0)
        for j in range(min_j[i], max_j[i] + 1):
            next_i = max_i[j] + 1
            info = F[next_i][0] - xlogx(sum(P_X[i:next_i]))
            F[i] = min(F[i], (info, next_i, j))

    # Solution reconstruction
    Y_given_X = np.zeros(n, dtype=int)
    i = 0
    while i != n:
        _, next_i, j = F[i]
        assert next_i > i
        Y_given_X[i:next_i] = j
        i = next_i

    P_Y_given_X = M.new()
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

    assert np.allclose(np.sum(P_X), 1)
    print('Running...')
    start = time.time()

    M, S_Y = CS_Matrix.from_sizes(S_X, c)
    P_Y_given_X: CS_Matrix
    if c == 1.0:
        P_Y_given_X = M.eye()
        scope = {}
    else:
        P_Y_given_X, scope = solver(M, P_X, **kwargs)
    end = time.time()
    elapsed = end - start

    print('Verifying solution...')
    assert np.allclose(P_Y_given_X.sum(axis=1), 1)

    print('Computing join matrix...')
    P_X = np.array(P_X)
    P_XY = P_Y_given_X * P_X[:, None]

    def leakages():
        P_Y = P_XY.sum(axis=0)
        Q_Y = P_XY.max(axis=0)
        H_X_inf = -np.log2(np.max(P_X))
        H_Y_given_X_inf = -np.log2(np.sum(Q_Y))
        H_Y = -xlog2x_array(P_Y).sum()
        H_Y_given_X = -np.dot(P_X, P_Y_given_X.xlog2x().sum(axis=1))
        renyi = H_X_inf - H_Y_given_X_inf
        shannon = H_Y - H_Y_given_X
        return renyi, shannon

    def xlog2x_array(x: FloatArray):
        out = x.copy()
        mask = x.nonzero()
        out[mask] = x[mask] * np.log2(x[mask])
        return out

    def bandwidth_factor():
        in_size = P_XY * S_X[:, None]
        out_size = P_XY * S_Y[None, :]
        used_Bandwidth = (out_size - in_size).sum()
        min_Bandwidth = np.dot(P_X, S_X)
        return used_Bandwidth / min_Bandwidth

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


def inspect_data():
    S_X, P_X = nodeJS()
    c = 1.1
    print('testing bound function on subset...')
    test_bounds(sub_dataset(S_X, P_X, 1000)[0], c)
    print('Test passed')
    print(len(S_X))
    for c in [1, 1.05, 1.3]:
        start = time.time()
        M, S_Y = CS_Matrix.from_sizes(S_X, c)
        end = time.time()
        print(f'c={c}. Took {(end-start):.2f} seconds.')
        delta = M.max_j - M.min_j + 1  # type:ignore
        print(f'rows have {delta.mean():.2f} elems on avg')
        print(f'the total number of elements is {delta.sum()}')
        print(f'row {delta.argmax()} has {delta.max()} elems')
        print(f'weighted number of elems: {(delta*P_X).sum():.2f}')
    # import matplotlib.pyplot as plt
    # counts, bin_edges = np.histogram(np.log2(S_X), bins=31, weights=P_X)
    # centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    # plt.bar(centers, counts)
    # #P_X
    # plt.show()
    main(*sub_dataset(*nodeJS(), 1000))
    return


def sub_dataset(S_X: IntArray, P_X: FloatArray, n):
    idx = np.argsort(P_X)[-n:]
    S_X = S_X[idx]
    P_X = P_X[idx]
    P_X /= np.sum(P_X)
    idx = np.argsort(S_X)
    S_X = S_X[idx]
    P_X = P_X[idx]
    return S_X, P_X


def main(S_X: IntArray, P_X: FloatArray):
    assert np.all(np.diff(S_X) >= 0)
    solvers = [
        ('PRP_Renyi_only', PRP_Renyi_only, None),
        ('POP_Renyi_only', POP_Renyi_only, None),
        ('POP_Shannon_only', POP_Shannon_only, None),
        ('PRP_Renyi_Bandwidth', PRP_Renyi_Bandwidth, 'PRP_Renyi_only'),
        ('POP_Renyi_Bandwidth', POP_Renyi_Bandwidth, 'POP_Renyi_only'),
        ('POP_Renyi_Shannon', POP_Renyi_Shannon, 'POP_Renyi_only'),
    ]
    # Independent first order:
    solvers.sort(key=lambda x: x[2] != None)

    with open(f'paper-cases/.{len(S_X)}.txt', 'a') as f:
        for c in [1, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
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
    #main(*nodeJS())
    main(*sub_dataset(*nodeJS(), 1000))
    #inspect_data()