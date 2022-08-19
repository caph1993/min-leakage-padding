"""
Reimplementation of all the algorithms using n x m sparse matrices.

To run from command line, just

    python this_file.py COMMAND

COMMAND should be one of:
        large_all:                 run all algorithms      in nodeJS dataset
        medium_all:                run all algorithms      in subset of nodeJS dataset
        large_POP_Renyi_Bandwidth: run POP_Renyi_Bandwidth in nodeJS dataset
        large_POP_Renyi_only:      run POP_Renyi_only      in nodeJS dataset
        medium_POP_Shannon_only:   run POP_Shannon_only    in subset of nodeJS dataset
        
        etc. (see the dictionary called "the_solvers")
"""
from functools import lru_cache, wraps
from pathlib import Path
import sys
import time
from typing import List, Optional, Sequence, Union
import numpy as np
from typing import Tuple
from scipy.sparse import dok_array
from tqdm import tqdm as _tqdm

IntArray = np.ndarray  # Just for reference
FloatArray = np.ndarray  # Just for reference

cwd = Path(__file__).parent if __file__ else Path.cwd()

sys.setrecursionlimit(10**9)


@wraps(_tqdm)
def tqdm(*args, **kwargs):
    kwargs['ascii'] = True
    return _tqdm(*args, **kwargs)


# Sparse matrix


class CS_Matrix:
    '''
    Constrained sparse matrix.
    It can only be non-zero at coordinates [i, j] where
        min_j[i] <= j <= max_j[i]
    '''

    def __init__(self, min_j: IntArray, max_j: IntArray):
        # Safety checks
        assert np.all(min_j <= max_j)
        assert np.all(np.diff(min_j) >= 0)
        assert np.all(np.diff(max_j) >= 0)
        n = len(min_j)
        m = max_j[-1] + 1
        assert min_j[0] <= 0 <= max_j[0]
        assert min_j[n - 1] <= m - 1 <= max_j[n - 1]

        self.shape = n, m
        self.min_j, self.max_j = min_j, max_j
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
        'copy of self with new data (zeros if None)'
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

    def check(self):
        X, Y = self.dok.nonzero()
        for i, j in zip(X, Y):
            assert self.min_j[i] <= j <= self.max_j[i], (self.min_j[i], j,
                                                         self.max_j[i])
        return

    def eye(self):
        out = self.new()
        out[np.arange(self.shape[0]), out.min_j] = 1
        return out

    def allclose(self, other):
        diff = self - other
        X, Y = diff.dok.nonzero()
        diff_arr = np.array([diff[x, y] for x, y in zip(X, Y)], dtype=float)
        return np.allclose(diff_arr, 0)


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
        mid_i = max_i[j]
        while mid_i > 0 and min_j[mid_i - 1] == j:
            mid_i -= 1
        greedy = max(B_X[mid_i:max_i[j] + 1])
        if greedy == 0:
            continue
        for i in range(max_i[j], min_i[j] - 1, -1):
            P_XY[i, j] = min(greedy, B_X[i])
            B_X[i] -= P_XY[i, j]

    P_Y_given_X = P_XY * (1 / P_X[:, None])
    return P_Y_given_X, locals()


def PRP_Renyi_Bandwidth(M: CS_Matrix, P_X: FloatArray, pre=None):
    # Improve bandwidth
    n, m = M.shape
    P_Y_given_X, scope = pre or PRP_Renyi_only(M, P_X)
    old_P_XY: CS_Matrix = scope['P_XY']
    min_j, max_j = M.min_j, M.max_j
    min_i, max_i = M.min_i, M.max_i

    P_XY_max = old_P_XY.max(axis=0)

    # Find pinned coordinates
    argmax_i = {
        j: min_i[j] +
        np.argmax([old_P_XY[i, j] for i in range(min_i[j], max_i[j] + 1)])
        for j in tqdm(range(m))
        if P_XY_max[j] > 0
    }
    pinned = {i: [] for i in range(n)}
    for j, i in argmax_i.items():
        pinned[i].append(j)

    # Pin them
    P_XY = M.new()
    for i in pinned:
        for j in pinned[i]:
            assert old_P_XY[i, j] == P_XY_max[j]
            P_XY[i, j] = old_P_XY[i, j]

    # Greedily assign the rest favoring the leftmost
    for i in tqdm(range(n)):
        budget = P_X[i] - sum(P_XY[i, j] for j in pinned[i])
        for j in range(min_j[i], max_j[i] + 1):
            if np.allclose(budget, 0, atol=1e-15):
                break
            if j in pinned[i]:
                continue
            db = min(budget, P_XY_max[j])
            P_XY[i, j] = db
            budget -= db
        assert np.allclose(budget, 0)
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

    with tqdm(total=(m * m - m) // 2) as progress:
        f(0, n)
        sys.stderr.flush()

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
    min_j = M.min_j
    Y_given_X = scope['Y_given_X']

    P_XY_max = (P_Y_given_X * P_X[:, None]).max(axis=0)

    Y_given_X = np.array([
        next(
            (j for j in range(min_j[i], Y_given_X[i]) if P_X[i] <= P_XY_max[j]),
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

    poss_j = [  # Possibilities of j given i
        [j
         for j in range(min_j[i], max_j[i] + 1)
         if P_X[i] <= P_XY_max[j]]
        for i in tqdm(range(n))
    ]

    @lru_cache(maxsize=None)
    def f(LO: int, HI: int) -> Tuple[float, int, int, int]:
        '''
        Optimal assignment of the elements i such that
            - possibilities_Y_given_X[i] is a subset of [LO, HI)
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
            i for i in range(min_i[LO], max_i[HI - 1] + 1)
            if LO <= poss_j[i][0] <= poss_j[i][-1] < HI
        ]
        columns = sorted(set([j for i in elems for j in poss_j[i]]))
        if not columns:
            return (0, -1, LO, HI)
        ANS = (float('inf'), -1, -1, -1)
        for j in columns:
            # Greedy capture: all elems are mapped to j
            captured = [i for i in elems if poss_j[i][0] <= j <= poss_j[i][-1]]
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

    with tqdm(total=(m * m - m) // 2) as progress:
        f(0, m)
        sys.stderr.flush()

    # Channel reconstruction
    Y_given_X = np.array([-1] * n)
    Q: List[Tuple[int, int]] = [(0, m)]
    while Q:
        LO, HI = Q.pop()
        _, j, lo, hi = f(LO, HI)
        if j == -1:
            continue
        elems = [
            i for i in range(min_i[LO], max_i[HI - 1] + 1)
            if LO <= poss_j[i][0] <= poss_j[i][-1] < HI
        ]
        captured = [i for i in elems if poss_j[i][0] <= j <= poss_j[i][-1]]
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
    S_X: IntArray
    P_X: FloatArray
    file = cwd / 'paper-cases' / 'nodeJS.txt'
    if not file.exists():
        print('Creating dataset from original...')
        import pandas as pd
        df = pd.read_csv(
            './datasets/npm_no_scope_full_stats_nonzero_downloads.csv',
            names=['name', 'size', 'visits'],
        )
        df.sort_values(by='visits', ascending=False, inplace=True)
        df.sort_values(by='size', inplace=True)
        S_X = df['size'].values  # type: ignore
        P_X = (df['visits'] / df['visits'].sum()).values
        del df
        with open(file, 'w') as f:
            f.write(f'{" ".join(map(str, S_X))}\n')
            f.write(f'{" ".join(map(str, P_X))}\n')
        print(f'Saved as {file}.')
        print('Avg Bandwidth:', np.dot(S_X, P_X))

    with open(file, 'r') as f:
        S_X = np.array([int(x) for x in f.readline().split()])
        P_X = np.array([float(x) for x in f.readline().split()])
    return S_X, P_X


the_solvers = {
    'PRP_Renyi_only': (PRP_Renyi_only, None),
    'POP_Renyi_only': (POP_Renyi_only, None),
    'POP_Shannon_only': (POP_Shannon_only, None),
    'PRP_Renyi_Bandwidth': (PRP_Renyi_Bandwidth, 'PRP_Renyi_only'),
    'POP_Renyi_Bandwidth': (POP_Renyi_Bandwidth, 'POP_Renyi_only'),
    'POP_Renyi_Shannon': (POP_Renyi_Shannon, 'POP_Renyi_only'),
}
# Make sure the order matches dependencies
assert list(the_solvers.values()) == sorted(the_solvers.values(),
                                            key=lambda x: x[1] != None)


def eye_tests():
    '''
    When c=1, the optimal solution is the identity.
    More precisely, the "rectangular" identity when n!=m.
    '''
    c = 1.0
    S_X, P_X = nodeJS()
    M, S_Y = CS_Matrix.from_sizes(S_X, c)
    expected = M.eye()
    for name, (solver, _) in the_solvers.items():
        print(f'{name} c={c}')
        P_Y_given_X, scope = solver(M, P_X)
        print(f'checking output...')
        assert expected.allclose(P_Y_given_X)
        print(f'ok')
    return


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
    h_sum = P_Y_given_X.sum(axis=1)
    assert np.allclose(h_sum, 1), (min(h_sum), max(h_sum))
    P_Y_given_X.check()

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
        used_bandwidth = (out_size - in_size).sum()
        min_bandwidth = np.dot(P_X, S_X)
        return used_bandwidth / min_bandwidth

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
    for c in [1, 1.05, 1.3, 3]:
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
    # Filter top n
    idx = np.argsort(P_X)[-n:]
    S_X = S_X[idx]
    P_X = P_X[idx]
    P_X /= np.sum(P_X)
    # Sort by size
    idx = np.argsort(S_X)
    S_X = S_X[idx]
    P_X = P_X[idx]
    return S_X, P_X


def main(S_X: IntArray, P_X: FloatArray, solver_name='all'):
    assert np.all(np.diff(S_X) >= 0)

    # Filter
    if solver_name != 'all':
        solvers = {solver_name: the_solvers[solver_name]}
    else:
        solvers = {**the_solvers}

    with open(cwd / 'paper-cases' / f'{solver_name}-{len(S_X)}.txt', 'a') as f:
        for c in [1, 1.02, 1.04, 1.06, 1.08, 1.1]:
            Measurements = {}
            Outputs = {}
            for name, (solver, dependency) in solvers.items():
                print('-' * 30)
                print(name, c)

                # Inject pre-computed outputs:
                kwargs = {'pre': Outputs.get(dependency)}
                if kwargs['pre'] is None:
                    kwargs.pop('pre')
                # Run and measure
                measurements, output = measure(solver, S_X, P_X, c, **kwargs)
                # Fix time
                if kwargs:
                    prev = Measurements[dependency]
                    measurements['elapsed'] += prev['elapsed']

                Outputs[name] = output
                Measurements[name] = measurements

                measurements = {'name': name, 'c': c, **measurements}
                f.write(f'{measurements}\n')
                f.flush()
                print(measurements)
    return


def correctness_tests(n_cases=500, n_objects=10, also_brute_force=False):

    #np.random.seed(0)

    # def generate(n):
    #     # http://www.eecs.harvard.edu/~michaelm/NEWWORK/postscripts/filesize.pdf
    #     # They criticize lognormal, but it's an approximation.
    #     # Search for "change [12]" in the PDF.
    #     S_X = np.power(2, np.random.normal(15, 3, size=n))
    #     S_X = 1 + np.array(S_X, dtype=int)
    #     # Promote some repetitions
    #     S_X = np.random.choice(S_X, size=n, replace=True)
    #     S_X.sort()
    #     # I will assume a similar behavior for the probabilities.
    #     P_X = np.power(2, np.random.normal(15, 3, size=n))
    #     P_X /= P_X.sum()
    #     c = 1.1
    #     return S_X, P_X, c

    def generate(max_n_objects):
        object_size = max(100, 10 * max_n_objects)
        n = max(*(np.random.randint(1, max_n_objects) for _ in range(3)))
        S_X = np.random.choice(range(1, object_size + 1), n, replace=False)
        S_X.sort()
        c = (np.random.choice(S_X) / np.random.choice(S_X))
        c = max(c, 1 / c)
        c = round(c + 0.01 * np.random.random(), 2)
        P_X = np.random.random(n)
        P_X /= P_X.sum()
        return S_X, P_X, c

    from itertools import product

    def POP_bruteforce(function):

        def minimizer(M: CS_Matrix, P_X: FloatArray):
            'Can not handle more than 10'
            n, m = M.shape
            poss_j = [range(M.min_j[i], M.max_j[i] + 1) for i in range(n)]
            Y_given_X = min(product(*poss_j), key=lambda f: function(f, P_X))
            Y_given_X = np.array(Y_given_X, dtype=int)
            P_Y_given_X = M.new()
            P_Y_given_X[np.arange(n), Y_given_X] = 1
            return P_Y_given_X, locals()

        return minimizer

    plogp = lambda t: t * np.log2(t)

    def renyi_shannon(f, P_X):
        p = {}
        for x, y in enumerate(f):
            p[y] = p.get(y, [])
            p[y].append(P_X[x])
        renyi = sum(max(l) for l in p.values())
        shannon = -sum(plogp(sum(l)) for l in p.values())
        return renyi, shannon

    def shannon_renyi(f, P_X):
        return tuple(reversed(renyi_shannon(f, P_X)))

    if also_brute_force:
        solvers = {
            **the_solvers,
            'POP_BF_Renyi_Shannon': (POP_bruteforce(renyi_shannon), None),
            'POP_BF_Shannon_Renyi': (POP_bruteforce(shannon_renyi), None),
        }
    else:
        solvers = {**the_solvers}
    for _ in range(n_cases):
        S_X, P_X, c = generate(n_objects)
        test_bounds(S_X, c)

        measurements = {
            name: print('-' * 30, f'{name} {c} {len(S_X)}', sep='\n') or
            measure(solver, S_X, P_X, c)[0]
            for name, (solver, _) in solvers.items()
        }

        checks = [
            # PRP. renyi leakage and bandwidth
            ('renyi', 'PRP_Renyi_only', '<=', 'POP_Renyi_only'),
            ('renyi', 'PRP_Renyi_only', '==', 'PRP_Renyi_Bandwidth'),
            ('bandwidth', 'PRP_Renyi_Bandwidth', '<=', 'PRP_Renyi_only'),
            # POP. renyi leakage and bandwidth
            ('renyi', 'POP_Renyi_only', '==', 'POP_BF_Renyi_Shannon'),
            ('renyi', 'POP_Renyi_only', '==', 'POP_Renyi_Bandwidth'),
            ('renyi', 'POP_Shannon_only', '>=', 'POP_Renyi_only'),
            ('renyi', 'POP_Shannon_only', '==', 'POP_BF_Shannon_Renyi'),
            ('bandwidth', 'POP_Renyi_Bandwidth', '<=', 'POP_Renyi_only'),
            # POP. shannon leakage
            ('shannon', 'POP_Renyi_Shannon', '>=', 'POP_BF_Renyi_Shannon'),
            ('shannon', 'POP_Renyi_Shannon', '<=', 'POP_Renyi_only'),
            ('shannon', 'POP_Shannon_only', '==', 'POP_BF_Shannon_Renyi'),
            ('shannon', 'POP_Shannon_only', '<=', 'POP_Renyi_Shannon'),
        ]

        all_leq = lambda a, b: np.all((a <= b) | np.isclose(a, b))
        failed = {}
        for check in checks:
            prop, first, op, second = check
            if first not in measurements or second not in measurements:
                continue  # Skip BF checks.
            assert op in ['==', '<=', '>=']
            values = (measurements[first][prop], measurements[second][prop])
            if op == '==' and not np.allclose(values[0], values[1]):
                failed[check] = values
            elif op == '<=' and not all_leq(values[0], values[1]):
                failed[check] = values
            elif op == '>=' and not all_leq(values[1], values[0]):
                failed[check] = values
        if failed:
            from pprint import pprint
            pprint(measurements)
            print('=' * 30)
            print('FAILURE:')
            pprint(failed)
            print(S_X, P_X, c)
            sys.exit(1)

        # interesting = [
        #     not np.allclose(
        #         measurements['PRP_Renyi_only']['renyi'],
        #         measurements['POP_Renyi_only']['renyi'],
        #     ),
        #     not np.allclose(
        #         measurements['POP_Renyi_only']['shannon'],
        #         measurements['POP_Renyi_Shannon']['shannon'],
        #     ),
        #     not np.allclose(
        #         measurements['POP_Renyi_only']['bandwidth'],
        #         measurements['POP_Renyi_Bandwidth']['bandwidth'],
        #     ),
        #     not np.allclose(
        #         measurements['POP_Renyi_only']['shannon'],
        #         measurements['POP_Shannon_only']['shannon'],
        #     ),
        # ]
        # if all(interesting):
        #     print('BINGO!')
        #     print(S_X, P_X, c, interesting)
        #     sys.exit(1)
    return


def cli():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('command', type=str)
    args = parser.parse_args()
    command = args.command
    if command.startswith('large_'):
        command = command[len('large_'):]
        assert command == 'all' or command in the_solvers, command
        main(*nodeJS(), solver_name=command)
    elif command.startswith('medium_'):
        command = command[len('medium_'):]
        assert command == 'all' or command in the_solvers, command
        main(*sub_dataset(*nodeJS(), 1000), solver_name=command)
    elif command == 'correctness_tests':
        # 250 tests of around and at most 100 objects each.
        correctness_tests(n_cases=250, n_objects=100, also_brute_force=False)
        # 250 tests against bruteforce of around and at most 10 objects each.
        correctness_tests(n_cases=250, n_objects=10, also_brute_force=True)
    elif command == 'eye_tests':
        eye_tests()
    else:
        print(__doc__)
        raise NotImplementedError(command)


if __name__ == '__main__':
    cli()
