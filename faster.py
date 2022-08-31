"""
Reimplementation of all the algorithms using n x m sparse matrices.

To run from command line, just

    python this_file.py COMMAND

COMMAND should be one of:
        large_all:                 run all algorithms      in nodeJS dataset
        medium_all:                run all algorithms      in subset of nodeJS dataset
        large_PopReBa:             run PopReBa in nodeJS dataset
        large_PopRe:               run PopRe      in nodeJS dataset
        medium_PopSh:              run PopSh    in subset of nodeJS dataset

        etc. (see the dictionary called "the_solvers" for all solvers, and function cli for more commands)
"""
from functools import lru_cache, wraps
from pathlib import Path
from socket import timeout
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Union
import numpy as np
from typing import Tuple, Dict
from scipy.sparse import dok_array
from tqdm import tqdm as _tqdm
import pandas as pd
import numba
import numba.typed
# from numba import typed
# #from numba.experimental import jitclass
import awkward as ak


IntArray = np.ndarray  # Just for reference
FloatArray = np.ndarray  # Just for reference

cwd = Path(__file__).parent if __file__ else Path.cwd()

sys.setrecursionlimit(10**9)


@wraps(_tqdm)
def tqdm(*args, **kwargs):
    kwargs['ascii'] = True
    return _tqdm(*args, **kwargs)


# Sparse matrix

# numba_key_type = numba.types.UniTuple(numba.types.int64, 2)

# spec = [
#     ('d', numba.types.DictType(numba_key_type, numba.float64)),
#     ('n', numba.int32),
#     ('m', numba.int32),
#     ('row', numba.types.ListType(numba.types.ListType(numba.types.int32))),
#     ('col', numba.types.ListType(numba.types.ListType(numba.types.int32))),
# ]

# @numba.njit
# def jit_row_col(n:int, m:int, d:Cache):
#     row = np.array(n)
#     col = np.array(m)
#     #row = [[] for i in range(n)]
#     #col = [[] for j in range(m)]
#     for (i,j), x in d.items():
#         d[(i,j)] = x
#         row[i].append(j)
#         col[j].append(i)
#     for i in range(n):
#         row[i].sort()
#     for j in range(m):
#         col[j].sort()
#     return row, col

# class JitDok:

#     def __init__(self,n,m):
#         self.n = n
#         self.m = m
#         d:Cache = typed.Dict.empty(numba_key_type, numba.float64) # type: ignore
#         self.row, self.col = jit_row_col(n,m,d)


# aux = JitDok(5, 5)
# sys.exit(0)


Cache = Dict[Tuple[int,int], float]

class CS_Matrix:

    def __init__(self, min_j: IntArray, max_j: IntArray):
        # Safety checks
        assert np.all(min_j <= max_j)
        assert np.all(np.diff(min_j) >= 0)
        assert np.all(np.diff(max_j) >= 0)
        n = len(min_j)
        m = max_j[-1] + 1
        assert min_j[0] <= 0 <= max_j[0]
        assert min_j[n - 1] <= m - 1 <= max_j[n - 1]

        self.n = n
        self.m = m
        self.min_j, self.max_j = min_j, max_j
        self.min_i, self.max_i = self.inverse_bounds()
        self.S_X = None
        self.S_Y = None

        self.d: Cache = {}
        self._d_keys = np.array((0,2), dtype=np.int32)
        #self._d_values = np.array((0,), dtype=np.float64)
        self.row:List[List[int]] = ak.Array([[] for i in range(n)])
        self.col:List[List[int]] = ak.Array([[] for j in range(m)])

    def new(self, d:Union[None, Cache]=None):
        'copy of self with new data (zeros if None)'
        out = self.__new__(self.__class__)
        out.n = self.n
        out.m = self.m
        out.min_j = self.min_j
        out.max_j = self.max_j
        out.S_X = self.S_X
        out.S_Y = self.S_Y
        if d is None:
            d = {}
        out.d = d
        with PrintStartEnd('sort', 1):
            out._d_keys = sorted(d.keys())
        with PrintStartEnd('row-cols', 1):
            row, col = self.compute_row_cols(self.n, self.m, out._d_keys)
            out.row = ak.Array(row)
            out.col = ak.Array(col)
        return out

    @staticmethod
    def compute_row_cols(n:int, m:int, d_keys):
        row = [[] for i in range(n)]
        col = [[] for j in range(m)]
        for idx in range(len(d_keys)):
            (i,j) = d_keys[idx]
            row[i].append(j)
            col[j].append(i)
        return row, col


    def total_entries(self):
        return np.sum(self.max_j - self.min_j + 1)  # type: ignore

    def inverse_bounds(self):
        'j can only be output of i in [min_i[j]..max_i[j]]'
        n, m = self.n, self.m
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


    def __mul__(self, other):
        n, m = self.n, self.m
        other_shape = tuple(other.shape)
        assert other_shape in [(n, m), (n, 1), (1, m)], other_shape
        if isinstance(other, self.__class__):
            out = self.multiply_n_m_dicts(self.d, other.d)
        elif other_shape == (n,m):
            out = self.multiply_n_m(self.d, other)
        elif other_shape == (n,1):
            out = self.multiply_n_1(self.d, other.reshape(n))
        elif other_shape == (1,m):
            out = self.multiply_1_m(self.d, other.reshape(m))
        else:
            raise NotImplementedError(other_shape)
        return self.new(out)

    @staticmethod
    @numba.njit
    def multiply_1_m(d, vector):
        return {(i,j):vector[j]*x for (i,j), x in d.items() if vector[j]!=0.0}

    @staticmethod
    @numba.njit
    def multiply_n_1(d, vector):
        return {(i,j):vector[i]*x for (i,j), x in d.items() if vector[i]!=0.0}

    @staticmethod
    @numba.njit
    def multiply_n_m(d, mat):
        return {(i,j):mat[i,j]*x for (i,j), x in d.items() if mat[i,j]!=0.0}

    @staticmethod
    @numba.njit
    def multiply_n_m_dicts(d, dd):
        return {(i,j):dd[(i,j)]*x for (i,j), x in d.items() if dd[(i,j)]!=0.0}


    def sum(self, axis=None) -> FloatArray:
        n, m = self.n, self.m
        if axis==0:
            return self.sum_vertically(m, self.d, self.col)
        elif axis==1:
            return self.sum_horizontally(n, self.d, self.row)
        assert axis==None
        return sum(self.d.values()) # type:ignore

    @staticmethod
    @numba.njit
    def sum_vertically(m, d, col):
        return np.array([sum([d[(i,j)] for i in col[j]]) for j in range(m)])
    @staticmethod
    @numba.njit
    def sum_horizontally(n, d, row):
        return np.array([sum([d[(i,j)] for j in row[i]]) for i in range(n)])

    @staticmethod
    @numba.njit
    def max_vertically(m, d, col):
        return np.array([max([d[(i,j)] for i in col[j]]) if len(col[j])>0 else 0.0 for j in range(m)])
    @staticmethod
    @numba.njit
    def max_horizontally(n, d, row):
        return np.array([max([d[(i,j)] for j in row[i]]) if len(row[i])>0 else 0.0 for i in range(n)])

    def max(self, axis=None):
        n, m = self.n, self.m
        if axis==0:
            return self.max_vertically(m, self.d, self.col)
        elif axis==1:
            return self.max_horizontally(n, self.d, self.row)
        raise NotImplementedError

    def xlog2x(self):
        out = self.jit_xlog2x(self.d)
        return self.new(out)

    @staticmethod
    @numba.njit
    def jit_xlog2x(d):
        return {(i,j): x*np.log2(x) for (i,j),x in d.items() if x>0}

    def __sub__(self, other):  # Subtraction
        n, m = self.n, self.m
        out = self.d.copy()
        for (i,j), x in other.d.items():
            out[(i,j)] = out.get((i,j), 0) - x
        return self.new(out)

    def check(self):
        n, m = self.n, self.m
        for (i,j), x in self.d.items():
            i, j = i, j
            if x==0:
                continue
            assert self.min_j[i] <= j <= self.max_j[i], (self.min_j[i], j,
                                                         self.max_j[i])
        return

    def deterministic(self, Y_given_X:IntArray):
        n, m = self.n, self.m
        @numba.njit
        def jit(Y_given_X):
            # out:Cache = numba.typed.Dict.empty(
            #     key_type=numba.types.UniTuple(numba.types.int64, 2),
            #     value_type=numba.float64,
            # )
            out = {}
            for i in range(n):
                out[(i, Y_given_X[i])] = 1.0
            return out
        out = jit(Y_given_X)
        return self.new(out)

    def allclose(self, other):
        diff = self - other
        diff_arr = np.array(diff.values(), dtype=float)
        return np.allclose(diff_arr, 0)

    def to_array(self):
        n, m = self.n, self.m
        mat = np.zeros((n,m))
        for (i,j), x in self.d.items():
            i, j = i, j
            mat[i,j] = x
        return mat

    def __setitem__(self, slice, value):
        raise NotImplementedError

    def __getitem__(self, slice):
        raise NotImplementedError

    @classmethod
    def from_sizes(cls, S_X: IntArray, c: float):
        S_Y = np.unique(S_X)
        min_j, max_j = bounds_Y_given_X(S_X, S_Y, c)
        M = cls(min_j, max_j)
        M.S_X = S_X
        M.S_Y = S_Y
        return M, S_Y



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
    n, m = M.n, M.m
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


from gui import VisualizerPath, new_visualizer
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

PLT_RC_PARAMS: Dict = plt.rcParams  # type:ignore


class ExamplePlot:

    def __init__(self, S_X: IntArray, P_X: FloatArray, c: float):
        self.S_X = S_X
        self.P_X = P_X
        self.c = c

    def plot(self, P_Y_given_X: CS_Matrix, measurements,
             ax: Optional[plt.Axes] = None, save: Optional[Path] = None,
             tight=True, printer=None, title=None):
        mat_Y_given_X = P_Y_given_X.to_array()
        S_X = self.S_X
        P_X = self.P_X
        if printer:
            printer(f'{measurements["name"]}. {len(S_X)} objects')
            printer(f'c: {self.c}')
            printer(f'Sizes: {self.S_X}')
            printer(f'Freqs: {self.P_X}')
            printer(f'Renyi: {measurements["renyi"]}')
            printer(f'Shannon: {measurements["shannon"]}')
            printer(f'Elapsed: {measurements["elapsed"]}')
            printer(f'Bandwidth: {measurements["bandwidth"]}')
            printer(f'Matrix:')
            printer(mat_Y_given_X)
        # P_Y = np.matmul(P_X, mat_Y_given_X) # This shows the sum
        P_Y = np.max(P_X[:, None] * mat_Y_given_X, axis=0)  # This shows the max
        n = len(P_X)

        prev = PLT_RC_PARAMS['font.size']
        if tight:
            PLT_RC_PARAMS.update({'font.size': 20})

        _ax = ax
        ax = plt.gca() if ax is None else ax

        min_X = min(S_X)
        max_X = max(S_X)
        d_X = (max_X - min_X) * 0.12
        min_X -= d_X
        max_X += d_X
        ax.set_xlim([min_X, max_X])

        ax.plot([0, max_X], [1, 1], color='black', alpha=0.5)
        ax.plot([0, max_X], [-1, -1], color='black', alpha=0.5)
        ax.scatter(S_X, [+1] * n, color='black', alpha=0.5)
        ax.scatter(S_X, [-1] * n, color='black', alpha=0.5)
        width = 5+max(2, min(4, 5 / d_X))
        ax.bar(S_X, -P_Y / P_Y.max() * 0.45, color='tab:red', bottom=-1,
               width=width, alpha=0.8)
        ax.bar(S_X, P_X / P_X.max() * 0.45, color='tab:blue', bottom=1,
               width=width, alpha=0.8)
        for i in range(n):
            for j in range(n):
                strength = mat_Y_given_X[i, j]
                if strength > 0:
                    arrowprops = dict(arrowstyle="->", alpha=strength,
                                      color='tab:green')
                    ax.annotate("", xytext=(S_X[i], 1), xy=(S_X[j], -1),
                                arrowprops=arrowprops)
                    ax.annotate("", xytext=(S_X[i], 1),
                                xy=((S_X[i] + S_X[j]) / 2, 0),
                                arrowprops=arrowprops)
        if tight:
            ax.set_ylabel('Padded' + ' ' * 10 + 'Original')
        else:
            ax.set_ylabel('Padded' + ' ' * 39 + 'Original')
        ax.set_xlabel('Object size')
        ax.set_ylim([-1.5, 1.5])
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelright=False)
        if title is not None:
            ax.set_title(title)

        # if tight:
        #     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        #         label.set_fontsize(22)
        if save:
            with TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir) / save.name
                plt.savefig(tmp, bbox_inches='tight')
                plt.close()
                tmp.replace(save)
        elif _ax is None:
            plt.show()
        PLT_RC_PARAMS.update({'font.size': prev})
        return


# Solvers

class PrintStartEnd:
    def __init__(self, msg: str, max_elapsed=-1):
        'if max_elapsed is -1, prints start and end'
        self.msg = msg
        self.max_elapsed = max_elapsed
    def __enter__(self):
        if self.max_elapsed<0:
            print(f'{time.strftime("%X %x %Z")} Started ({self.msg})')
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start
        if (self.max_elapsed<0) or elapsed>=self.max_elapsed:
            space_fix = f'({self.msg})'
            print(f'{time.strftime("%X %x %Z")} Ended   {space_fix:20} took {timedelta(seconds=elapsed)}')


def PrpRe(M: CS_Matrix, P_X: FloatArray):
    n, m = M.n, M.m
    min_i, max_i = M.min_i, M.max_i
    min_j, max_j = M.min_j, M.max_j

    B_X = P_X.copy()  # budget for each X

    @numba.njit
    def jit1(B_X:FloatArray, min_i:IntArray, max_i:IntArray, min_j:IntArray):
        P_XY = {}
        for j in range(m - 1, -1, -1):
            mid_i = max_i[j]
            while mid_i > 0 and min_j[mid_i - 1] == j:
                mid_i -= 1
            greedy = max(B_X[mid_i:max_i[j] + 1])
            if greedy == 0.:
                continue
            for i in range(max_i[j], min_i[j] - 1, -1):
                P_XY[(i, j)] = min(greedy, B_X[i])
                B_X[i] -= P_XY[(i, j)]
        return P_XY

    with PrintStartEnd('PrpRe-algor'):
        d_XY = jit1(B_X, min_i, max_i, min_j)

    with PrintStartEnd('PrpRe-build'):
        P_XY = M.new(d_XY)
        P_Y_given_X = P_XY * (1 / P_X[:, None])

    return P_Y_given_X, locals()


def PrpReBa(M: CS_Matrix, P_X: FloatArray, pre=None):
    # Improve bandwidth
    n, m = M.n, M.m
    P_Y_given_X, scope = pre or PrpRe(M, P_X)
    old_P_XY: CS_Matrix = scope['P_XY']

    P_XY_max = old_P_XY.max(axis=0)
    old_d_XY = old_P_XY.d
    min_j, max_j = M.min_j, M.max_j
    min_i, max_i = M.min_i, M.max_i

    # argmax_i = {
    #     j: min_i[j] +
    #     np.argmax([old_d_XY[(i, j)] for i in range(min_i[j], max_i[j] + 1)])
    #     for j in tqdm(range(m))
    #     if P_XY_max[j] > 0
    # }
    @numba.njit
    def jit1(old_d_XY: Cache):
        # Find pinned coordinates
        INF = 1e30
        tups = []
        for j in range(m):
            if P_XY_max[j] > 0:
                ANS, ANS_i = -INF, min_i[j]
                for i in range(min_i[j], max_i[j] + 1):
                    assert (i,j) in old_d_XY
                    ans = old_d_XY[(i, j)]
                    if ans > ANS:
                        ANS, ANS_i = ans, i
                tups.append((ANS_i, j))
        return tups
    with PrintStartEnd('PrpReBa-jit1'):
        tups = jit1(old_d_XY)
        tups = np.array(tups, dtype=int)


    with PrintStartEnd('PrpReBa-nojit'):
        pinned = [[] for i in range(n)]
        # Pin them
        for i,j in tups:
            pinned[i].append(j)
        pinned = ak.Array(pinned)

    @numba.njit
    def jit2(old_d_XY: Cache, tups):
        return {(i,j): old_d_XY[(i, j)] for i,j in tups}

    # d_XY:Cache = numba.typed.Dict.empty(
    #     key_type=numba.types.UniTuple(numba.types.int64, 2),
    #     value_type=numba.float64,
    # )
    with PrintStartEnd('PrpReBa-jit2'):
        d_XY = jit2(old_d_XY, tups)

    @numba.njit
    def jit3(d_XY: Cache, pinned,P_X, P_XY_max):
        # Greedily assign the rest favoring the leftmost
        for i in range(n):
            budget = P_X[i]
            for j in pinned[i]:
                budget -= d_XY[(i, j)]
            for j in range(min_j[i], max_j[i] + 1):
                # if np.allclose(budget, 0, rtol=1e-12,atol=1e-12, equal_nan=True):
                #     break
                if budget < 1e-15:
                    break
                if j in pinned[i]:
                    continue
                db = min(budget, P_XY_max[j])
                d_XY[(i, j)] = db
                budget -= db
            #assert np.allclose(budget, 0)
        return d_XY

    with PrintStartEnd('PrpReBa-jit3'):
        d_XY = jit3(d_XY, pinned, P_X, P_XY_max)

    with PrintStartEnd('PrpReBa-new'):
        P_XY = M.new(d_XY)
    with PrintStartEnd('PrpReBa-product'):
        P_Y_given_X = P_XY * (1 / P_X[:, None])
    return P_Y_given_X, locals()


def PopRe(M: CS_Matrix, P_X: FloatArray):
    # Assume S_X are sorted
    n, m = M.n, M.m
    min_i, max_i = M.min_i, M.max_i
    min_j, max_j = M.min_j, M.max_j

    #cache: Dict[Tuple[int,int], Tuple[float, int]] = {}

    numba_key_type = numba.types.UniTuple(numba.types.int64, 2)

    cache: Dict[int, float]
    cache = numba.typed.Dict.empty(key_type=numba.int64, value_type=numba.float64) # type: ignore

    @numba.njit
    def iterative_DP():
        INF = 1e30
        cache, cache_j = {}, {}
        call = [(0,n)]
        while call:
            LO, HI = call[-1]
            if (LO,HI) in cache:
                call.pop()
                continue
            assert LO <= HI
            if LO == HI:
                cache[(LO,HI)] = 0.
                call.pop()
                continue
            i_max = LO + np.argmax(P_X[LO:HI])
            # ANS = float('inf')
            ANS, ANS_j = INF, -1
            deps = []
            for j_max in range(min_j[i_max], max_j[i_max] + 1):
                lo = max(LO, min_i[j_max])
                hi = min(HI, max_i[j_max] + 1)
                assert LO <= lo <= i_max <= hi <= HI
                if (LO,lo) not in cache:
                    deps.append((LO,lo))
                if (hi,HI) not in cache:
                    deps.append((hi,HI))
                if not deps:
                    ans = cache[(LO, lo)] + P_X[i_max] + cache[(hi, HI)]
                    if ans < ANS:
                        ANS, ANS_j = ans, j_max
            if not deps:
                cache[(LO,HI)], cache_j[(LO,HI)] = ANS, ANS_j
                call.pop()
            else:
                call.extend(deps)
        return cache, cache_j

    with PrintStartEnd('PopRe-DP'):
        cache, cache_j = iterative_DP()
    with PrintStartEnd('PopRe-post'):
        # Reconstruction
        Y_given_X = np.zeros(n, dtype=int)
        Q: List[Tuple[int, int]] = [(0, n)]
        while Q:
            LO, HI = Q.pop()
            j = cache_j[(LO, HI)]
            lo = max(LO, min_i[j])
            hi = min(HI, max_i[j] + 1)
            Y_given_X[lo:hi] = j
            if LO < lo:
                Q.append((LO, lo))
            if hi < HI:
                Q.append((hi, HI))
    P_Y_given_X = M.deterministic(Y_given_X)
    return P_Y_given_X, locals()


def PopReBa(M: CS_Matrix, P_X: FloatArray, pre=None):
    P_Y_given_X, scope = pre or PopRe(M, P_X)
    n, m = M.n, M.m
    min_j = M.min_j
    Y_given_X = scope['Y_given_X']

    P_XY_max = (P_Y_given_X * P_X[:, None]).max(axis=0)

    @numba.njit
    def jit1():
        out = Y_given_X.copy()
        for i in range(n):
            for j in range(min_j[i], Y_given_X[i]):
                if P_X[i] <= P_XY_max[j]:
                    out[i] = j
        return out
    with PrintStartEnd('PopReBa-jit1'):
        Y_given_X = jit1()
    P_Y_given_X = M.deterministic(Y_given_X)
    return P_Y_given_X, locals()


def PopReSh(M: CS_Matrix, P_X: FloatArray, pre=None):
    P_Y_given_X, scope = pre or PopRe(M, P_X)
    n, m = M.n, M.m
    min_j, max_j = M.min_j, M.max_j
    min_i, max_i = M.min_i, M.max_i

    with PrintStartEnd('PopReSh-prod-max'):
        P_XY_max = (P_Y_given_X * P_X[:, None]).max(axis=0)
    with PrintStartEnd('PopReSh-poss_j'):
        poss_j = ak.Array([  # Possibilities of j given i
            [j
            for j in range(min_j[i], max_j[i] + 1)
            if P_X[i] <= P_XY_max[j]]
            for i in range(n)
        ])
        poss_j:List[List[int]] = poss_j # Mere type hinting

    @numba.njit
    def iterative_DP(P_X: FloatArray):
        INF = 1e30
        cache:Dict[Tuple[int,int],float] = {}
        cache_j:Dict[Tuple[int,int],int] = {}
        for LO in range(m, -1, -1):
            for HI in range(LO, m+1):
                if LO == HI:
                    cache[(LO,HI)] = 0.0
                    continue
                ANS, ANS_j = INF, -1
                for j in range(LO, HI):
                    # Greedy capture: all elems are mapped to j
                    captured = []
                    for i in range(min_i[j], max_i[j] + 1):
                        if poss_j[i][-1] >= HI:
                            break
                        if LO<=poss_j[i][0] <= poss_j[i][0] <= j <= poss_j[i][-1] < HI:
                            captured.append(i)
                    if not captured:
                        continue
                    # Subproblems
                    lo, hi = j, j + 1
                    s = 0.
                    for i in captured:
                        s += P_X[i]
                    mid_shannon = 0.0 if s==0.0 else -s * np.log2(s)
                    ans = cache[(LO, lo)] + mid_shannon + cache[(hi, HI)]
                    if ans < ANS:
                        ANS, ANS_j = ans, j
                if ANS_j == -1:
                    cache[(LO,HI)] = 0.0
                    continue
                cache[(LO, HI)] = ANS
                cache_j[(LO, HI)] = ANS_j
        return cache, cache_j

    with PrintStartEnd('PopReSh-DP'):
        print((m*(m-1)*(m-2))//6)
        cache, cache_j = iterative_DP(P_X)
    with PrintStartEnd('PopReSh-post'):
        Y_given_X = scope['Y_given_X'].copy() #np.array([-10] * n)
        Q: List[Tuple[int, int]] = [(0, m)]
        while Q:
            LO, HI = Q.pop()
            j = cache_j.get((LO, HI), None)
            if j == None:
                continue
            lo, hi = j, j + 1
            captured = []
            for i in range(min_i[j], max_i[j] + 1):
                if poss_j[i][-1] >= HI:
                    break
                if LO<=poss_j[i][0] <= poss_j[i][0] <= j <= poss_j[i][-1] < HI:
                    captured.append(i)
            # print(LO, HI, captured)
            Y_given_X[captured] = j
            Q.append((LO, lo))
            Q.append((hi, HI))
        assert np.all(Y_given_X>=0), (Y_given_X, cache[(0,m)])
    with PrintStartEnd('PopRe-matrix-build', 3):
        P_Y_given_X = M.deterministic(Y_given_X)
    return P_Y_given_X, locals()


def PopSh(M: CS_Matrix, P_X: FloatArray):
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
    n, m = M.n, M.m
    min_j, max_j = M.min_j, M.max_j
    min_i, max_i = M.min_i, M.max_i


    @numba.njit
    def DP(P_X: FloatArray):
        xlogx = lambda x: x and (x * np.log2(x))
        INF = 1e30
        # Solution computation
        F: List[Tuple[float, int, int]]
        F = [(0.0, 0, 0) for _ in range(n + 1)]
        F[n] = (0, n, m)
        for i in range(n - 1, -1, -1):
            F[i] = ANS = (INF, 0, 0)
            for j in range(min_j[i], max_j[i] + 1):
                next_i = max_i[j] + 1
                info = F[next_i][0] - xlogx(np.sum(P_X[i:next_i]))
                ans = (info, next_i, j)
                less_than = (ans[0] < ANS[0]) or ((ans[0] == ANS[0]) and (
                    (ans[1] < ANS[1]) or ((ans[1] == ANS[1]) and (ans[2] < ANS[2]))))
                if less_than:
                    F[i] = ANS = ans
        return F

    with PrintStartEnd('PopSh-DP'):
        F = DP(P_X)
    with PrintStartEnd('PopSh-post'):
        Y_given_X = np.zeros(n, dtype=int)
        i = 0
        while i != n:
            _, next_i, j = F[i]
            assert next_i > i
            Y_given_X[i:next_i] = j
            i = next_i

    P_Y_given_X = M.deterministic(Y_given_X)
    return P_Y_given_X, locals()


def nodeJS():
    S_X: IntArray
    P_X: FloatArray
    file = cwd / '__exec_files' / 'nodeJS.txt'
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
    'PrpRe': (PrpRe, None),
    'PopRe': (PopRe, None),
    'PopSh': (PopSh, None),
    'PrpReBa': (PrpReBa, 'PrpRe'),
    'PopReBa': (PopReBa, 'PopRe'),
    'PopReSh': (PopReSh, 'PopRe'),
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
    expected = M.deterministic(M.min_j)
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
    # if c == 1.0:
    #     P_Y_given_X = M.deterministic(M.min_j)
    #     scope = {}
    # else:
    P_Y_given_X, scope = solver(M, P_X, **kwargs)
    end = time.time()
    elapsed = end - start

    with PrintStartEnd('check'):
        print('Verifying solution...')
        h_sum = P_Y_given_X.sum(axis=1)
        assert np.allclose(h_sum, 1), (min(h_sum), max(h_sum))
        P_Y_given_X.check()

    with PrintStartEnd('joint'):
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
        'name': solver.__name__,
    }
    alg_output = P_Y_given_X, scope
    return measurements, alg_output


def bound_test_nodeJS():
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

def inspect_data():
    vpath = new_visualizer()
    for (S_X, P_X) in [sub_dataset(*nodeJS(), 100), sub_dataset(*nodeJS(), 1000), nodeJS()]:
        vpath.print(f'n={len(S_X)}')
        vpath.print(f'm={len(np.unique(S_X))}')
        vpath.print(f'mean={np.dot(S_X, P_X)}')
        vpath.print(f'median={S_X[np.searchsorted(np.cumsum(P_X), 0.5)]}')
        vpath.print(f'Number of available positions:')
        for c in [1, 1.02, 1.04, 1.06, 1.08, 1.1]:
            M, _ = CS_Matrix.from_sizes(S_X, c)
            vpath.print(f'c={c:.2f} positions={np.sum(M.max_j+1-M.min_j)}')
        counts, bin_edges = np.histogram(np.log2(S_X), bins=31, weights=P_X)
        centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        plt.bar(centers, counts)
        plt.xlabel('log2(file size)')
        plt.ylabel('weighted counts')
        vpath.print(counts.sum())
        vpath.plot_and_close(plt)
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
    n = len(S_X)
    # Filter
    if solver_name != 'all':
        solvers = {solver_name: the_solvers[solver_name]}
    else:
        solvers = {**the_solvers}
    vpath = new_visualizer()
    _df = []
    with open(cwd / '__exec_files' / f'{len(S_X)}-{solver_name}.py.txt',
              'a') as f:
        for c in [1, 1, 1.02, 1.04, 1.06, 1.08, 1.1]:
            # The first c=1 is to see the the JIT time.
            Measurements = {}
            Outputs = {}
            for name, (solver, dependency) in solvers.items():
                print('-' * 30)
                print(name, c)

                dep = dependency if dependency in Outputs else None
                # Inject pre-computed outputs:
                kwargs = {'pre': Outputs.get(dep)} if dep else {}
                # Run and measure
                metrics, output = measure(solver, S_X, P_X, c, **kwargs)
                # Fix time
                dependency = kwargs.get('pre')
                if dep:
                    prev = Measurements[dep]
                    metrics['elapsed'] += prev['elapsed']

                Outputs[name] = output
                Measurements[name] = metrics

                metrics = {'name': name, 'c': c, **metrics}
                f.write(f'{metrics}\n')
                _df.append(metrics)
                f.flush()
                print(metrics)
            if c > 1:
                df = pd.DataFrame(_df)
                for prop in ['elapsed', 'bandwidth', 'renyi', 'shannon']:
                    ax = plt.gca()
                    for name in df['name'].unique():
                        sub_df = df[df['name'] == name]
                        sub_df.plot('c', prop, label=name, ax=ax, marker='o')
                    plt.ylabel(prop)
                    plt.grid()
                    vpath.print(df[['c', prop, 'name']])
                    vpath.plot_and_close(plt)
    df = pd.DataFrame(_df)
    df['bandwidth_percent'] = df['bandwidth'] * 100
    df.to_csv(cwd / '__exec_files' / f'{n}.txt')
    for name in df['name'].unique():
        df[df['name'] == name].to_csv(cwd / '__exec_files' / f'{n}-{name}.txt',
                                      sep=' ', index=False)
    return


def generate(max_n_objects: int):
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


def correctness_tests(n_cases=5000, n_objects=10, also_brute_force=False):
    #np.random.seed(0)
    from itertools import product

    def POP_bruteforce(function):

        def minimizer(M: CS_Matrix, P_X: FloatArray):
            'Can not handle more than 10'
            n, m = M.n, M.m
            poss_j = [range(M.min_j[i], M.max_j[i] + 1) for i in range(n)]

            if function is renyi_bandwidth:
                key = lambda f: function(f, P_X, M.S_X, M.S_Y)
            else:
                key = lambda f: function(f, P_X)
            Y_given_X = min(product(*poss_j), key=key)
            Y_given_X = np.array(Y_given_X, dtype=int)
            P_Y_given_X = M.deterministic(Y_given_X)
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

    def renyi_bandwidth(f, P_X, S_X, S_Y):
        renyi, shannon = renyi_shannon(f, P_X)
        bandwidth = sum(P_X[x] * (S_Y[y]-S_X[x]) for x, y in enumerate(f))
        return (renyi, bandwidth)

    if also_brute_force:
        solvers = {
            **the_solvers,
            'BF_ReSh': (POP_bruteforce(renyi_shannon), None),
            'BF_ShRe': (POP_bruteforce(shannon_renyi), None),
            'BF_ReBa': (POP_bruteforce(renyi_bandwidth), None),
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
            ('renyi', 'PrpRe', '<=', 'PopRe'),
            ('renyi', 'PrpRe', '==', 'PrpReBa'),
            ('bandwidth', 'PrpReBa', '<=', 'PrpRe'),
            # POP. renyi leakage and bandwidth
            ('renyi', 'PopRe', '==', 'BF_ReSh'),
            ('renyi', 'PopRe', '==', 'PopReBa'),
            ('renyi', 'PopSh', '>=', 'PopRe'),
            ('renyi', 'PopSh', '==', 'BF_ShRe'),
            ('bandwidth', 'PopReBa', '<=', 'PopRe'),
            ('bandwidth', 'PopReBa', '==', 'PopRe'),
            # POP. shannon leakage
            ('shannon', 'PopReSh', '>=', 'BF_ReSh'),
            ('shannon', 'PopReSh', '<=', 'PopRe'),
            ('shannon', 'PopSh', '==', 'BF_ShRe'),
            ('shannon', 'PopSh', '<=', 'PopReSh'),
            # These are expected NOT to hold. Uncomment and run:
            # ('bandwidth', 'PopReBa', '==', 'BF_ReBa'),
            # ('bandwidth', 'PrpReBa', '==', 'BF_ReBa'),
        ]

        all_leq = lambda a, b: np.all((a <= b) | np.isclose(a, b))
        failed = {}
        for check in checks:
            prop, first, op, second = check
            if first not in measurements or second not in measurements:
                assert first.startswith('BF_')
                assert second.startswith('BF_')
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
        #         measurements['PrpRe']['renyi'],
        #         measurements['PopRe']['renyi'],
        #     ),
        #     not np.allclose(
        #         measurements['PopRe']['shannon'],
        #         measurements['PopReSh']['shannon'],
        #     ),
        #     not np.allclose(
        #         measurements['PopRe']['bandwidth'],
        #         measurements['PopReBa']['bandwidth'],
        #     ),
        #     not np.allclose(
        #         measurements['PopRe']['shannon'],
        #         measurements['PopSh']['shannon'],
        #     ),
        # ]
        # if all(interesting):
        #     print('BINGO!')
        #     print(S_X, P_X, c, interesting)
        #     sys.exit(1)
    return

def find_paper_example_PopReBa(n_objects=16, n_examples=10):
    vpath = new_visualizer('Example')
    tc = 1
    while n_examples >= 0:
        S_X, P_X, c = generate(max_n_objects=n_objects)
        # c = 1.1
        m2, (out2, _) = measure(PopRe, S_X, P_X, c)
        m3, (out3, _) = measure(PopReBa, S_X, P_X, c)
        if m2['bandwidth'] != m3['bandwidth']:
            example = ExamplePlot(S_X, P_X, c)
            print(S_X, P_X, c)
            example.plot(out2, m2, save=vpath.png(), printer=vpath.print)
            next(vpath)
            example.plot(out3, m3, save=vpath.png(), printer=vpath.print)
            next(vpath)
            n_examples -= 1
        tc += 1
    return

def find_paper_example_PopReSh(n_objects=10, n_examples=10):
    vpath = new_visualizer('Example')
    tc = 1
    while n_examples >= 0:
        S_X, P_X, c = generate(max_n_objects=n_objects)
        # c = 1.1
        m1, (out1, _) = measure(PopSh, S_X, P_X, c)
        m2, (out2, _) = measure(PopRe, S_X, P_X, c)
        m3, (out3, _) = measure(PopReSh, S_X, P_X, c)
        if m1['renyi'] != m2['renyi'] and m2['shannon'] != m3['shannon']:
            example = ExamplePlot(S_X, P_X, c)
            print(S_X, P_X, c)
            example.plot(out1, m1, save=vpath.png(), printer=vpath.print)
            next(vpath)
            example.plot(out2, m2, save=vpath.png(), printer=vpath.print)
            next(vpath)
            example.plot(out3, m3, save=vpath.png(), printer=vpath.print)
            next(vpath)
            n_examples -= 1
        tc += 1
    return

def actual_paper_example_PopReSh():
    vpath = new_visualizer('Example')
    S_X = np.array([1000, 1050, 1100, 1110, 1120, 1140])
    #P_X = np.array([0.20, 0.05, 0.21, 0.11, 0.16, 0.18])
    #P_X /= P_X.sum()
    P_X = np.array([0.22, 0.05, 0.23 , 0.12, 0.18, 0.20])
    c = 1.1
    m1, (out1, _) = measure(PopSh, S_X, P_X, c)
    m2, (out2, _) = measure(PopRe, S_X, P_X, c)
    m3, (out3, _) = measure(PopReSh, S_X, P_X, c)
    assert m1['renyi'] != m2['renyi'] and m2['shannon'] != m3['shannon']
    example = ExamplePlot(S_X, P_X, c)
    print(S_X, P_X, c)
    example.plot(out1, m1, save=vpath.png(), printer=vpath.print)
    next(vpath)
    example.plot(out2, m2, save=vpath.png(), printer=vpath.print)
    next(vpath)
    example.plot(out3, m3, save=vpath.png(), printer=vpath.print)
    next(vpath)
    return


def cli():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('command', type=str)
    args = parser.parse_args()
    command = args.command
    (cwd / '__exec_files').mkdir(exist_ok=True)
    if command.startswith('large_'):
        command = command[len('large_'):]
        assert command == 'all' or command in the_solvers, command
        main(*nodeJS(), solver_name=command)
    elif command.startswith('small_'):
        command = command[len('small_'):]
        assert command == 'all' or command in the_solvers, command
        main(*sub_dataset(*nodeJS(), 100), solver_name=command)
    elif command.startswith('medium_'):
        command = command[len('medium_'):]
        assert command == 'all' or command in the_solvers, command
        main(*sub_dataset(*nodeJS(), 1000), solver_name=command)
    elif command == 'inspect_data':
        inspect_data()
    elif command == 'correctness_tests':
        # 250 tests against bruteforce of around and at most 10 objects each.
        correctness_tests(n_cases=250, n_objects=10, also_brute_force=True)
        # 250 tests of around and at most 100 objects each.
        correctness_tests(n_cases=250, n_objects=100, also_brute_force=False)
    elif command == 'eye_tests':
        eye_tests()
    elif command == 'find_paper_example_PopReSh':
        find_paper_example_PopReSh()
        eye_tests()
    elif command == 'find_paper_example_PopReBa':
        find_paper_example_PopReBa()
    elif command == 'actual_paper_example_PopReSh':
        actual_paper_example_PopReSh()
    else:
        print(__doc__)
        raise NotImplementedError(command)


if __name__ == '__main__':
    cli()
