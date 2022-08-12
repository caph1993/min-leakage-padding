"""
Sparse reimplementation of all the algorithms
"""
from functools import wraps
import itertools
from pathlib import Path
import time
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from shared_functions import sorting_sizes, file_parser_iterator, verify_solution
from scipy.sparse import dok_array
from tqdm import tqdm


def decorated(f):

    def xlog2x_sparse(x: dok_array):
        out = x.copy()
        mask = x.nonzero()
        masked = x[mask].toarray()
        out[mask] = masked * np.log2(masked)
        return out

    def xlog2x_array(x: NDArray):
        out = x.copy()
        mask = x.nonzero()
        out[mask] = x[mask] * np.log2(x[mask])
        return out

    @wraps(f)
    def wrapper(sizes: List[int], freqs: List[float], c: float):
        assert np.allclose(np.sum(freqs), 1)
        print('Running...')
        start = time.time()
        if c == 1.0:
            P_Y_given_X = dok_array((len(sizes), len(sizes)))
            P_Y_given_X.setdiag(1)
        else:
            P_Y_given_X: dok_array = f(sizes, freqs, c)
        end = time.time()
        elapsed = end - start

        print('Computing join...')
        assert np.allclose(P_Y_given_X.sum(axis=1), 1)
        P_X = np.array(freqs)
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

        def bandwidths():
            S_X = np.array(sizes)
            ones = P_XY.copy()
            ones[ones.nonzero()] = 1
            in_size = ones * S_X[:, None]
            out_size = ones * S_X[None, :]
            abs = (P_XY * (out_size - in_size)).sum()
            rel = (P_XY * (out_size * in_size.power(-1))).sum()
            return abs, rel

        # from shared_functions import leakage_renyi, leakage_shannon
        # print(leakage_renyi(P_Y_given_X.toarray(), freqs))
        # print(leakage_shannon(P_Y_given_X.toarray(), freqs))

        print('Computing leakages...')
        renyi, shannon = leakages()
        print('Computing bandwidths...')
        abs_bandwidth, rel_bandwidth = bandwidths()
        print('Done.')
        return renyi, shannon, elapsed, abs_bandwidth, rel_bandwidth

    return wrapper


@decorated
def Renyi_PRP(sizes: List[int], freqs: List[float], c: float):
    n = len(sizes)
    S_X = np.array(sizes)
    P_X = np.array(freqs)

    P_XY = dok_array((n, n))

    for j in tqdm(range(n - 1, -1, -1)):
        q = P_X[j]
        if q == 0:
            continue
        for i in range(j, -1, -1):
            if not S_X[i] <= S_X[j] <= c * S_X[i]:
                break
            P_XY[i, j] = min(q, P_X[i])
            P_X[i] -= P_XY[i, j]

    P_X = P_XY.sum(axis=1)
    P_Y_given_X = P_XY.multiply(1 / P_X[:, None]).asformat('dok')
    return P_Y_given_X


def nodeJS():
    file = Path('paper-cases/npm.txt')
    if not file.exists():
        print('Creating dataset from original...')
        import pandas as pd
        df = pd.read_csv(
            './datasets/npm_no_scope_full_stats_nonzero_downloads.csv',
            names=['name', 'size', 'visits'],
        )
        df.sort_values(by=['visits'], ascending=False, inplace=True)
        df.sort_values(by=['size'], inplace=True)
        sizes = df['size'].to_list()
        freqs = (df['visits'] / df['visits'].sum()).to_list()
        del df
        c = 1.1
        with open(file, 'w') as f:
            f.write(f'1\n{len(sizes)} {c}\n')
            f.write(f'{" ".join(map(str, sizes))}\n')
            f.write(f'{" ".join(map(str, freqs))}\n')
        print(f'Saved as {file}.')
        print('Avg Bandwidth:', np.dot(sizes, freqs))
    _, _, _, sizes, freqs = next(file_parser_iterator(file)[1])
    return sizes, freqs


def main():
    sizes, freqs = nodeJS()
    print(Renyi_PRP(sizes, freqs, 1.0))

    return
    with open('paper-cases/.output.txt', 'a') as f:
        for c in [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
            # sizes = sizes[:1000]
            # freqs = freqs[:1000]
            # freqs /= np.sum(freqs)
            measurements = Renyi_PRP(sizes, freqs, c)
            msg = f'RenyiPRP c={c}: {measurements}'
            f.write(f'{msg}\n')
            print(msg)


if __name__ == '__main__':
    main()