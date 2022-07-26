from typing import List
import numpy as np
from shared_functions import sorting_sizes, leakage_shannon


@sorting_sizes
def Shannon_PRP(sizes: List[int], freqs: List[float], c: float):

    def shannon_info(x):
        if x != 0:
            return -x * np.log2(x)
        elif x == 0:
            return 0
        raise ValueError('x must be non-negative')

    def next_functions(u, v, sizes, freqs, c):
        n = len(sizes)
        v_t = []
        for s in range(n):
            l = []
            for y in sizes:
                i = sizes.index(y)
                if sizes[s] <= y and y / sizes[s] <= c:
                    l.append(u[i] / sum(
                        u[j]
                        for j in range(n)
                        if sizes[s] <= sizes[j] and sizes[j] / sizes[s] <= c))
                else:
                    l.append(0)
            v_t.append(l)
        u_t = []
        for y in sizes:
            i = sizes.index(y)
            u_t.append(sum(freqs[s] * v_t[s][i] for s in range(n)))
        return u_t, v_t

    def leakage_function(sizes, freqs, u, v):
        n = len(sizes)
        entropy = sum(shannon_info(u[i]) for i in range(n))
        conditional_entropy = sum([
            freqs[s] * sum(shannon_info(v[s][i])
                           for i in range(n))
            for s in range(n)
        ])
        return entropy - conditional_entropy

    def algorithm_new(sizes: List[int], freqs: List[float], c: float):
        n = len(sizes)
        # =============================================================================
        #     idx = np.argsort(sizes)
        #     sizes = [sizes[i] for i in idx]
        #     freqs = [freqs[i] for i in idx]
        # =============================================================================
        data = []
        r = 0
        delta = 10**(-4)
        v = []
        for s in range(n):
            l = []
            ct = 0
            for y in sizes:
                if sizes[s] <= y and y / sizes[s] <= c:
                    ct += 1
            for y in sizes:
                i = sizes.index(y)
                if sizes[s] <= sizes[i] and sizes[i] / sizes[s] <= c:
                    l.append(1 / ct)
                else:
                    l.append(0)
            v.append(l)
        u = []
        for y in sizes:
            i = sizes.index(y)
            u.append(sum(freqs[s] * v[s][i] for s in range(n)))
        while leakage_function(sizes, freqs, u, v) > delta and r < 1000:
            r += 1
            u, v = next_functions(u, v, sizes, freqs, c)
        leakage = leakage_function(sizes, freqs, u, v)
        u_t, v_t = next_functions(u, v, sizes, freqs, c)
        i = 0
        for s in range(n):
            l = []
            l.append(f'P{s}')
            for y in sizes:
                i = sizes.index(y)
                l.append(v_t[s][i])
                # l.append(round(v_t[s][i], 2)) # carlos: Deleted on purpose
            data.append(l)
            i += 1
        return leakage, data

    her_shanon, data = algorithm_new(sizes, freqs, c)
    Y_given_X = np.array([row[1:] for row in data])
    shanon = leakage_shannon(Y_given_X, freqs)
    assert np.allclose(her_shanon, shanon, atol=1e-3), (her_shanon, shanon)
    return Y_given_X
