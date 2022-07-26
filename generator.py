from typing import Dict, Tuple
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog=None,
    description='Generate random test cases',
)
parser.add_argument(
    '--seed',
    type=int,
    nargs='?',
    default=0,
    help='seed for random number generator',
)
parser.add_argument(
    'ntc_small',
    type=int,
    help=
    'number of small test cases to generate (n_objects<=10, object_size<=99)',
)
parser.add_argument(
    'ntc_medium',
    type=int,
    help=
    'number of medium test cases to generate (n_objects<=100, object_size<=999)',
)
parser.add_argument(
    'ntc_large',
    type=int,
    help=
    'number of large test cases to generate (n_objects<=1000, object_size<=9999999)',
)
parser.add_argument(
    'ntc_custom',
    type=int,
    help='number of custom test cases to generate (--n_objects, --object_size)',
)
parser.add_argument(
    '--max-object-size',
    type=int,
    nargs='?',
    default=99,
    help='maximum object size for custom test cases',
)
parser.add_argument(
    '--max-collection-size',
    type=int,
    nargs='?',
    default=100,
    help='maximum number of objects for custom test cases',
)
args = parser.parse_args()
seed: int = args.seed

cases: Dict[str, Tuple[int, int, int]] = dict(
    small=(args.ntc_small, 10, 99),
    medium=(args.ntc_medium, 100, 999),
    large=(args.ntc_large, 1000, 9999999),
    custom=(args.ntc_custom, args.max_collection_size, args.max_object_size),
)


def random_freqs(n: int):
    resolution = 100
    while resolution <= n:
        resolution *= 10
    pvals = R.rand(n)
    pvals = pvals / pvals.sum()
    freqs = (1 + R.multinomial(resolution - n, pvals=pvals)) / resolution
    assert np.allclose(freqs.sum(), 1)
    return freqs


print(sum(n for _, (n, _, _) in cases.items()))
for _, (n_tc, n_objects, object_size) in cases.items():
    R = np.random.RandomState(seed)
    for tc in range(n_tc):
        n = R.randint(1, n_objects)
        sizes = R.choice(range(1, object_size + 1), n, replace=False)
        c = (R.choice(sizes) / R.choice(sizes))
        c = max(c, 1 / c)
        c = round(c + 0.01 * R.random(), 2)
        freqs = random_freqs(n)
        idx = np.argsort(sizes)
        sizes = sizes[idx]
        freqs = freqs[idx]
        print(n, c)
        print(*sizes)
        print(*freqs)
