from dataclasses import dataclass
from itertools import combinations
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, List, Tuple, cast
from POP_shannon import Shannon_POP
from POP_renyi import Renyi_POP, Renyi_POP_basic
from PRP_shannon import Shannon_PRP
from PRP_renyi import Renyi_PRP
import matplotlib.pyplot as plt
import numpy as np

from gui import new_visualizer
from shared_functions import (
    file_parser_iterator,
    leakage_shannon,
    leakage_renyi,
    plot_solution,
    verify_solution,
)
from alternative_implementations.POP_renyi_bruteforce import (
    Renyi_POP_bruteforce_1,
    Renyi_POP_bruteforce_non_decreasing,
)
from alternative_implementations.POP_renyi_pinning import Renyi_POP_pinning


@dataclass
class Solver:
    i: int
    name: str
    solver: Callable[[List[int], List[float], float], np.ndarray]
    marker: Any
    alpha = 0.9


def parse_args():
    # Parse args
    solvers = {
        'Shannon_POP':
            Shannon_POP,
        'Renyi_POP0':
            Renyi_POP_basic,
        'Renyi_PRP':
            Renyi_PRP,
        'Renyi_POP':
            Renyi_POP,
        'Renyi_POP_bruteforce_1':
            Renyi_POP_bruteforce_1,
        'Renyi_POP_bruteforce_non_decreasing':
            Renyi_POP_bruteforce_non_decreasing,
        'Renyi_POP_pinning':
            Renyi_POP_pinning,
        #'Renyi_POP_seb': Renyi_POP_sebastian,
        'Shannon_PRP':
            Shannon_PRP,
        'Renyi_PRP':
            Renyi_PRP,
        #'naive': naive,
    }

    parser = ArgumentParser()
    parser.add_argument('solver_main', type=str, default=None)
    parser.add_argument('solver_ref', type=str, default=None,
                        help='Solver to use as reference, e.g. POP_renyi')
    parser.add_argument('test_cases', type=Path, default=None)
    parser.add_argument('--n_examples', type=int, default=10)
    args = parser.parse_args()

    sol_name: str = args.solver_main
    ref_name: str = args.solver_ref
    test_file: Path = args.test_cases
    n_examples = args.n_examples

    sol = Solver(i=0, name=sol_name, solver=solvers[sol_name], marker='x')
    ref = Solver(i=1, name=ref_name, solver=solvers[ref_name], marker='+')
    return sol, ref, test_file, n_examples


def main(sol: Solver, ref: Solver, test_file: Path, n_examples: int):

    def plot_B(examples):
        n = len(examples)
        alpha = 1 if n < 10 else 1 / np.log10(n / 10)
        renyi = np.zeros((n, 2))
        shannon = np.zeros((n, 2))
        sizes = np.zeros(n)

        for s in [sol, ref]:
            sizes[:] = np.array([e[0] for e in examples[:, s.i]])
            renyi[:, s.i] = np.array([e[1] for e in examples[:, s.i]])
            shannon[:, s.i] = np.array([e[2] for e in examples[:, s.i]])

        plot_x = renyi[:, sol.i] - renyi[:, ref.i]
        plot_y = shannon[:, sol.i] - shannon[:, ref.i]

        plt.scatter(x=plot_x, y=plot_y, marker=cast(Any, 'x'), alpha=alpha)
        plt_origin(plot_x, plot_y, s=1)
        plt.xlabel(f'Rényi leakage ({sol.name} wrt {ref.name})')
        plt.ylabel(f'Shannon leakage ({sol.name} wrt {ref.name})')
        plt.title(f'{n} examples of avg. size {sizes.mean():.1f}')
        vpath2.plot_and_close(plt)

        plt.scatter(x=renyi[:, sol.i], y=shannon[:, sol.i], label=sol.name,
                    marker=cast(Any, 'x'), alpha=alpha)
        plt.scatter(x=renyi[:, ref.i], y=shannon[:, ref.i], label=ref.name,
                    marker=cast(Any, '+'), alpha=alpha)
        plt_origin(np.concatenate([renyi[:, sol.i], renyi[:, ref.i]]),
                   np.concatenate([shannon[:, sol.i], shannon[:, ref.i]]), s=0)
        plt.xlabel(f'Rényi leakage')
        plt.ylabel(f'Shannon leakage')
        plt.legend()
        plt.title(f'{n} examples of avg. size {sizes.mean():.1f}')
        vpath3.plot_and_close(plt)

    def plt_origin(plot_x, plot_y, s=0):
        plt.gca().axvline(x=0, color='k')
        plt.gca().axhline(y=0, color='k')
        lim_x = max(1e-6, np.max(np.abs(plot_x)))
        lim_y = max(1e-6, np.max(np.abs(plot_y)))
        plt.xlim(-(s + .1) * lim_x, 1.1 * lim_x)
        plt.ylim(-(s + .1) * lim_y, 1.1 * lim_y)

    # Run
    n_cases, it = file_parser_iterator(test_file)
    n_examples = min(n_examples, n_cases)
    tc_full_examples = set(range(1, 1 + n_examples))
    full_examples = [{} for _ in range(2)]

    full_examples = np.empty(shape=(n_examples, 2), dtype=object)
    examples = np.empty(shape=(n_cases, 2), dtype=object)

    vpath1 = new_visualizer(title=test_file.name if test_file else 'STDIN')
    vpath2 = new_visualizer(title=test_file.name if test_file else 'STDIN')
    vpath3 = new_visualizer(title=test_file.name if test_file else 'STDIN')

    for test_case in it:
        tc, n, c, sizes, freqs = test_case
        for s in [sol, ref]:
            P_Y_given_X = s.solver(sizes, freqs, c)
            verify_solution(P_Y_given_X, sizes, freqs, c)
            shannon = leakage_shannon(P_Y_given_X, freqs)
            renyi = leakage_renyi(P_Y_given_X, freqs)
            print(f'Case #{tc:3d}: ({n:3d} objects)'
                  f' renyi={renyi:.5f} shannon={shannon:.5f} {s.name}')
            examples[tc - 1, s.i] = (n, renyi, shannon)
            if tc in tc_full_examples:
                full_examples[tc - 1, s.i] = P_Y_given_X
                title = f'{s.name}. Case #{tc:3d} ({n:3d} objects)'
                plot_solution(P_Y_given_X, sizes, freqs, title=title,
                              save=vpath1 / f'A-{tc:03d}_{s.name}.png')
        if tc % 100 == 0:
            plot_B(examples[:tc])
    plot_B(examples)
    return


if __name__ == '__main__':
    main(*parse_args())
