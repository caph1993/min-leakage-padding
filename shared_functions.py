import functools
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, TypeVar, Union
import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.typing import NDArray
from gui import new_visualizer as launch_viewer, VisualizerPath


def xlog2x(x: np.ndarray):
    assert np.all(x >= 0), x
    out = np.zeros_like(x)
    mask = x > 0
    out[mask] = x[mask] * np.log2(x[mask])
    return out


def leakage_renyi(P_Y_given_X: np.ndarray, freqs: List[float]):
    assert np.allclose(P_Y_given_X.sum(axis=1), 1)
    P_X = np.array(freqs)
    H_X = -np.log2(np.max(P_X))
    cond_H = -np.log2(np.sum(np.max(P_Y_given_X * P_X[:, None], axis=0)))
    return H_X - cond_H


def leakage_shannon(P_Y_given_X: np.ndarray, freqs: List):
    assert np.allclose(P_Y_given_X.sum(axis=1), 1)
    P_X = np.array(freqs)
    P_Y = np.sum(P_X[:, None] * P_Y_given_X, axis=0)
    assert np.allclose(P_X.sum(), 1), P_X
    assert np.allclose(P_Y.sum(), 1), P_Y
    H_Y = -np.sum(xlog2x(P_Y))
    H_Y_given_X = -np.dot(P_X, np.sum(xlog2x(P_Y_given_X), axis=1))
    return H_Y - H_Y_given_X


def verify_solution(P_Y_given_X: NDArray, sizes: List[int], freqs: List[float],
                    c: float):
    assert np.allclose(P_Y_given_X.sum(axis=1), 1)
    n = len(sizes)
    for i in range(n):
        for j in range(n):
            if P_Y_given_X[i, j] > 0:
                if not sizes[i] <= sizes[j] <= c * sizes[i]:
                    if np.allclose(sizes[j], c * sizes[i]):
                        continue
                    print(P_Y_given_X)
                    plot_solution(P_Y_given_X, sizes, freqs)
                    raise Exception('ERROR', sizes[i], sizes[j], c, i, j)
    return


import sys


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


def parse_command_line(description: Optional[str] = None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('test_cases_file', type=Path,
                        help='File containing test cases')
    args = parser.parse_args()
    if args.test_cases_file == Path('STDIN'):
        return None
    return Path(args.test_cases_file)


def main_iterator(test_cases: Union[Path, str, None],
                  solver: Callable[..., np.ndarray], start_at: int = 0):

    _, cases = file_parser_iterator(test_cases)

    for tc, n, c, sizes, freqs in cases:
        if tc < start_at:
            continue
        P_Y_given_X = solver(sizes, freqs, c)
        assert np.allclose(P_Y_given_X.sum(axis=1), 1)
        P_Y_given_X = P_Y_given_X.astype(np.float64)
        P_Y_given_X /= P_Y_given_X.sum(axis=1)[:, None]
        yield tc, n, c, sizes, freqs, P_Y_given_X
    return


_F = TypeVar('_F', bound=Callable[[List[int], List[float], float], np.ndarray])


def sorting_sizes(solve: _F) -> _F:

    @functools.wraps(solve)
    def wrapper(sizes: List[int], freqs: List[float], c: float):
        n = len(freqs)
        idx = np.argsort(sizes)
        if np.any(idx != np.arange(n)):
            # If sizes are not sorted
            sorted_sizes = [sizes[i] for i in idx]
            sorted_freqs = [freqs[i] for i in idx]
            P_Y_given_X = solve(sorted_sizes, sorted_freqs, c)
            inv_idx = np.argsort(idx)
            P_Y_given_X = P_Y_given_X[inv_idx, :][:, inv_idx]
            return P_Y_given_X
        return solve(sizes, freqs, c)

    return wrapper  # type:ignore


def print_solution(
    groups: List[List[int]],
    group_freqs: List[float],
    shannon_info: float,
):
    for i, (gr, fr) in enumerate(zip(groups, group_freqs)):
        print(f'  Group #{i+1}: {fr:.6f}', *gr)
    print(f'  {shannon_info:.6f} bits')
    print()
    return


def plot_solution(
    P_Y_given_X: np.ndarray,
    sizes: List[int],
    freqs: List[float],
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save: Optional[Path] = None,
    renyi=None,
    shannon=None,
):
    S_X = np.array(sizes)
    P_X = np.array(freqs)
    P_Y = np.matmul(P_X, P_Y_given_X)
    n = len(P_X)
    _ax = ax
    ax = plt.gca() if ax is None else ax

    ax.plot([0, max(S_X) * 1.25], [1, 1], color='black', alpha=0.5)
    ax.plot([0, max(S_X) * 1.25], [-1, -1], color='black', alpha=0.5)
    ax.scatter(S_X, [+1] * n, color='black', alpha=0.5)
    ax.scatter(S_X, [-1] * n, color='black', alpha=0.5)
    width = 5
    ax.bar(S_X, -P_Y / P_Y.max() * 0.45, color='tab:red', bottom=-1,
           width=width, alpha=0.8)
    ax.bar(S_X, P_X / P_X.max() * 0.45, color='tab:blue', bottom=1, width=width,
           alpha=0.8)

    if renyi is None:
        renyi = leakage_renyi(P_Y_given_X, freqs)
    if shannon is None:
        shannon = leakage_shannon(P_Y_given_X, freqs)
    ax.text(x=max(S_X) * 1.1, y=0.5, s=f'Rényi: {renyi:.4f}')
    ax.text(x=max(S_X) * 1.1, y=-0.5, s=f'Shannon: {shannon:.4f}')
    ax.set_xlim([-max(S_X) * 0.1, max(S_X) * 1.5])
    for i in range(n):
        for j in range(n):
            strength = P_Y_given_X[i, j]
            if strength > 0:
                arrowprops = dict(arrowstyle="->", alpha=strength,
                                  color='tab:green')
                ax.annotate("", xytext=(S_X[i], 1), xy=(S_X[j], -1),
                            arrowprops=arrowprops)
                ax.annotate("", xytext=(S_X[i], 1),
                            xy=((S_X[i] + S_X[j]) / 2, 0),
                            arrowprops=arrowprops)
    ax.set_ylabel('Padded' + ' ' * 39 + 'Original')
    ax.set_xlabel('Object size')
    ax.set_ylim([-1.5, 1.5])
    if title is not None:
        ax.set_title(title)
    if save:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / save.name
            plt.savefig(tmp)
            plt.close()
            tmp.replace(save)
    elif _ax is None:
        plt.show()
    return


def print_plot_solution(
    vpath: VisualizerPath,
    test_case,
    name: str,
    P_Y_given_X: np.ndarray,
    title: Optional[str] = None,
    renyi=None,
    shannon=None,
):
    tc, n, c, sizes, freqs = test_case
    if renyi is None:
        renyi = leakage_renyi(P_Y_given_X, freqs)
    if shannon is None:
        shannon = leakage_shannon(P_Y_given_X, freqs)
    if title is None:
        title = f'{name}. Case #{tc:3d} ({n:3d} objects)'
    plot_solution(P_Y_given_X, sizes, freqs, title=title, save=vpath.png())
    vpath.print(f'Case #{tc:3d} ({n:3d} objects)')
    vpath.print(f'Sizes: {sizes}')
    vpath.print(f'Freqs: {freqs}')
    vpath.print(f'Renyi: {renyi}')
    vpath.print(f'Shannon: {shannon}')
    vpath.print(f'Output of {name}:')
    vpath.print(P_Y_given_X)
    next(vpath)