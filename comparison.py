from itertools import combinations
from typing import Any, cast
from POP_shannon import Shannon_POP
from POP_renyi import Renyi_POP, Renyi_POP_basic
from PRP_shannon import Shannon_PRP
from PRP_renyi import Renyi_PRP
import matplotlib.pyplot as plt
import numpy as np

from gui import new_visualizer
from shared_functions import (
    parse_command_line,
    main_iterator,
    leakage_shannon,
    leakage_renyi,
    plot_solution,
    verify_solution,
)


def main():
    test_file = parse_command_line(__doc__)
    vpath = new_visualizer(title=test_file.name if test_file else 'STDIN')
    solvers = {
        #'Shannon_POP': Shannon_POP,
        'Renyi_POP0': Renyi_POP_basic,
        'Renyi_POP': Renyi_POP,
        'Renyi_PRP': Renyi_PRP,
        #'Renyi_POP_seb': Renyi_POP_sebastian,
        #'Shannon_PRP': Shannon_PRP,
        #'Renyi_PRP': Renyi_PRP,
        #'naive': naive,
    }

    def make_datas():
        datas = {}
        examples = {tc: {} for tc in range(1, 11)}
        for solver_name, solver in solvers.items():
            print(solver_name)
            data = []
            for packet in main_iterator(test_file, solver):
                tc, n, c, sizes, freqs, P_Y_given_X = packet
                verify_solution(P_Y_given_X, sizes, freqs, c)
                shannon = leakage_shannon(P_Y_given_X, freqs)
                renyi = leakage_renyi(P_Y_given_X, freqs)
                print(f'Case #{tc:3d}: ({n:3d} objects)'
                      f' shannon={shannon:.5f} renyi={renyi:.5f}')
                data.append((shannon, renyi))
                if tc in examples:
                    examples[tc][solver_name] = packet
            datas[solver_name] = np.array(data, dtype=float)
        return datas, examples

    datas, examples = make_datas()

    markers = {
        'POP_shannon': ('x', 1.0),
        'POP_renyi': ('+', 1.0),
        'POP_renyi_seb': ('x', 1.0),
        'PRP_shannon': ('x', 1.0),
        'PRP_renyi_car': ('x', 1.0),
        'PRP_renyi_seb': ('+', 1.0),
        'POP_renyi_shannon': ('x', 1.0),
        'naive': ('o', 0.1),
        'default': ('o', 0.1),
    }

    # REF = 'naive'
    REF = 'POP_renyi'

    def plot_diffs():
        for solver_name, data in datas.items():
            marker, alpha = markers.get(solver_name, markers['default'])
            plt.scatter(x=data[:, 0], y=data[:, 1], label=solver_name,
                        marker=marker, alpha=alpha)
        plt.xlabel('Shannon leakage')
        plt.ylabel('Rényi leakage')
        plt.legend()
        plt.savefig(vpath / 'A.png')
        plt.close()

        l = list(datas.items())
        i = 0
        for (name1, data1), (name2, data2) in combinations(l, 2):
            diff = data1 - data2
            fig, ax = plt.subplots()
            ax.scatter(x=diff[:, 0], y=diff[:, 1], marker=cast(Any, 'x'),
                       alpha=0.8)
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')
            plt.xlabel(f'Shannon {name1} minus {name2}')
            plt.ylabel(f'Rényi {name1} minus {name2}')
            #plt.legend()
            plt.savefig(vpath / f'B{i}.png')
            plt.close()
            i += 1

    plot_diffs()

    def plot_examples():
        for tc in examples:
            for solver_name in examples[tc]:
                packet = examples[tc][solver_name]
                tc, n, c, sizes, freqs, P_Y_given_X = packet
                verify_solution(P_Y_given_X, sizes, freqs, c)
                plot_solution(
                    P_Y_given_X, sizes, freqs,
                    title=f'{solver_name}. Case #{tc:3d} ({n:3d} objects)',
                    save=vpath / f'D-{tc:03d}_{solver_name}.png')

    plot_examples()


if __name__ == '__main__':
    main()
