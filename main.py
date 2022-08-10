from subprocess import run, PIPE
from pathlib import Path
import sys

cwd = Path(__file__).parent  # Working directory

# For Sebastian, please run: conda init powershell


def install_requirements():
    pip_args = [sys.executable, '-m', 'pip']
    run([*pip_args, 'install', '--upgrade', 'pip'])
    run([*pip_args, 'install', '-r', 'requirements.txt'])
    return


def test1():
    cmp_args = [sys.executable, cwd / 'comparison.py']
    run([*cmp_args, 'Renyi_PRP', 'Renyi_POP', 'samples/xsmall-200.txt'])


def test2():
    cmp_args = [sys.executable, cwd / 'comparison.py']
    run([*cmp_args, 'Renyi_POP', 'Shannon_POP', 'samples/medium-200.txt'])


def case_3384():
    from comparison import Renyi_POP, Shannon_POP, new_visualizer
    from shared_functions import print_plot_solution, file_parser_iterator, leakage_shannon, leakage_renyi
    f_tests = 'samples/xsmall-20000.txt'
    n_cases, it = file_parser_iterator(cwd / f_tests)

    def solve(solver, sizes, freqs, c):
        P_Y_given_X = solver(sizes, freqs, c)
        shannon = leakage_shannon(P_Y_given_X, freqs)
        renyi = leakage_renyi(P_Y_given_X, freqs)
        return renyi, shannon, P_Y_given_X

    vpath1 = new_visualizer(title=f_tests)
    for test_case in it:
        tc, n, c, sizes, freqs = test_case
        if tc < 3384:
            continue
        if tc > 3384:
            break
        ours = solve(Renyi_POP, sizes, freqs, c)
        theirs = solve(Shannon_POP, sizes, freqs, c)

        if ours[0] - theirs[0] < -0.3 and ours[1] - theirs[1] < 0.2:
            print(
                f'Case #{tc:3d}: ({n:3d} objects)'
                f' delta-renyi={ours[0]-theirs[0]:.5f} delta-shannon={ours[1]-theirs[1]:.5f}'
            )
            print_plot_solution(vpath1, test_case, 'Renyi_POP+', ours[2],
                                renyi=ours[0], shannon=ours[1],
                                title='Padding scheme according to Renyi POP+')
            print_plot_solution(vpath1, test_case, 'Shannon_POP', theirs[2],
                                renyi=theirs[0], shannon=theirs[1],
                                title='Padding scheme according to Shannon POP')


def generate_all():
    generator_args = [sys.executable, cwd / 'generator_all.py']
    run([*generator_args, 'samples', '--seed', "0"])


def main():
    # install_requirements()
    # generate_all()
    # test1()
    # test2()
    case_3384()
    pass


if __name__ == '__main__':
    main()