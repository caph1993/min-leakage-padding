from subprocess import run, PIPE
from pathlib import Path
import sys

cwd = Path(__file__).parent  # Working directory


def install_requirements():
    pip_args = [sys.executable, '-m', 'pip']
    run([*pip_args, 'install', '--upgrade', 'pip'])
    run([*pip_args, 'install', '-r', 'requirements.txt'])
    return


def test1():
    run([
        sys.executable,
        cwd / 'comparison.py',
        'Renyi_POP',
        'Shannon_POP',
        'samples/medium-200.txt',
    ], stdout=PIPE)


def test2():
    run([
        sys.executable,
        cwd / 'comparison.py',
        'Renyi_POP',
        'Shannon_POP',
        'samples/xsmall-200.txt',
    ], stdout=PIPE)


def main():
    #install_requirements()
    #test1()
    test2()
    pass


if __name__ == '__main__':
    main()