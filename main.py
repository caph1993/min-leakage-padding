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
    cmp_args = [sys.executable, cwd / 'comparison.py']
    run([*cmp_args, 'Renyi_PRP', 'Renyi_POP', 'samples/xsmall-200.txt'])


def test2():
    cmp_args = [sys.executable, cwd / 'comparison.py']
    run([*cmp_args, 'Renyi_POP', 'Shannon_POP', 'samples/medium-200.txt'])


def generate_all():
    generator_args = [sys.executable, cwd / 'generator_all.py']
    run([*generator_args, 'samples', '--seed', "0"])


def main():
    install_requirements()
    generate_all()
    test1()
    test2()
    pass


if __name__ == '__main__':
    main()