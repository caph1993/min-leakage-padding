from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog=None,
    description='Generate all default random test cases',
)
parser.add_argument(
    'out_dir',
    type=Path,
    help='output directory',
)
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='seed for random number generator',
)
args = parser.parse_args()
seed: int = args.seed
out_dir: Path = args.out_dir
assert not out_dir.is_file(), out_dir
out_dir.mkdir(exist_ok=True)

from subprocess import run
import sys

cases = [
    ('xsmall', 20000),
    ('small', 20),
    ('small', 200),
    ('small', 2000),
    ('small', 20000),
    ('medium', 20),
    ('medium', 200),
    ('medium', 2000),
    ('large', 20),
    ('large', 200),
    ('large', 2000),
]

cwd = Path(__file__).parent
args = [sys.executable, cwd / 'generator.py', '--seed', str(seed)]
for tag, n in cases:
    file = out_dir / f'{tag}-{n}.txt'
    print(f'Generating {file}...')
    run([*args, file, f'--{tag}', f'{n}'], cwd=cwd, check=True)
    print(f'Done')
