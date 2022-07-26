How to execute all the examples of this repository:

```sh
pip3 install matplotlib numpy Pillow PySimpleGUI
python3 -c "from pathlib import Path; Path('./samples/').mkdir(exist_ok=True)"

python3 generator.py 10  0 0 0 --seed 0 > samples/small-10.txt
python3 comparison.py samples/small-10.txt

python3 generator.py 200 0 0 0 --seed 0 > samples/small-200.txt
python3 comparison.py samples/small-200.txt

python3 generator.py 0 10  0 0 --seed 0 > samples/medium-10.txt
python3 comparison.py samples/medium-10.txt

python3 generator.py 0 200 0 0 --seed 0 > samples/medium-200.txt
python3 comparison.py samples/medium-200.txt

python3 generator.py 0 0 10  0 --seed 0 > samples/large-10.txt
python3 comparison.py samples/large-10.txt

python3 generator.py 0 0 200 0 --seed 0 > samples/large-200.txt
python3 comparison.py samples/large-200.txt
```
