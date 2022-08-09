How to execute all the examples of this repository:

```sh
pip3 install matplotlib numpy Pillow PySimpleGUI
python3 generator_all.py samples --seed 0

python3 comparison.py Renyi_POP Renyi_POP_bruteforce samples/xsmall-20000.txt
python3 comparison.py Renyi_POP Shannon_POP samples/medium-200.txt
```
