How to execute all the examples of this repository:

```sh
pip3 install scipy matplotlib numpy Pillow PySimpleGUI
pip3 install scipy --upgrade # We use dok arrays for sparse matrices
pip3 install numpy --upgrade # We use numpy.typing

python3 main.py correctness_tests
python3 main.py actual_paper_example_POP_plus
python3 main.py inspect_data
python3 main.py small_all # Takes less than 10 seconds
python3 main.py medium_all # Takes less than 3 minutes
python3 main.py large_all # Takes some days to run due to PopSh complexity
```
