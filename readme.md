How to execute all the examples of this repository:


```sh
pip3 install scipy matplotlib numpy Pillow PySimpleGUI numba awkward # See requirements.txt
# Be sure of being updated:
pip3 install scipy --upgrade # We use dok arrays for sparse matrices (in main.py)
pip3 install numpy --upgrade # We use numpy.typing

#python3 main.py actual_paper_example_PopReSh
#python3 faster.py actual_paper_example_PopReSh # This file reimplements all the functionality using numba for speed. Use main for clarity and faster for performance

# NodeJS tests. Small means 100 top elements. Medium means 1000.
# Large means all, i.e. around 500k.
python3 main.py inspect_data # Histograms of the dataset
python3 main.py small_all # Takes less than 10 seconds
python3 main.py medium_all # Takes around 30 minutes due to PrpSh
#python3 main.py large_all # Takes several days to run due to PopSh complexity
python3 main.py large_PrpReBa # If you only care about PrpReBa (Renyi then bandwidth)

# Correctness tests (250 brute-force and 250 non-brute-force per execution)
# Run it multiple times for more checks (there is no seed)
# search for "checks = [" in main.py for the list of tested assertions
python3 main.py correctness_tests

```
