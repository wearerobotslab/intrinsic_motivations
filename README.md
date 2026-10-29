# Intrinsic motivations
Repositories for intrinsic motivation implementations for dynamical systems. Mainly used for robotics and optimisation, but can be used in any feedback loop.

## Repository layout
One folder per intrinsic motivation in the repository root; inside it, one folder per language:
```
homeokinesis/                        Python package `homeokinesis` (__init__.py)
  cpp/eigen/                         C++ (Eigen) version
  python/
    homeokinesis_numpy.py            numpy version
    homeokinesis_torch.py            torch version
    examples/                        example scripts (not part of the package)
tests/                               tests for all versions
```

## C++
### Homeokinesis (Eigen)
`homeokinesis/cpp/eigen/homeokinesis_controller.cpp` and `.hpp`. 1-to-1 implementation of the SOX controller from The Playful Machine (Der and Martius, 2012). Depends on the Eigen library. As used in AREFramework.

## Python
### Homeokinesis
#### Torch version
`homeokinesis/python/homeokinesis_torch.py`. 1-to-1 implementation of the SOX controller. Depends on Torch and is wrapped for the Gymnasium RL framework. Used for running multiple parallel robots in NVidia Isaac Sim.

`homeokinesis/python/examples/play_hk.py` is an example of running the controller on Isaac Gym.

#### Numpy version
`homeokinesis/python/homeokinesis_numpy.py`. 1-to-1 implementation of the SOX controller. Depends on Numpy. Used for running homeokinesis on the Qutee robot.

`homeokinesis/python/examples/qutee_homeokinesis.py` is an example of running the controller on the Qutee robot.

## Installing the Python package with pip
The Python code installs as the package `homeokinesis` (distribution name `intrinsic-motivations`):
```python
from homeokinesis import HomeokinesisNumpy   # numpy version
from homeokinesis import HomeokinesisTorch   # torch version

hk = HomeokinesisNumpy(number_sensor=8, number_motor=8)
y = hk.step(x)   # sensor values in, motor values out
```
`HomeokinesisTorch` is only imported when you ask for it. Its module also imports `gymnasium` and Isaac Lab (`omni.isaac.lab`), so those must be installed to use it.

### As a user
Install straight from GitHub (needs only numpy):
```
pip install git+https://github.com/wearerobotslab/intrinsic_motivations.git
```
With the torch version's dependency as well:
```
pip install "intrinsic-motivations[torch] @ git+https://github.com/wearerobotslab/intrinsic_motivations.git"
```
From a local clone: `pip install .` (or `pip install ".[torch]"`) in the repository root.

### As a developer
Clone the repository and install it in editable mode, so your changes to the code are used without reinstalling:
```
git clone https://github.com/wearerobotslab/intrinsic_motivations.git
cd intrinsic_motivations
conda env create -f environment.yml     # numpy, pytest, CPU PyTorch, Eigen, C++ compiler (see below)
conda activate homeokinesis
pip install -e ".[dev]"                 # editable install + pytest and torch
```
Then run the tests as described in [Testing homeokinesis](#testing-homeokinesis). The tests load the implementations straight from the source files, so they always test your checkout.

### Without installing
Add the repository root to the Python path and use the same imports:
```python
import sys
sys.path.insert(0, "path/to/intrinsic_motivations")
from homeokinesis import HomeokinesisNumpy
```

## Testing homeokinesis
The `tests/` folder checks all three homeokinesis implementations (numpy, torch, C++/Eigen) against the original SOX controller of lpzrobots (Der & Martius), both its code and its numerical results. See [tests/README.md](tests/README.md) for what each test does.

### 1. Create the conda environment
`environment.yml` installs everything the tests need: Python, numpy, pytest, CPU-only PyTorch, Eigen and a C++ compiler. Isaac Lab and the robots are not needed.
```
conda env create -f environment.yml
conda activate homeokinesis
```
To update an existing environment after `environment.yml` changes: `conda env update -f environment.yml --prune`.

### 2. Get lpzrobots (for the comparison with the original SOX)
The tests compile the original `Sox` from [lpzrobots](https://github.com/georgmartius/lpzrobots) (`selforg/controller/sox.cpp`) and compare against it. Clone it into `external/` (ignored by git), where the tests find it automatically:
```
git clone --depth 1 https://github.com/georgmartius/lpzrobots external/lpzrobots
```
If you already have lpzrobots somewhere else, point the tests to it instead:
`export LPZROBOTS_SELFORG=/path/to/lpzrobots/selforg` (fish: `set -x LPZROBOTS_SELFORG /path/to/lpzrobots/selforg`).
Without lpzrobots these tests are skipped (shown as `s`); everything else still runs.

### 3. Run the tests
From the repository root:
```
pytest tests              # everything
pytest tests -v           # list every test by name
pytest tests -k numpy     # only tests whose name contains "numpy" (or torch, cpp, lpzrobots)
pytest tests -x           # stop at the first failure
```
Expected result with lpzrobots present: all tests pass (`72 passed`). A failure prints which matrix or motor output differs from SOX, at which step and by how much.

If Eigen is not found automatically, set `EIGEN_INCLUDE_DIR` to the folder that contains `Eigen/Core` (e.g. `$CONDA_PREFIX/include/eigen3`).
