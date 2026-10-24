# Intrinsic motivations
Repositories for intrinsic motivation implementations for dynamical sustems. Mainly used for robotics and optimisation, but can be used in any feedback loop.

## C++
### Homeokinesis (Eigen)
/cpp/homeokinesis/eigen/homekinesis.cpp .hpp. 1-to-1 implementation of the SOX controller from The Playful Machine (Der and Martius, 2012). Depends on Eigen library. As used in AREFramework. 

## Python
### Homeokinesis
#### Torch version
/python/homeokinesis/torch/homekinesis.py. 1-to-1 implementaition of SOX controller. Depends on Torch and wrapped for Gymnasium RL framework. Used for running multiple parallel robots in NVidia Isaac Sim.

The play_hk.py file is an example of running the controller on Isaac Gym.

#### Numpy version
/python/homeokinesis/numpy/homekinesis.py. 1-to-1 implementaition of SOX controller. Depends on Numpy. Used for running homeokinesis on the Qutee robot.
The qutee_homeokinesis.py file is an example of running the controller on the Qutee robot.

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
