# Homeokinesis tests

Setup and run instructions are in the main [README](../README.md#testing-homeokinesis). In short:

    conda env create -f environment.yml && conda activate homeokinesis
    git clone --depth 1 https://github.com/georgmartius/lpzrobots external/lpzrobots
    pytest tests

## How it fits together

    lpzrobots Sox (compiled C++) ──must equal── sox_reference.py ──must equal── numpy / torch / C++ ports
                                                       │
                                     checked against the theory (The Playful Machine)

`sox_reference.py` is a literal Python port of the original SOX controller
(lpzrobots `selforg/controller/sox.cpp`). It is not a test itself; it is the
yardstick. It is checked
* numerically against the compiled original Sox (`test_lpzrobots_baseline.py`,
  equal to ~1e-15), and
* against the theory (`test_reference.py`): L is the Jacobian of the
  sensorimotor loop, the C/h update is gradient descent on the time-loop error
  (The Playful Machine, Eqs. 5.22-5.27), and the model learns a known world.

Each implementation is compared with the reference using that
implementation's own hyperparameters, so only the maths is compared, not the tuning:

| file | what it tests |
|---|---|
| `test_reference.py` | the reference against the theory |
| `test_lpzrobots_baseline.py` | the reference and all three ports against the original lpzrobots Sox |
| `test_numpy_homeokinesis.py` | `homeokinesis/python/homeokinesis_numpy.py` |
| `test_torch_homeokinesis.py` | `homeokinesis/python/homeokinesis_torch.py` (Isaac Lab imports are stubbed if missing; noise switched off for comparisons) |
| `test_cpp_homeokinesis.py`   | `homeokinesis/cpp/eigen` |

Most tests follow one pattern: give the implementation and the reference the
same settings and the same input (one learning step from an identical random
state, or the same sensor stream for 60-300 steps), then compare with
`np.testing.assert_allclose`.

## Helper files
* `conftest.py` - pytest fixtures: loads the Python ports, compiles the C++ port and lpzrobots Sox.
* `cpp/driver.cpp` - wraps the Eigen controller; Python sends it settings and sensor values on stdin and reads the results back.
* `cpp/lpz_sox_driver.cpp` - the same for the original lpzrobots Sox.
* `cpp/stubs/` - minimal stand-ins for the AREFramework headers the Eigen controller includes.

## Environment variables (all optional)
* `LPZROBOTS_SELFORG` - path to `lpzrobots/selforg` if it is not in `external/lpzrobots`.
* `EIGEN_INCLUDE_DIR` - folder containing `Eigen/Core`, if not found automatically.
* `CXX` - C++ compiler to use (default: `g++`).
