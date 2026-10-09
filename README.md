# Intrinsic motivations
Repositories for intrinsic motivation implementations for dynamical sustems. Mainly used for robotics and optimisation, but can be used in any feedback loop.

## C++
### Homeokinesis (Eigen)
/cpp/homeokinesis/eigen/homekinesis.cpp .hpp. 1-to-1 implementation of the SOX controller from The Playful Machine (Der and Martius, 2012). Depends on Eigen library. As used in AREFramework. 

### Homeokinesis (portable: Eigen only, float or double)
/cpp/homeokinesis/portable/homeokinesis.hpp. Header-only version of the same SOX controller for embedded use (used on the Qutee's ESP32-S3): no AREFramework dependency, fixed-size matrices (no memory allocation while running) and float or double.

```cpp
#include "homeokinesis.hpp"
im::hk::Homeokinesis<float, 12, 12> hk;   // 12 sensors, 12 motors
hk.params().epsA = 0.1f;                  // all parameters are in hk.params()
auto y = hk.step(x);                      // act on the sensor values x, then learn
```

- Same equations and defaults as the Eigen version above, in LpzRobots' order: each step stores the new sensor values and computes the motor values, then learns from the newest sensor values and from the sensors and motors `steps4Delay` steps earlier. (The Eigen version's `update()` calls `learn()` before `step()`, so at that point its "future" sensor value is the oldest one in the buffer.)
- `test_homeokinesis.cpp` compares it with a reference written like the Eigen version: identical in double, within about 1e-5 in float over 2000 steps. `example.cpp` runs it on a simple simulated system and times a step.
- Build: `g++ -std=c++17 -O2 -isystem /usr/include/eigen3 test_homeokinesis.cpp -o test_homeokinesis && ./test_homeokinesis`
- Note on the numpy version below: its vectors are 1-D, so several updates that should be outer products (A, S, C) are element-wise or dot products, `L` adds a vector to each row of S instead of computing `A·diag(g')·C + S`, and `mu` scales columns instead of rows. Its results therefore differ from the C++ versions.

## Python
### Homeokinesis
#### Torch version
/python/homeokinesis/torch/homekinesis.py. 1-to-1 implementaition of SOX controller. Depends on Torch and wrapped for Gymnasium RL framework. Used for running multiple parallel robots in NVidia Isaac Sim.

The play_hk.py file is an example of running the controller on Isaac Gym.

#### Numpy version
/python/homeokinesis/numpy/homekinesis.py. 1-to-1 implementaition of SOX controller. Depends on Numpy. Used for running homeokinesis on the Qutee robot.
The qutee_homeokinesis.py file is an example of running the controller on the Qutee robot.
