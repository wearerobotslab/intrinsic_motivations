# Intrinsic motivations
Repositories for intrinsic motivation implementations.

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
