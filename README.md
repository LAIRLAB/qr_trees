# lqr_trees
Hindsight Optimization using LQR

## Requirements
- Eigen3
- OpenMP
- Python (python-dev)

## Quick Build Instructions:
```
git clone --recursive https://github.com/LAIRLAB/lqr_trees
cd lqr_trees
mkdir build && cd build
cmake ..
make -j8
```
We do a recursive build to clone all the submodules. Currently the following submodules are used:
- [pybind11](https://github.com/pybind/pybind11) submodule

#### Building on Mac OSX
For compiling the python bindings, the default behavior will link using the
built in python. If you want to use the homebrew python (in its default
location), you can replace the `cmake ..` command from above with the following.
```
cmake .. -DUSE_PYTHON_HOMEBREW=True
```
This wil use the python includes and libraries from 
`/usr/local/Cellar/python/2.7.12_2/Frameworks/Python.framework/Versions/2.7`.

You may also be able to manually specify `PYTHON_INCLUDE_DIRS` and
`PYTHON_LIBRARIES` (e.g. `cmake .. -DPYTHON_INCLUDE_DIRS=[include_dir]
-DPYTHON_LIBRARIES=[libary_dir]`. This option may or may not work and has not
been tested.
