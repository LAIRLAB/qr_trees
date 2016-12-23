# qr_trees
Extremely alpha code.

An implementation of (templated) iLQR can be found in headers `src/templated/iLQR.hh` and `src/templated/iLQR_impl.hh`. 

## Requirements
- Eigen3
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
- [pybind11](https://github.com/pybind/pybind11) 

#### Building on Mac OSX with Homebrew
For compiling the python bindings, the default behavior will link using the
built in Apple's python (which works fine). If you want to use the homebrew python in its default
location (e.g. to use the homebrew IPython), you can replace the `cmake ..` command from above with the following.
```
cmake .. -DUSE_PYTHON_HOMEBREW=True
```
This will use the python includes and libraries from 
`/usr/local/Cellar/python/2.7.12_2/Frameworks/Python.framework/Versions/2.7`.
This assumes you have homebrew python version 2.7.12_2. This is not the cleanest
way to do this, but it works for now.

You may also be able to manually specify `PYTHON_INCLUDE_DIRS` and
`PYTHON_LIBRARIES` (e.g. `cmake .. -DPYTHON_INCLUDE_DIRS=[include_dir]
-DPYTHON_LIBRARIES=[libary_dir]`. This option may or may not work and has not
been tested.
