# Setup Python
if (USE_PYTHON_HOMEBREW)
    if (NOT APPLE)
        message(FATAL_ERROR "USE_PYTHON_HOMEBREW only valid on Apple systems")
    endif (NOT APPLE)
    set(PYTHON_HOMEBREW_DIR "/usr/local/Cellar/python/2.7.12_2/Frameworks/Python.framework/Versions/2.7")
    set(PYTHON_INCLUDE_DIRS "${PYTHON_HOMEBREW_DIR}/include/python2.7")
    set(PYTHON_LIBRARIES "${PYTHON_HOMEBREW_DIR}/lib/python2.7/config/libpython2.7.dylib")
else (USE_PYTHON_HOMEBREW)
    find_package(PythonLibs 2 REQUIRED)
endif (USE_PYTHON_HOMEBREW)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/pybind11/include")

set(PYTHONLIBS_FOUND True)

# We have to specifiy a binary directory since the 
# subdirectory is higher level.
add_subdirectory("../../pybind11" "../../pybind11/build")
#pybind11_add_module(bind_test bind_test.cc)
pybind11_add_module(ilqr_diffdrive ilqr_diffdrive.cc)
set_target_properties(ilqr_diffdrive PROPERTIES SUFFIX ".so")
target_link_libraries(ilqr_diffdrive PRIVATE single_obs_diffdrive)

