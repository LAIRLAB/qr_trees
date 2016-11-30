add_library(simulators SHARED pendulum.cc simulator_utils.cc)
target_link_libraries(simulators)

