
add_subdirectory(simulators)

add_executable(ilqr_tree_pendulum ilqr_tree_pendulum.cc)
target_link_libraries(ilqr_tree_pendulum ilqr_tree ilqr utils simulators)

