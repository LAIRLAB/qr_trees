
add_subdirectory(simulators)

add_executable(ilqr_tree_pendulum ilqr_tree_pendulum.cc)
target_link_libraries(ilqr_tree_pendulum 
    ilqr_tree ilqr utils simulators helpers policies)

add_executable(single_obs_circle_world single_obs_circle_world.cc)
target_link_libraries(single_obs_circle_world
    ilqr_tree ilqr utils simulators helpers policies)

