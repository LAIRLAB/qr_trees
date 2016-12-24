
add_subdirectory(simulators)

add_executable(ilqr_tree_pendulum ilqr_tree_pendulum.cc)
target_link_libraries(ilqr_tree_pendulum 
    ilqr_tree ilqr utils simulators helpers policies)

add_executable(single_obs_circle_world single_obs_circle_world.cc)
target_link_libraries(single_obs_circle_world
    ilqr_tree ilqr utils simulators helpers policies)

# Templatized ilQR versions
add_executable(ilqr_templated_pendulum ilqr_templated_pendulum.cc)
target_link_libraries(ilqr_templated_pendulum simulators)

add_executable(ilqr_templated_diffdrive ilqr_templated_diffdrive.cc)
target_link_libraries(ilqr_templated_diffdrive simulators)

add_executable(ilqr_hindsight_diffdrive ilqr_hindsight_diffdrive.cc)
target_link_libraries(ilqr_hindsight_diffdrive simulators)
