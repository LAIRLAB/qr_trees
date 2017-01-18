add_library(simulators 
    SHARED 
        user_goal.cc
        circle_world.cc
        diffdrive.cc
        directdrive.cc
        pendulum.cc 
        simplecar.cc
        simulator_utils.cc
        )
target_link_libraries(simulators)

