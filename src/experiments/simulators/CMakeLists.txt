add_library(simulators 
    SHARED 
        circle_world.cc
        pendulum.cc 
        simplecar.cc
        simulator_utils.cc
        )
target_link_libraries(simulators)

