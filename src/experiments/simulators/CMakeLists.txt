add_library(simulators 
    SHARED 
        circle_world.cc
        diffdrive.cc
        objects.cc
        pendulum.cc 
        pusher.cc
        simplecar.cc
        simulator_utils.cc
        )
target_link_libraries(simulators)


