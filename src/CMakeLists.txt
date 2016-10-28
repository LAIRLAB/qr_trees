add_subdirectory(utils)
add_subdirectory(ilqr)

add_executable(test test.cc)
target_link_libraries(test utils)

include_directories(ilqr)
