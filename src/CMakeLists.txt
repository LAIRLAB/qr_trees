add_subdirectory(utils)

add_executable(test test.cc)
target_link_libraries(test utils)
