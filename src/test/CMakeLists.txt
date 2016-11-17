add_executable(test_tree test_tree.cc)

add_executable(test_math test_math.cc)
target_link_libraries(test_math utils)

add_executable(test_as_lqr test_as_lqr.cc)
target_link_libraries(test_as_lqr ilqr_tree lqr ilqr)

add_executable(test_lqr_tree test_lqr_tree.cc)
target_link_libraries(test_lqr_tree lqr_tree lqr)

add_executable(test_ilqr_chain test_ilqr_chain.cc)
target_link_libraries(test_ilqr_chain ilqr lqr utils)

