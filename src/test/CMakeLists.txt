add_executable(test_tree test_tree.cc)

add_executable(test_math test_math.cc)
target_link_libraries(test_math utils)

add_executable(test_multivariate_gaussian test_multivariate_gaussian.cc)
target_link_libraries(test_multivariate_gaussian utils helpers)

add_executable(test_discrete_filter test_discrete_filter.cc)
target_link_libraries(test_discrete_filter simulators)

add_executable(test_ilqr_taylor_expansions test_ilqr_taylor_expansions.cc)
target_link_libraries(test_ilqr_taylor_expansions ilqr lqr utils)

add_executable(test_ilqr_as_lqr test_ilqr_as_lqr.cc)
target_link_libraries(test_ilqr_as_lqr ilqr lqr)

add_executable(test_lqr_tree test_lqr_tree.cc)
target_link_libraries(test_lqr_tree lqr_tree lqr)

add_executable(test_ilqrtree_as_lqr test_ilqrtree_as_lqr.cc)
target_link_libraries(test_ilqrtree_as_lqr ilqr_tree ilqr lqr helpers)

add_executable(test_ilqr_templated test_ilqr_templated.cc)
target_link_libraries(test_ilqr_templated utils lqr ilqr)

add_executable(test_ilqr_hindsight_vs_ilqr test_ilqr_hindsight_vs_ilqr.cc)
target_link_libraries(test_ilqr_hindsight_vs_ilqr simulators)

