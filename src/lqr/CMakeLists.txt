add_library(lqr_tree SHARED lqr_tree.cc lqr_types.cc LQR.cc)
target_link_libraries(lqr_tree utils)
