add_library(ilqr_tree SHARED ilqr_tree.cc ilqr_helpers.cc types.cc)
target_link_libraries(ilqr_tree utils)

add_library(ilqr SHARED iLQR.cc ilqr_helpers.cc)
target_link_libraries(ilqr utils)
