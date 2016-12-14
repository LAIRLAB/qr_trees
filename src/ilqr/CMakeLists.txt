add_library(ilqr_tree 
            SHARED 
                ilqr_tree.cc 
                ilqr_taylor_expansions.cc 
                ilqr_node.cc 
                ilqrtree_helpers.cc
                )
target_link_libraries(ilqr_tree utils)

add_library(ilqr SHARED iLQR.cc ilqr_taylor_expansions.cc)
target_link_libraries(ilqr utils)

add_library(policies SHARED mpc_tree_policies.cc)
target_link_libraries(policies ilqr utils helpers)
