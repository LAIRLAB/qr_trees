#!/usr/bin/env python

#
# Shervin Javdani (sjavdani@cs.cmu.edu)
# January 2017
#

# If we are not running from the build directory, then add lib to path from
# build assuming we are running from the python folder
import os
full_path = os.path.realpath(__file__)
if full_path.count("src/python") > 0:
    import sys
    to_add = os.path.abspath(os.path.join(os.path.split(full_path)[0], "../../build/"))
    sys.path.append(to_add)

from IPython import embed

import lib.shared_autonomy_circle_bindings as ilqr
import visualize_circle_world as vis

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    obs_prior = [0.5, 0.5] 
    world_dims = [-30, 30, -30, 30]

    w1 = ilqr.CircleWorld(world_dims)
    w2 = ilqr.CircleWorld(world_dims)

    obs_pos_1 = [-2, 0.0]
    obs_pos_2 = [2, 0.0]

    obs_radius = 10.0
    obstacle_1 = ilqr.Circle(obs_radius, obs_pos_1);
    obstacle_2 = ilqr.Circle(obs_radius, obs_pos_2);

    # add obstacle to world 1
    w1.add_obstacle(obstacle_1);
    # add obstacle to world 2
    w2.add_obstacle(obstacle_2);


    cost, states_true_1, obs_fname_1 = ilqr.control_shared_autonomy(ilqr.TRUE_ILQR, 
            w1, w2, obs_prior, "true1", "true1")
    cost, states_true_2, obs_fname_2 = ilqr.control_shared_autonomy(ilqr.TRUE_ILQR, 
            w2, w1, obs_prior, "true2", "true2")

    cost, states_weighted_1, obs_fname_3 =\
            ilqr.control_shared_autonomy(ilqr.PROB_WEIGHTED_CONTROL, 
                w1, w2, obs_prior, "weight3", "weight3")
    cost, states_weighted_2, obs_fname_4 =\
            ilqr.control_shared_autonomy(ilqr.PROB_WEIGHTED_CONTROL, 
                w2, w1, obs_prior, "weight4", "weight4")

    cost, states_hind_1, obs_fname_5 =\
            ilqr.control_shared_autonomy(ilqr.HINDSIGHT, 
                w1, w2, obs_prior, "hind3", "hind3")
    cost, states_hind_2, obs_fname_6 =\
            ilqr.control_shared_autonomy(ilqr.HINDSIGHT,
                w2, w1, obs_prior, "hind4", "hind4")

    
    print("Drawing world 1")
    ax1 = vis.parse_draw_files([states_true_1, states_weighted_1, states_hind_1], obs_fname_1,
            show=False) 
    plt.title('World 1')

    print("Drawing world 2")
    ax2 = vis.parse_draw_files([states_true_2, states_weighted_2, states_hind_2], 
            obs_fname_2, show=False) 
    plt.title('World 2')
    plt.show()

    embed()


