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

import lib.shared_autonomy_circle_class_bindings as ilqr
import visualize_circle_world as vis

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    world_dims = [-30, 30, -30, 30]

    world = ilqr.CircleWorld(world_dims)

    obs_pos_1 = [0, 0.0]
    obs_radius = 10.0
    obstacle_1 = ilqr.Circle(obs_radius, obs_pos_1);

    # add obstacle to world 1
    world.add_obstacle(obstacle_1);

    goal_states = []
    goal_states.append(np.array([-15., 25.]))
    goal_states.append(np.array([15., 25.]))

    goal_priors = [0.5, 0.5]

    
    true_goal_ind = 0
    num_timesteps = 20
    ilqr_true = ilqr.SharedAutonomyCircle(ilqr.TRUE_ILQR, world, goal_states, goal_priors, true_goal_ind, num_timesteps)
    ilqr_true.run_control(ilqr_true.get_num_timesteps_remaining())

    ilqr_hindsight = ilqr.SharedAutonomyCircle(ilqr.HINDSIGHT, world, goal_states, goal_priors, true_goal_ind, num_timesteps)
    ilqr_hindsight.run_control(ilqr_hindsight.get_num_timesteps_remaining())

    ilqr_weighted = ilqr.SharedAutonomyCircle(ilqr.PROB_WEIGHTED_CONTROL, world, goal_states, goal_priors, true_goal_ind, num_timesteps)
    ilqr_weighted.run_control(ilqr_weighted.get_num_timesteps_remaining())

    controllers = [ilqr_true, ilqr_hindsight, ilqr_weighted]
    labels = ['true', 'hindsight', 'weighted']

    COLORS = [(0.3,0.3,0.3, 0.2), (0.1,0.8,0.8, 0.2), (0.1,0.3,0.8, 0.2)]#, (0.8,0.3,0.8, 0.2), (0.7,0.8,0.2, 0.2)] 

    # draw
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    start_pos = ilqr_true.get_state_at_ind(0)
    end_pos = ilqr_true.get_last_state()

    obs_poses = [o.position for o in world.obstacles]
    obs_radii = [np.array(o.radius) for o in world.obstacles]

    vis.draw_env_multiend(ax, start_pos, goal_states, true_goal_ind, vis.robot_radius, obs_poses, obs_radii);
    labels_circ = []
    for controller, label, color in zip(controllers, labels, COLORS):
        states = np.array(controller.get_states())
        label_circ = vis.draw_traj(ax, states, vis.robot_radius, color=color, label=label, skip=1)
        labels_circ.append(label_circ)
    plt.axis('square')
    plt.axis([float(d) for d in world_dims])
    ax.legend(handles=labels_circ)
    plt.tight_layout()
    plt.show()

#    cost, states_true_1, obs_fname_1 = ilqr.control_shared_autonomy(ilqr.TRUE_ILQR, 
#            world, goal_states, goal_priors, 0, "true1", "true1")
#    cost, states_true_2, obs_fname_2 = ilqr.control_shared_autonomy(ilqr.TRUE_ILQR, 
#            world, goal_states, goal_priors, 1, "true2", "true2")
#    #cost, states_true_2, obs_fname_2 = ilqr.control_shared_autonomy(ilqr.TRUE_ILQR, 
#
#    cost, states_weighted_1, obs_fname_3 = ilqr.control_shared_autonomy(ilqr.PROB_WEIGHTED_CONTROL, 
#            world, goal_states, goal_priors, 0, "weight3", "weight3")
#    cost, states_weighted_2, obs_fname_4 = ilqr.control_shared_autonomy(ilqr.PROB_WEIGHTED_CONTROL, 
#            world, goal_states, goal_priors, 1, "weight4", "weight4")
#
#    cost, states_hind_1, obs_fname_5 = ilqr.control_shared_autonomy(ilqr.HINDSIGHT, 
#            world, goal_states, goal_priors, 0, "hind3", "hind3")
#    cost, states_hind_2, obs_fname_6 = ilqr.control_shared_autonomy(ilqr.HINDSIGHT, 
#            world, goal_states, goal_priors, 1, "hind4", "hind4")
#
#    
#    print("Drawing world 1")
#    ax1 = vis.parse_draw_files([states_true_1, states_weighted_1, states_hind_1], obs_fname_1,
#            show=False) 
#    plt.title('World 1')
#
#    print("Drawing world 2")
#    ax2 = vis.parse_draw_files([states_true_2, states_weighted_2, states_hind_2], 
#            obs_fname_2, show=False) 
#    plt.title('World 2')
#    plt.show()
#
    #embed()


