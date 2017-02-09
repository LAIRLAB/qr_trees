#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

from IPython import embed

robot_radius = 3.35/2.0;

def draw_circle(ax, pos_ang, radius, color, label=None):
    pos = pos_ang[:2]
    circ = plt.Circle(pos, radius, color=color, label=label)
    ax.add_artist(circ);
    if len(pos_ang) == 3:
        ang = pos_ang[-1]
        end = np.asarray((cos(ang)*radius, sin(ang)*radius))
        arr = plt.arrow(pos[0], pos[1], end[0], end[1], 
                fc=np.asarray(color)*0.5, ec="k", 
                head_width=radius/2., head_length=radius/2.)
        ax.add_artist(arr);
    return circ

def draw_value_at_positions(ax, positions, values, num_xy_points=None):
  #assume square if the number of x and y points not passed in
  if num_xy_points is None:
    num_x = int(np.sqrt(len(positions)))
    num_xy_points = np.array([num_x, num_x])

  positions_arr = np.array(positions)
  x = positions_arr[:,0].reshape(num_xy_points)
  y = positions_arr[:,1].reshape(num_xy_points)
  z = np.array(values).reshape(num_xy_points)
  ax.contourf(x, y, z, 30, alpha=0.4)

  

def draw_env(ax, start_pos, end_pos, robot_radius, obs_poses, obs_radii):
    draw_circle(ax, start_pos, robot_radius, color=(0,1,0))
    draw_circle(ax, end_pos, robot_radius, color=(1,0,0))

    for obs_radius, obs_pos in zip(obs_radii, obs_poses):
        ax.add_artist(plt.Circle(obs_pos, obs_radius, color=(0.1,0.1,0.1)))

def draw_env_multiend(ax, start_pos, end_poses, true_goal_ind, robot_radius, obs_poses, obs_radii):
    draw_circle(ax, start_pos, robot_radius, color=(0,1,0))
    for ind,end_pos in enumerate(end_poses):
        if ind == true_goal_ind:
            draw_circle(ax, end_pos, robot_radius, color=(1,0,0))
        else:
            draw_circle(ax, end_pos, robot_radius, color=(0.9,0,0, 0.5))

    for obs_radius, obs_pos in zip(obs_radii, obs_poses):
        ax.add_artist(plt.Circle(obs_pos, obs_radius, color=(0.1,0.1,0.1)))

def draw_traj(ax, states, robot_radius, color, label, skip = 3):
    #label_circ = draw_circle(ax, states[0], robot_radius, color=color, label=label)
    label_circ = None
    for i in xrange(len(states)):
        if i % skip != 0:
            continue;
        state = states[i]
        plt_color = color[i] if len(color) == len(states) else color
        if label_circ is None:
          label_circ = draw_circle(ax, state, robot_radius, color=plt_color, label=label)
        else:
          draw_circle(ax, state, robot_radius, color=plt_color, label=None)
        #ax.add_artist(plt.Circle(state, 0.5, color=color))
    #plt.plot(states[:,0], states[:,1], color=0.5*np.asarray(color), linewidth=3)
    return label_circ

def load_and_draw(ax, state_file, color, skip):
    # First two rows are the start and end state. Rest is trajectory.
    states = np.genfromtxt(state_file, delimiter=[13,13,13]);
    start_pos = states[0,:]
    end_pos = states[1,:]
    states = states[2:,:]
    label_circ = draw_traj(ax, states, robot_radius, color=color, label=state_file, skip=skip);
    return label_circ

def parse_draw_files(states_files, obstacles_file, COLORS = None, show = True):
    if COLORS is None:
        COLORS = [(0.3,0.3,0.3, 0.2), (0.1,0.8,0.8, 0.2), (0.1,0.3,0.8, 0.2), (0.8,0.3,0.8, 0.2), (0.7,0.8,0.2, 0.2)] 
    if len(COLORS) < len(states_files):
        for i in range(len(states_files) - len(COLORS)):
            COLORS.append(tuple(np.random.random(3)) + (0.2,))

    DRAW_Z = 0
    BASE_SCALE = 1.0

    world_dim = np.genfromtxt(obstacles_file, delimiter=[13,13,13,13], max_rows = 1);

    obstacles = np.genfromtxt(obstacles_file, delimiter=[13,13,13], skip_header = 1);
    obstacles = np.atleast_2d(obstacles)
    obs_poses = obstacles[:,:2]
    obs_radii = obstacles[:,2:]

    states = np.genfromtxt(states_files[0], delimiter=[13,13,13]);
    start_pos = states[0,:]
    end_pos = states[1,:]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    draw_env(ax, start_pos, end_pos, robot_radius, obs_poses, obs_radii);
    skip = 3
    labels = []
    for i in range(len(states_files)):
        color = COLORS[i]
        label_circ = load_and_draw(ax, states_files[i], color, skip)
        labels.append(label_circ)

    plt.axis('square')
    plt.axis(world_dim)
    ax.legend(handles=labels)
    plt.tight_layout()
    if show:
        plt.show()
    return ax



if __name__ == "__main__":
    #obstacles_file = "../../build/obstacles.csv"
    #states_file = "../../build/states.csv"
    #obstacles_file = "obstacles.csv"
    obstacles_file = "./has_obs_obstacles.csv"
    #states_file = ["states.csv"]
    #states_files = ["./ilqr_true_states.csv", "./hindsight_50-50_states.csv", "./hindsight_25-75_states.csv", "./hindsight_10-90_states.csv", "./argmax_states.csv"]
    #states_files = ["./ilqr_true_states.csv", "./hindsight_10-90_states.csv", "./weighted_10-90_states.csv"]
    states_files = ["./has_obs_ilqr_true_states.csv", "./no_obs_ilqr_true_states.csv"]

    parse_draw_files(states_files, obstacles_file);

    print('hi')
