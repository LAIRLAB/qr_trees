#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

from IPython import embed


def draw_circle(ax, pos_ang, radius, color):
    pos = pos_ang[:2]
    circ = plt.Circle(pos, radius, color=color)
    ax.add_artist(circ);
    if len(pos_ang) == 3:
        ang = pos_ang[-1]
        end = np.asarray((cos(ang)*radius, sin(ang)*radius))
        arr = plt.arrow(pos[0], pos[1], end[0], end[1], 
                fc=np.asarray(color)*0.5, ec="k", 
                head_width=radius/2., head_length=radius/2.)
        ax.add_artist(arr);


def draw_env(ax, start_pos, end_pos, robot_radius, obs_poses, obs_radii):
    draw_circle(ax, start_pos, robot_radius, color=(0,1,0))
    draw_circle(ax, end_pos, robot_radius, color=(1,0,0))

    for i in xrange(len(obs_radii)):
        obs_radius = obs_radii[i]
        obs_pos = obs_poses[i]
        ax.add_artist(plt.Circle(obs_pos, obs_radius, color=(0.1,0.1,0.1)))

def draw_traj(ax, states, robot_radius, color, skip = 3):
    for i in xrange(len(states)):
        if i % skip != 0:
            continue;
        state = states[i]
        draw_circle(ax, state, robot_radius, color=color)
        #ax.add_artist(plt.Circle(state, 0.5, color=color))
    #plt.plot(states[:,0], states[:,1], color=0.5*np.asarray(color), linewidth=3)

if __name__ == "__main__":
    #obstacles_file = "../../build/obstacles.csv"
    #states_file = "../../build/states.csv"
    obstacles_file = "obstacles.csv"
    states_file = "states.csv"

    DRAW_Z = 0
    BASE_SCALE = 1.0

    robot_radius = 3.35/2.0;

    # First two rows are the start and end state. Rest is trajectory.
    states = np.genfromtxt(states_file, delimiter=[13,13,13]);
    start_pos = states[0,:]
    end_pos = states[1,:]
    states = states[2:,:]
    print start_pos

    world_dim = np.genfromtxt(obstacles_file, delimiter=[13,13,13,13], max_rows = 1);

    obstacles = np.genfromtxt(obstacles_file, delimiter=[13,13,13], skip_header = 1);
    obstacles = np.atleast_2d(obstacles)
    obs_poses = obstacles[:,:3]
    obs_radii = obstacles[:,-1]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    draw_env(ax, start_pos, end_pos, robot_radius, obs_poses, obs_radii);
    skip = 3
    draw_traj(ax, states, robot_radius, color=(0.1,0.3,0.8, 0.2), skip=skip);

    plt.axis('square')
    plt.axis(world_dim)
    plt.tight_layout()
    plt.show()


    print('hi')
