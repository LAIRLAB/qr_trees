#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import mayavi.mlab as mlab

TRUE_ILQR_FNAME = "true_ilqr_states.csv"

DRAW_Z = 0
BASE_SCALE = 1.0

robot_radius = 1; 
start_pos = np.asarray((-10, 0))
end_pos = -start_pos

obs_radius = 2; 
obs_pos = (end_pos + start_pos) / 2.0;
print end_pos 
print start_pos 

mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1), size=(800, 800))

mlab.points3d(start_pos[0], start_pos[1], DRAW_Z, color=(0,0,1), scale_factor=BASE_SCALE*robot_radius)
mlab.points3d(end_pos[0], end_pos[1], DRAW_Z, color=(1,0,0), scale_factor=BASE_SCALE*robot_radius)

mlab.points3d(obs_pos[0], obs_pos[1], DRAW_Z, color=(0,0,0), scale_factor=BASE_SCALE*obs_radius) 


states = np.genfromtxt(TRUE_ILQR_FNAME);

for state in states:
    print state

mlab.show()

print('hi')
