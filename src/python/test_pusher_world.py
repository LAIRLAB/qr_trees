#!/usr/bin/env python

# If we are not running from the build directory, then add lib to path from
# build assuming we are running from the python folder
import os
full_path = os.path.realpath(__file__)
if full_path.count("src/python") > 0:
    import sys
    to_add = os.path.abspath(os.path.join(os.path.split(full_path)[0], "../../build/"))
    sys.path.append(to_add)


import lib.pusher_world as pw
import lib.sim_objects as so 

import visualize_circle_world as vis

import numpy as np
import matplotlib.pyplot as plt


from IPython import embed

def draw_world(ax, world, show=True):
    pusher = world.pusher()
    obj = world.object()

    pusher_color = (0.7,0.5,0.5)
    obj_color = (0,0,1)

    labels = []
    l = vis.draw_circle(ax, pusher.position, pusher.radius, pusher_color, "pusher")
    labels.append(l)
    l = vis.draw_circle(ax, obj.position, obj.radius, obj_color, "obj")
    labels.append(l)

    ax.legend(handles=labels)

    if show:
        plt.draw()
        plt.show(block=False);

def draw_traj(ax, states, robot_radius, color, label, skip = 3):
    label_circ = vis.draw_circle(ax, states[0], robot_radius, color=color, label=label)
    for i in xrange(len(states)):
        if i % skip != 0:
            continue;
        state = states[i]
        vis.draw_circle(ax, state, robot_radius, color=color)
        #ax.add_artist(plt.Circle(state, 0.5, color=color))
    #plt.plot(states[:,0], states[:,1], color=0.5*np.asarray(color), linewidth=3)
    return label_circ

def test_pusher_sim():
    T = 10;

    dt = 1
    pusher = so.Circle(2, 0, 0);
    obj = so.Circle(1, 5, 2);

    world_dim = [-10, 10, -10, 10]

    world = pw.PusherWorld(pusher, obj, [0,0], dt)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    plt.axis('square')
    plt.axis(world_dim)

    draw_world(ax, world, show=True);
    inp = raw_input("anything to continue, q to quit: ")
    if inp.lower() == 'q':
        exit()

    quit = False;
    x0 = world.state()
    xt = x0.copy()
    for t in range(T):
        if np.linalg.norm([xt[pw.V_X], xt[pw.V_Y]]) == 0:
            ut = (1,1)
        else:
            ut = (0,0)
        xt = world.step(xt, ut);
        print "t={},w={}".format(t,world)
        ax.cla()    
        draw_world(ax, world, show=True);
        inp = raw_input("anything to continue, q to quit: ")
        if inp.lower() == 'q':
            quit = True
            break

        if t == int(T/2.):
            x0[pw.OBJ_X] = 3
            world.reset(x0)
            xt = world.state()
            

    #if not quit:
    #    plt.show(block=True)
    #embed()

def animate_world(robot_poses, obj_poses):
    quit = False;
    x0 = robot_poses[0,:]
    xt = x0.copy()
    T = len(robot_poses)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    world_dim = [-40, 40, -40, 40]
    plt.axis('square')
    plt.axis(world_dim)

    pusher_color = (0.7,0.5,0.5)
    obj_color = (0,0,1)

    for t in range(T):
        xt = robot_poses[t]
        print "t={}".format(t)
        ax.cla()

        labels = []
        l = vis.draw_circle(ax, robot_poses[t], 4, pusher_color, "pusher")
        labels.append(l)
        l = vis.draw_circle(ax, obj_poses[t], 3, obj_color, "obj")
        labels.append(l)

        ax.legend(handles=labels)

        plt.draw()
        plt.show(block=False);

        inp = raw_input("anything to continue, q to quit: ")
        if inp.lower() == 'q':
            quit = True
            break


if __name__ == "__main__":
    cost, states = pw.control_pusher()
    states = np.asarray(states)
    robot_poses = states[:, 0:2]
    obj_poses = states[:, 4:6]

    robot_radius = 4;
    obj_radius = 3;

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    world_dim = [-40, 40, -40, 40]
    plt.axis('square')
    plt.axis(world_dim)

    pusher_color = (0.7,0.5,0.5, 0.5)
    obj_color = (0,0,1, 0.5)
    labels = []
    l = draw_traj(ax, robot_poses, robot_radius, pusher_color, "pusher", skip=1); 
    labels.append(l)
    l = draw_traj(ax, obj_poses, obj_radius, obj_color, "obj", skip=1); 
    labels.append(l)

    l = vis.draw_circle(ax, obj_poses[1], obj_radius, np.asarray(obj_color)[0:4])
    l._edgecolor = (0,0,0,1); l._linewidth = 3
    l = vis.draw_circle(ax, obj_poses[-1], obj_radius, np.asarray(obj_color)[0:4])
    l._edgecolor = (0,0,0,1); l._linewidth = 1
    plt.text(obj_poses[-1][0], obj_poses[-1][1], "End", color=(0,1,0));
    plt.text(obj_poses[1][0], obj_poses[1][1], "Start", color=(0,1,0));

    l = vis.draw_circle(ax, robot_poses[1], robot_radius, np.asarray(pusher_color)[0:4])
    l._edgecolor = (0,0,0,1); l._linewidth = 3
    l = vis.draw_circle(ax, robot_poses[-1], robot_radius, np.asarray(pusher_color)[0:4])
    l._edgecolor = (0,0,0,1); l._linewidth = 1
    plt.text(obj_poses[-1][0], obj_poses[-1][1], "End", color=(0,1,0));
    plt.text(obj_poses[1][0], obj_poses[1][1], "Start", color=(0,1,0));

    ax.legend(handles=labels)
    plt.show()


