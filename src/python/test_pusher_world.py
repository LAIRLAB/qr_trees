#!/usr/bin/env python

# If we are not running from the build directory, then add lib to path from
# build assuming we are running from the python folder
import os
full_path = os.path.abspath(__file__)
if full_path.count("build") == 0 and full_path.count("src/python") > 0:
    import sys
    sys.path.append(os.path.abspath("../../build/"))


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


if __name__ == "__main__":
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
            world.reset(x0)
            xt = world.state()
            

    #if not quit:
    #    plt.show(block=True)
    #embed()

