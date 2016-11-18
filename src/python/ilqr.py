#!/usr/bin/env python

import numpy as np
import numdifftools as ndt

from IPython import embed


class iLQR(object):
    def __init__(self, dynamics, cost, Xs, Us):
        self.dynamics = dynamics
        self.cost = cost
        self.Xs = Xs 
        self.Us = Us 
        self.state_dim = Xs[0].size
        self.control_dim = Us[0].size
        self.T = len(Xs)
        self.taylor_dynamics = [self.update_dynamics(x, u) for (x,u) in zip(Xs, Us)] 
        self.taylor_costs = [self.update_cost(x, u) for (x,u) in zip(Xs, Us)] 

    def update_dynamics(self, x, u):
        xu = np.hstack((x,u))
        J = ndt.Jacobian(self.dynamics_wrapper)
        AB = J(xu);
        A = AB[:,:self.state_dim]
        B = AB[:,self.state_dim:]
        return A,B

    def update_cost(self, x, u):
        xu = np.hstack((x,u))
        Hf = ndt.Hessian(self.dynamics_wrapper)
        H = Hf(xu);
        Q = H[:self.state_dim, :self.state_dim]
        P = H[:self.state_dim, self.state_dim:]
        R = H[self.state_dim:,self.state_dim:]

        gf = ndt.Gradient(self.dynamics_wrapper)
        g = gf(xu)
        gx = g[:self.state_dim]
        gu = g[self.state_dim:]
        return Q,P,R,gx,gu

    def dynamics_wrapper(self, xu):
        x = xu[:self.state_dim]
        u = xu[self.state_dim:]
        return self.dynamics(x, u)

    def cost_wrapper(self, xu):
        x = xu[:self.state_dim]
        u = xu[self.state_dim:]
        return self.cost(x, u)
        

def linear_dynamics(A, B):
    return lambda x,u : A.dot(x) + B.dot(u)

def quadtratic_cost(Q, R):
    return lambda x,u : x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)

state_dim = 3
control_dim = 3

A = np.random.random((state_dim, state_dim));
B = np.random.random((state_dim, control_dim));

Q = np.random.random((state_dim, state_dim));
R = np.random.random((control_dim, control_dim));

dyn_func = linear_dynamics(A, B);
cost_func = linear_dynamics(Q, R);

T = 6
Xs = np.random.random((T, state_dim))
Us = np.random.random((T, control_dim))

ilqr = iLQR(dyn_func, cost_func, Xs, Us);
ilqr.update_dynamics(Xs[0], Us[0])



