#!/usr/bin/env python

import numpy as np
import numdifftools as ndt

from collections import namedtuple

from IPython import embed

TaylorDynamics = namedtuple('TaylorDynamics', ['A', 'B', 'f'])
TaylorCost = namedtuple('TaylorCost', ['Q', 'R', 'P', 'gx', 'gu', 'c'])

ValueTerms = namedtuple('ValueTerms', ['V', 'G', 'w'])


class iLQR(object):
    def __init__(self, dynamics, cost, Xs, Us):
        self.dynamics_func = dynamics
        self.cost_func = cost
        self.Xs = np.asarray(Xs)
        self.Us = np.asarray(Us)
        self.state_dim = Xs[0].size
        self.control_dim = Us[0].size
        self.T = len(Xs)
        self.dynamics = []
        self.costs = []
        for (x,u) in zip(Xs, Us):
            self.dynamics.append(self.update_dynamics(x, u))
            self.costs.append(self.update_cost(x, u))

    def update_dynamics(self, x, u):
        xu = np.hstack((x,u))
        Jf = ndt.Jacobian(self.dynamics_wrapper)
        J = Jf(xu);
        A = J[:,:self.state_dim]
        B = J[:,self.state_dim:]
        f = self.dynamics_wrapper(xu);
        return TaylorDynamics(A, B, f)

    def update_cost(self, x, u):
        xu = np.hstack((x,u))
        Hf = ndt.Hessian(self.cost_wrapper)
        H = Hf(xu);
        H[np.abs(H) < 1e-11] = 0

        Q = H[:self.state_dim, :self.state_dim]
        P = H[:self.state_dim, self.state_dim:]
        R = H[self.state_dim:,self.state_dim:]

        Q = project_to_psd(Q, 1e-8)
        R = project_to_psd(R, 1e-8)

        gf = ndt.Gradient(self.cost_wrapper)
        g = gf(xu)
        g[np.abs(g) < 1e-11] = 0

        gx = g[:self.state_dim]
        gu = g[self.state_dim:]

        c = self.cost_wrapper(xu)

        return TaylorCost(Q, R, P, gx, gu, c)

    def dynamics_wrapper(self, xu):
        x = xu[:self.state_dim]
        u = xu[self.state_dim:]
        return self.dynamics_func(x, u)

    def cost_wrapper(self, xu):
        x = xu[:self.state_dim]
        u = xu[self.state_dim:]
        return self.cost_func(x, u)

    def forward_pass(self, x0, update=False):
        states = []; controls = []; costs = []
        xt = x0.copy();
        alpha = 1e0
        for t in range(0, self.T):
            z = xt - self.Xs[t] 
            v = self.Ks[t].dot(z) + self.kfs[t]
            ut = alpha*v + self.Us[t]
            cost = self.cost_func(xt, ut)
            states.append(xt) 
            controls.append(ut)
            costs.append(cost)
            xt1 = self.dynamics_func(xt, ut)
            xt = xt1

        if update:
            self.Xs = np.asarray(states)
            self.Us = np.asarray(controls)
            self.dynamics = []; self.costs = [];
            for (x,u) in zip(self.Xs, self.Us):
                self.dynamics.append(self.update_dynamics(x, u))
                self.costs.append(self.update_cost(x, u))
        return states, controls, costs

    def backwards_pass(self):
        Ks = []; kfs = []; Vs = []; Gs = []; Ws = []
        Vt1 = np.zeros((self.state_dim, self.state_dim))
        Gt1 = np.zeros(self.state_dim)
        Wt1 = 0.0 
        for t in xrange(self.T-1, -1, -1):
            Kt, kft, Vt, Gt, Wt = self._compute_backup(Vt1, Gt1, Wt1,
                    self.dynamics[t], self.costs[t])
            Vt = project_to_psd(Vt, 1e-8)
            Ks.insert(0, Kt)
            kfs.insert(0, kft)
            Vs.insert(0, Vt)
            Gs.insert(0, Gt)
            Ws.insert(0, Wt)
            Vt1 = Vt; Gt1 = Gt; Wt1 = Wt
        self.Ks = Ks
        self.kfs = kfs
        self.Vs = Vs
        self.Gs = Gs
        self.Ws = Ws
        
    def _compute_backup(self, Vt1, Gt1, Wt1, dynamics, cost):
        A = dynamics.A; B = dynamics.B; 
        Q = cost.Q; R = cost.R; P = cost.P; gx = cost.gx; gu = cost.gu;

        inv_term = -1.0*np.linalg.inv(R + B.T.dot(Vt1).dot(B))
        K = inv_term.dot(P.T + B.T.dot(Vt1).dot(A))
        kf = inv_term.dot(gu + B.T.dot(Gt1.T))

        tmp = A + B.dot(K)
        Vt = Q + 2.0*P.dot(K) + K.T.dot(R).dot(K) + tmp.T.dot(Vt1).dot(tmp)

        Gt = kf.T.dot(P.T) + kf.T.dot(R).dot(K) + gx.T + gu.T.dot(K) \
                + kf.T.dot(B.T).dot(Vt1).dot(tmp) + Gt1.dot(tmp)

        Wt = 0.5*kf.T.dot(R).dot(kf) + gu.T.dot(kf) + cost.c\
               + Gt1.dot(B).dot(kf) + 0.5*kf.T.dot(B.T.dot(Vt1).dot(B)).dot(kf)\
               + Wt1
    
        return K, kf, Vt, Gt, Wt

def project_to_psd(mat, min_eval):
    eigval, eigvec = np.linalg.eig(mat)
    evaldiag = np.diag(np.maximum(eigval, min_eval))
    return eigvec.dot(evaldiag).dot(eigvec.T)
