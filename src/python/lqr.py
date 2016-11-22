
import numpy as np

from IPython import embed

class LQR(object):
    def __init__(self, A, B, Q, R, T):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.T =T
        self.state_dim = A.shape[0]
        self.control_dim = B.shape[1]

    def _compute_backup(self, Vt1):
        inv_term = -1.0*np.linalg.inv(self.R + self.B.T.dot(Vt1).dot(self.B))
        Kt = inv_term.dot(self.B.transpose().dot(Vt1).dot(self.A));
        tmp = self.A + self.B.dot(Kt)
        Vt = self.Q + Kt.T.dot(self.R).dot(Kt)\
            + tmp.T.dot(Vt1).dot(tmp)
        return Kt, Vt

    def solve(self):
        Ks = []; Vs = []
        Vt1 = np.zeros((self.state_dim, self.state_dim))
        for t in xrange(self.T-1, -1, -1):
            Kt, Vt = self._compute_backup(Vt1)
            Ks.insert(0, Kt)
            Vs.insert(0, Vt)
            Vt1 = Vt
        self.Ks = Ks
        self.Vs = Vs

    def forward_pass(self, x0):
        states = []; controls = []; costs = []
        xt = x0.copy();
        for t in range(0, self.T):
            ut = self.Ks[t].dot(xt);
            cost = self.cost_func(xt, ut)
            states.append(xt) 
            controls.append(ut)
            costs.append(cost)
            xt = self.dynamics_func(xt, ut)
        return states, controls, costs

    def cost_func(self, xt, ut):
        cost = xt.T.dot(self.Q).dot(xt) + ut.T.dot(self.R).dot(ut)
        return 0.5*cost

    def dynamics_func(self, xt, ut):
        xt1 = self.A.dot(xt) + self.B.dot(ut);
        return xt1

