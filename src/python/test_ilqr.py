#!/usr/bin/env python

from ilqr import iLQR, project_to_psd
from lqr import LQR

import numpy as np

from IPython import embed

def linear_dynamics(A, B):
    return lambda x,u : A.dot(x) + B.dot(u)

def quadtratic_cost(Q, R):
    return lambda x,u : 0.5*(x.T.dot(Q).dot(x) + u.T.dot(R).dot(u))

def make_symmetric_random_psd(dim, min_eval=1e-8):
    mat = np.random.random((dim, dim))
    mat = 0.5 * (mat.T + mat) 
    return project_to_psd(mat, min_eval)


if __name__ == "__main__":
    state_dim = 3
    control_dim = 2

    np.random.seed(1)

    #A = np.random.random((state_dim, state_dim))
    #B = np.random.random((state_dim, control_dim))
    A = np.identity(state_dim)
    A[0,1]= 1;
    A[2,1]= 2;
    B = np.eye(state_dim, control_dim)
    B[0, 1] = 1;

    #Q = make_symmetric_random_psd(state_dim)
    #R = make_symmetric_random_psd(control_dim)
    Q = np.identity(state_dim)
    R = np.identity(control_dim)


    dyn_func = linear_dynamics(A, B);
    cost_func = quadtratic_cost(Q, R);

    T = 4
    x0 = np.asarray((3, 2, 1))

    lqr = LQR(A, B, Q, R, T)
    lqr.solve()
    lqr_states, lqr_controls, lqr_costs = lqr.forward_pass(x0)

    #Us2 = np.random.random((T, control_dim))
    Us2 = np.ones((T, control_dim))
    #Us2 = np.asarray(lqr_controls) 
    xt = x0.copy(); Xs2 = xt;
    for t in range(0, T-1):
        xt1 = dyn_func(xt, Us2[t])
        Xs2 = np.vstack((Xs2, xt1))
        xt = xt1 
    #Xs = lqr_states; Us = lqr_controls;
    Xs = Xs2; Us = Us2;
    original_ilqr_cost = np.sum([cost_func(x, u) for (x,u) in zip(Xs, Us)])

    ilqr = iLQR(dyn_func, cost_func, Xs, Us);

    for _ in range(1):
        ilqr.backwards_pass()
        ilqr_states, ilqr_controls, ilqr_costs = ilqr.forward_pass(x0, update=True)

    for t in range(T):
        lqr_x = lqr_states[t]
        lqr_u = lqr_controls[t]
        ilqr_x = ilqr_states[t]
        ilqr_u = ilqr_controls[t]
        print("t={}, xlqr:  {}".format(t, lqr_x))
        print("  {}, xilqr: {} ".format(t, ilqr_x));
        print("  {}, xilqr_orig: {} ".format(t, Xs[t]));
        print("  {}, ulqr:  {}".format(t, lqr_u))
        print("  {}, uilqr: {} ".format(t, ilqr_u));
        print("  {}, uilqr_orig: {} ".format(t, Us[t]));
    lqr_cost = np.sum(lqr_costs)
    ilqr_cost = np.sum(ilqr_costs)
    print("LQR total cost: {}".format(lqr_cost))
    print("iLQR total cost: {}".format(ilqr_cost))
    print("original iLQR total cost: {}".format(original_ilqr_cost))

