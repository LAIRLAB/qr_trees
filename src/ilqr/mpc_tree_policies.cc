
#include <ilqr/mpc_tree_policies.hh>
#include <ilqr/ilqrtree_helpers.hh>
#include <utils/debug_utils.hh>
#include <utils/helpers.hh>


namespace
{
    // Optimizes the ilqr_tree and returns the first control action from the root.
    Eigen::VectorXd optimize_tree(ilqr::iLQRTree &ilqr_tree, const Eigen::VectorXd &x0)
    {
        for (int i = 0; i < 10; ++i)
        {
            ilqr_tree.bellman_tree_backup();
            ilqr_tree.forward_tree_update(0.4);
        }
        const std::shared_ptr<ilqr::iLQRNode> ilqr_root = ilqr_tree.root()->item();
        return ilqr_root->compute_control(x0);
    }
}

namespace policy
{

Eigen::VectorXd hindsight_tree_policy(const int t, 
        const Eigen::VectorXd& xt, 
        const int T,
        const Eigen::VectorXd& xT, 
        const Eigen::VectorXd& nominal_control,
        const std::vector<double> &probabilities, 
        const std::vector<ilqr::DynamicsFunc> &dynamics_funcs, 
        const ilqr::CostFunc &cost,
        ilqr::iLQRTree& ilqr_tree)

{
    const std::vector<Eigen::VectorXd> xstars 
        = linearly_interpolate(t, xt, T, xT);
    const std::vector<Eigen::VectorXd> ustars(T-t, nominal_control);

    ilqr::construct_hindsight_split_tree(xstars, ustars, probabilities, 
            dynamics_funcs, cost, ilqr_tree); 
    return optimize_tree(ilqr_tree, xt);
}

Eigen::VectorXd hindsight_tree_policy(const int t,
                                      const Eigen::VectorXd& xt, 
                                      const int T,
                                      const Eigen::VectorXd& xT, 
                                      const Eigen::VectorXd& nominal_control,
                                      const std::vector<double> &probabilities, 
                                      const ilqr::DynamicsFunc &dynamics_func,
                                      const std::vector<ilqr::CostFunc> &cost_funcs, 
                                      const std::vector<ilqr::CostFunc> &final_cost_funcs,
                                      ilqr::iLQRTree& ilqr_tree
                                      )
{
    const std::vector<Eigen::VectorXd> xstars 
        = linearly_interpolate(t, xt, T, xT);
    const std::vector<Eigen::VectorXd> ustars(T-t, nominal_control);

    ilqr::construct_hindsight_split_tree(xstars, ustars, probabilities, 
            dynamics_func, cost_funcs, final_cost_funcs, ilqr_tree); 
    return optimize_tree(ilqr_tree, xt);

}

Eigen::VectorXd probability_weighted_policy(const int t, 
        const Eigen::VectorXd& xt, 
        const int T,
        const Eigen::VectorXd& xT, 
        const Eigen::VectorXd& nominal_control,
        const std::vector<double> &probabilities, 
        const std::vector<ilqr::DynamicsFunc> &dynamics_funcs, 
        const ilqr::CostFunc &cost 
        )
{
    IS_GREATER(xt.size(), 0);
    IS_EQUAL(xt.size(), xT.size());
    const std::vector<Eigen::VectorXd> xstars 
        = linearly_interpolate(t, xt, T, xT);
    const std::vector<Eigen::VectorXd> ustars(T-t, nominal_control);

    const int state_dim = xt.size();
    const int control_dim = nominal_control.size();

    const int num_splits = probabilities.size();
    Eigen::VectorXd ut = Eigen::VectorXd::Zero(control_dim);
    for (int i = 0; i < num_splits; ++i)
    {
        ilqr::iLQRTree ilqr_tree(state_dim, control_dim);
        //TODO: support final_cost parameter.
        ilqr::construct_chain(xstars, ustars, dynamics_funcs[i], cost, cost, ilqr_tree);
        ut += probabilities[i] * optimize_tree(ilqr_tree, xt);
    }
    return ut;
}

Eigen::VectorXd probability_weighted_policy(const int t, 
        const Eigen::VectorXd& xt, 
        const int T,
        const Eigen::VectorXd& xT, 
        const Eigen::VectorXd& nominal_control,
        const std::vector<double> &probabilities, 
        const ilqr::DynamicsFunc &dynamics_func, 
        const std::vector<ilqr::CostFunc> &cost_funcs,
        const std::vector<ilqr::CostFunc> &final_cost_funcs
        )
{
    IS_GREATER(xt.size(), 0);
    IS_EQUAL(xt.size(), xT.size());
    const std::vector<Eigen::VectorXd> xstars 
        = linearly_interpolate(t, xt, T, xT);
    const std::vector<Eigen::VectorXd> ustars(T-t, nominal_control);

    const int state_dim = xt.size();
    const int control_dim = nominal_control.size();

    const int num_splits = probabilities.size();
    Eigen::VectorXd ut = Eigen::VectorXd::Zero(control_dim);
    for (int i = 0; i < num_splits; ++i)
    {
        ilqr::iLQRTree ilqr_tree(state_dim, control_dim);
        const ilqr::CostFunc &final_cost_func = final_cost_funcs.empty() ? cost_funcs[i] : final_cost_funcs[i];
        ilqr::construct_chain(xstars, ustars, dynamics_func, cost_funcs[i], final_cost_func, ilqr_tree);
        ut += probabilities[i] * optimize_tree(ilqr_tree, xt);
    }
    return ut;
}


Eigen::VectorXd chain_policy(const int t, 
        const Eigen::VectorXd& xt, 
        const int T,
        const Eigen::VectorXd& xT, 
        const Eigen::VectorXd& nominal_control,
        const ilqr::DynamicsFunc &dynamics, 
        const ilqr::CostFunc &cost,
        const ilqr::CostFunc &final_cost,
        ilqr::iLQRTree& ilqr_tree)
{
    IS_GREATER(xt.size(), 0);
    IS_EQUAL(xt.size(), xT.size());
    const std::vector<Eigen::VectorXd> xstars 
        = linearly_interpolate(t, xt, T, xT);
    const std::vector<Eigen::VectorXd> ustars(T-t, nominal_control);
    ilqr::construct_chain(xstars, ustars, dynamics, cost, final_cost, ilqr_tree);
    return optimize_tree(ilqr_tree, xt);
}

} // namespace policy
