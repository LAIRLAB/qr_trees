
#include <experiments/simulators/pendulum.hh>
#include <filters/discrete_filter.hh>
#include <ilqr/iLQR.hh>
#include <ilqr/ilqr_tree.hh>
#include <ilqr/ilqrtree_helpers.hh>
#include <test/helpers.hh>
#include <utils/debug_utils.hh>

#include <algorithm>
#include <memory>
#include <random>

namespace 
{

constexpr int STATE_DIM = 2;
constexpr int CONTROL_DIM = 1;

void construct_hindsight_split_tree(const std::vector<Eigen::VectorXd>& xstars, const std::vector<Eigen::VectorXd>& ustars, 
        const std::vector<double> &probabilities, const std::vector<ilqr::DynamicsFunc> &dynamics_funcs, 
        const ilqr::CostFunc &cost,  ilqr::iLQRTree& ilqr_tree)
{
    const int T = xstars.size();
    IS_GREATER(T, 0);
    IS_EQUAL(xstars.size(), ustars.size());

    const int num_splits = probabilities.size();
    IS_EQUAL(probabilities.size(), dynamics_funcs.size());

    ilqr::TreeNodePtr root_node = ilqr_tree.add_root(xstars[0], ustars[0], dynamics_funcs[0], cost);
    // Split on the next time step to each probabilistic dynamics.
    std::vector<ilqr::TreeNodePtr> last_tree_nodes;
    for (int t = 1; t < T; ++t)
    {
        if (t == 1)
        {
            std::vector<std::shared_ptr<ilqr::iLQRNode>> ilqr_splits(num_splits);
            for (int i = 0; i < num_splits; ++i)
            {
                ilqr_splits[i] = ilqr_tree.make_ilqr_node(xstars[t], ustars[t], dynamics_funcs[i], cost, probabilities[i]);
            }

            last_tree_nodes = ilqr_tree.add_nodes(ilqr_splits, root_node);
        }
        else
        {
            std::vector<ilqr::TreeNodePtr> new_tree_nodes(num_splits);
            for (int i = 0; i < num_splits; ++i)
            {
                std::shared_ptr<ilqr::iLQRNode> child_ilqr = ilqr_tree.make_ilqr_node(xstars[t], ustars[t], dynamics_funcs[i], cost, 1.0);
                new_tree_nodes[i] = ilqr_tree.add_nodes({child_ilqr}, last_tree_nodes[i])[0];
            }
            new_tree_nodes = last_tree_nodes;
        }
    }

}

double compute_total_cost(const ilqr::CostFunc &cost,  
                           std::vector<Eigen::VectorXd> &states, 
                           std::vector<Eigen::VectorXd> &controls)  
{
    IS_EQUAL(states.size(), controls.size());
    const int T = states.size();
    double total_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        total_cost += cost(states[t], controls[t]);
    }
    return total_cost;
}

double weighted_squared_norm(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
{
    return 100.0*(x1-x2).squaredNorm();
}

} // namespace


void control_pendulum_as_chain(const int T, const double dt, const Eigen::VectorXd &x0 = Eigen::VectorXd::Zero(2))
{
    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    // Constants for the pendulum.
    constexpr double LENGTH = 1.0;
    constexpr double DAMPING_COEFF = 0.1;
    IS_EQUAL(x0.size(), STATE_DIM);

    Eigen::VectorXd goal_state(STATE_DIM); 
    goal_state << M_PI, 0;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    Q(1,1) = 1e-5;

    const Eigen::MatrixXd R = 1e-5*Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
    const ilqr::CostFunc quad_cost = create_quadratic_cost(Q, R, goal_state);

    const ilqr::DynamicsFunc dynamics 
        = simulators::pendulum::make_discrete_dynamics_func(dt, LENGTH, DAMPING_COEFF);

    std::vector<Eigen::VectorXd> ilqr_init_states, ilqr_init_controls;
    for (int t = 0; t < T; ++t)
    {
        const Eigen::VectorXd target_state = static_cast<double>(t)/static_cast<double>(T-1) * (goal_state - x0);
        ilqr_init_states.push_back(target_state);
        const Eigen::VectorXd target_control = Eigen::VectorXd::Zero(CONTROL_DIM);
        ilqr_init_controls.push_back(target_control);
    }

    ilqr::iLQRTree ilqr_tree(STATE_DIM, CONTROL_DIM);
    std::vector<ilqr::TreeNodePtr> ilqr_chain = construct_chain(ilqr_init_states, ilqr_init_controls,
        dynamics, quad_cost, ilqr_tree);

    for (int i = 0; i < 10; ++i)
    {
        SUCCESS("i:" << i)
        std::vector<Eigen::VectorXd> ilqr_states, ilqr_controls;
        std::vector<double> ilqr_costs;

        ilqr_tree.bellman_tree_backup();
        ilqr_tree.forward_tree_update(1.0);
        get_forward_pass_info(ilqr_chain, quad_cost, ilqr_states, ilqr_controls, ilqr_costs);

        PRINT(" x(" << T-1 << ")= " << ilqr_states[T-1].transpose());
        WARN(" Total cost: " << compute_total_cost(quad_cost, ilqr_states, ilqr_controls));
    }
}

void discrete_damping_coeff_pendulum(const int T, const double dt, const Eigen::VectorXd &x0 = Eigen::VectorXd::Zero(2))
{

    std::mt19937 gen(1);

    constexpr double LENGTH = 1.0;
    Eigen::VectorXd goal_state(STATE_DIM); 
    goal_state << M_PI, 0;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    Q(1,1) = 1e-5;
    const Eigen::MatrixXd R = 1e-5*Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
    const ilqr::CostFunc quad_cost = create_quadratic_cost(Q, R, goal_state);


    const std::vector<double> damping_coeffs = {0.0};

    std::unordered_map<double, double> uniform_prior;
    for (const auto &d: damping_coeffs)
    {
        uniform_prior.emplace(d, 1.0/damping_coeffs.size());
    }

    filters::DiscreteFilter<double> filter(uniform_prior);
    std::uniform_int_distribution<> dis(0, damping_coeffs.size()-1);
    const double true_damping_coeff = damping_coeffs[dis(gen)];

    Eigen::VectorXd xt = x0;
    Eigen::VectorXd ut = Eigen::VectorXd::Random(CONTROL_DIM);
    filters::DiscreteFilter<double>::ObsFunc obs_func = [&xt, &ut, dt](double damping_coeff)
        {
            auto dynamics = simulators::pendulum::make_discrete_dynamics_func(dt, LENGTH, damping_coeff);
            return dynamics(xt, ut);
        };   
    
    auto true_dynamics = simulators::pendulum::make_discrete_dynamics_func(dt, LENGTH, true_damping_coeff);

    std::vector<double> probabilities;
    std::vector<ilqr::DynamicsFunc> dynamics_funcs;
    auto beliefs = filter.beliefs();
    for (const auto &belief : beliefs)
    {
        probabilities.push_back(belief.second);
        dynamics_funcs.push_back(simulators::pendulum::make_discrete_dynamics_func(dt, LENGTH, belief.first));
    }


    // Run the filter and control policy.
    for (int t = 0; t < T; ++t)
    {
        beliefs = filter.beliefs();
        std::transform(beliefs.begin(), beliefs.end(), probabilities.begin(), 
                [](const std::pair<double, double> &pair) { return pair.second; });

        PRINT(" t=" << t << ", b= " <<  filter);
        std::vector<Eigen::VectorXd> xstars, ustars;
        for (int t = 0; t < T; ++t)
        {
            const Eigen::VectorXd target_state = static_cast<double>(t)/static_cast<double>(T-1) * (goal_state - xt);
            xstars.push_back(target_state);
            const Eigen::VectorXd target_control = Eigen::VectorXd::Zero(CONTROL_DIM);
            ustars.push_back(target_control);
        }

        //ilqr::iLQRTree ilqr_tree(STATE_DIM, CONTROL_DIM);
        //construct_hindsight_split_tree(xstars, ustars, probabilities, dynamics_funcs, quad_cost, ilqr_tree); 
        ilqr::iLQRTree ilqr_tree(STATE_DIM, CONTROL_DIM);
        std::vector<ilqr::TreeNodePtr> ilqr_chain = construct_chain(xstars, ustars,
            true_dynamics, quad_cost, ilqr_tree);
        for (int i = 0; i < 8; ++i)
        {
            ilqr_tree.bellman_tree_backup();
            ilqr_tree.forward_tree_update(0.5);
        }
        std::shared_ptr<ilqr::iLQRNode> ilqr_root = ilqr_tree.root()->item();
        ut = ilqr_root->compute_control(xt);

        Eigen::VectorXd xt1 = true_dynamics(xt, ut);

        // Full state observation model.
        Eigen::VectorXd zt1 = xt1; 

        filter.update(zt1, obs_func, weighted_squared_norm);
        xt = xt1;
    }

    PRINT(" x(" << T-1 << ")= " << xt.transpose());
}

int main()
{
    //control_pendulum_as_chain(20, 0.1);
    discrete_damping_coeff_pendulum(20, 0.1);
    return 0;
}


