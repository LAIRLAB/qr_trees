
#include <experiments/simulators/pendulum.hh>
#include <filters/discrete_filter.hh>
#include <ilqr/iLQR.hh>
#include <ilqr/ilqr_tree.hh>
#include <ilqr/ilqrtree_helpers.hh>
#include <utils/debug_utils.hh>
#include <utils/helpers.hh>
#include <utils/math_utils.hh>

#include <algorithm>
#include <memory>
#include <random>

namespace 
{

constexpr int STATE_DIM = 2;
constexpr int CONTROL_DIM = 1;

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
    return 10.0*(x1-x2).squaredNorm();
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

    // Initial linearization points are linearly interpolated states and zero
    // control.
    std::vector<Eigen::VectorXd> ilqr_init_states 
        = linearly_interpolate(0, x0, T, goal_state);
    std::vector<Eigen::VectorXd> ilqr_init_controls(T, Eigen::VectorXd::Zero(CONTROL_DIM));

    ilqr::iLQRTree ilqr_tree(STATE_DIM, CONTROL_DIM);
    std::vector<ilqr::TreeNodePtr> ilqr_chain 
        = construct_chain(ilqr_init_states, ilqr_init_controls,
                          dynamics, quad_cost, ilqr_tree);


    std::vector<Eigen::VectorXd> ilqr_states, ilqr_controls;
    double total_cost = 0;
    for (int i = 0; i < 10; ++i)
    {
        SUCCESS("i:" << i)
        std::vector<double> ilqr_costs;

        ilqr_tree.bellman_tree_backup();
        ilqr_tree.forward_tree_update(1.0);
        get_forward_pass_info(ilqr_chain, quad_cost, ilqr_states, ilqr_controls, ilqr_costs);

        PRINT(" x(" << T-1 << ")= " << ilqr_states[T-1].transpose());
        total_cost = compute_total_cost(quad_cost, ilqr_states, ilqr_controls);
        WARN(" Total cost: " << total_cost);
    }

    constexpr double TOL = 1e-4;
    // Run the control policy.
    Eigen::VectorXd xt = x0;
    double rollout_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        IS_TRUE(math::is_equal(ilqr_states[t], xt, TOL));

        const Eigen::VectorXd ut = ilqr_chain[t]->item()->compute_control(xt);

        IS_TRUE(math::is_equal(ilqr_controls[t], ut, TOL));

        rollout_cost += quad_cost(xt, ut);

        const Eigen::VectorXd xt1 = dynamics(xt, ut);

        xt = xt1;
    }
    PRINT(" x_rollout(" << T-1 << ")= " << xt.transpose());
    WARN(" Total cost rollout: " << rollout_cost);
    IS_ALMOST_EQUAL(total_cost, rollout_cost, TOL);
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


    const std::vector<double> damping_coeffs = {0.0, 0.3, 0.5};

    std::unordered_map<double, double> uniform_prior;
    for (const auto &d: damping_coeffs)
    {
        uniform_prior.emplace(d, 1.0/damping_coeffs.size());
    }

    filters::DiscreteFilter<double> filter(uniform_prior);
    std::uniform_int_distribution<> dis(0, damping_coeffs.size()-1);
    const double true_damping_coeff = damping_coeffs[dis(gen)];
    SUCCESS("True Damping Coeff: " << true_damping_coeff);

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
    double rollout_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        beliefs = filter.beliefs();
        std::transform(beliefs.begin(), beliefs.end(), probabilities.begin(), 
                [](const std::pair<double, double> &pair) { return pair.second; });

        PRINT(" t=" << t << ", b= " <<  filter);
        // Create linearly interpolated target points for initial linearization.
        std::vector<Eigen::VectorXd> xstars 
            = linearly_interpolate(t, xt, T, goal_state);
        std::vector<Eigen::VectorXd> ustars(T-t, 
                Eigen::VectorXd::Zero(CONTROL_DIM));

        ilqr::iLQRTree ilqr_tree(STATE_DIM, CONTROL_DIM);
        ilqr::construct_hindsight_split_tree(xstars, ustars, probabilities, 
                dynamics_funcs, quad_cost, ilqr_tree); 

        // Compute and iLQR chain using the argmax(probabilities) dynamics.
        // This should perfectly when there is no noise in the observations.
        
        // const int arg_max_prob = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
        // const auto arg_max_dynamics = dynamics_funcs[arg_max_prob];
        //std::vector<ilqr::TreeNodePtr> ilqr_chain = construct_chain(xstars, ustars, arg_max_dynamics, quad_cost, ilqr_tree);
        
        for (int i = 0; i < 10; ++i)
        {
            ilqr_tree.bellman_tree_backup();
            ilqr_tree.forward_tree_update(0.5);
        }
        const std::shared_ptr<ilqr::iLQRNode> ilqr_root = ilqr_tree.root()->item();
        ut = ilqr_root->compute_control(xt);

        rollout_cost += quad_cost(xt, ut);

        const Eigen::VectorXd xt1 = true_dynamics(xt, ut);

        // Full state observation model.
        const Eigen::VectorXd zt1 = xt1; 

        filter.update(zt1, obs_func, weighted_squared_norm);
        xt = xt1;
    }

    PRINT(" x(" << T-1 << ")= " << xt.transpose());
    WARN(" Total cost rollout: " << rollout_cost);
}

int main()
{
    //control_pendulum_as_chain(20, 0.1);
    discrete_damping_coeff_pendulum(20, 0.1);
    return 0;
}


