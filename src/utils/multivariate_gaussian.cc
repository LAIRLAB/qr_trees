//
// Multivariate Guassian Noise
// Original from Humphrey Hu 
// (https://github.com/Humhu/argus_utils/blob/devel/include/argus_utils/random/MultivariateGaussian.hpp)
// Adapted by Arun Venkatraman
//

#include <utils/multivariate_gaussian.hh>

#include <utils/debug_utils.hh>

namespace math
{

MultivariateGaussian::MultivariateGaussian(const int dim)
    : MultivariateGaussian(Eigen::VectorXd::Zero(dim), Eigen::MatrixXd::Identity(dim, dim))
{
}

MultivariateGaussian::MultivariateGaussian(const int dim, const int seed)
    : MultivariateGaussian(Eigen::VectorXd::Zero(dim), Eigen::MatrixXd::Identity(dim, dim), seed)
{
}

MultivariateGaussian::MultivariateGaussian(const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma)
    : distribution_(0.0, 1.0), 
      mean_(mu)
{
    IS_GREATER(mu.size(), 0);
    std::random_device rng;
    gen_.seed(rng());
    initialize_cov(sigma);
}

MultivariateGaussian::MultivariateGaussian(const Eigen::VectorXd& mu, 
        const Eigen::MatrixXd& sigma, const int seed)
    : distribution_(0.0, 1.0),
      mean_(mu)
{
    IS_GREATER(mu.size(), 0);
    gen_.seed(seed);
    initialize_cov(sigma);
}

MultivariateGaussian::MultivariateGaussian(const MultivariateGaussian& other)
    : gen_(other.gen_),
      distribution_(0.0, 1.0),
      mean_(other.mean_)
{
}

MultivariateGaussian& MultivariateGaussian::operator=(const MultivariateGaussian& other)
{
    mean_ = other.mean_;
    z_ = other.z_;
    llt_ = other.llt_;
    gen_ = other.gen_;
    return *this;
}

void MultivariateGaussian::set_mean(const Eigen::VectorXd& mu) 
{ 
    IS_EQUAL(mu.size(), mean_.size());
    mean_ = mu; 
}
void MultivariateGaussian::set_covariance(const Eigen::MatrixXd& sigma)
{
    IS_EQUAL(sigma.rows(), llt_.matrixL().rows());
    IS_EQUAL(sigma.cols(), llt_.matrixL().cols());
    initialize_cov(sigma);
}

void MultivariateGaussian::set_information(const Eigen::MatrixXd& inv_sigma)
{
    IS_EQUAL(inv_sigma.rows(), llt_.matrixL().rows()); 
    IS_EQUAL(inv_sigma.cols(), llt_.matrixL().cols());

    Eigen::LDLT<Eigen::MatrixXd> llti(inv_sigma);
    Eigen::MatrixXd cov = llti.solve(Eigen::MatrixXd::Identity(inv_sigma.rows(), inv_sigma.cols()));
    initialize_cov(cov);
}

Eigen::VectorXd MultivariateGaussian::sample(double max_var)
{
    Eigen::VectorXd samples(mean_.size());
    for(int i = 0; i < mean_.size(); i++)
    {
        double s;
        do
        {
            s = distribution_(gen_); 
        } while(std::abs(s) > max_var);
        samples(i) = s;
    }
    
    return mean_ + llt_.matrixL()*samples;
}

/*! \brief Evaluate the multivariate normal PDF for the specified sample. */
double MultivariateGaussian::evaluate_probability(const Eigen::VectorXd& x) const
{
    IS_EQUAL(x.size(), mean_.size());

    Eigen::VectorXd diff = x - mean_;
    Eigen::MatrixXd exponent = -0.5 * diff.transpose() * llt_.solve(diff);
    return z_ * std::exp(exponent(0));
}


void MultivariateGaussian::initialize_cov(const Eigen::MatrixXd& cov)
{
    IS_EQUAL(mean_.size(), cov.rows());
    IS_EQUAL(mean_.size(), cov.cols());

    llt_ = Eigen::LLT<Eigen::MatrixXd>(cov);
    z_ = std::pow(2*M_PI, -static_cast<double>(mean_.size())*0.5)
         * std::pow(cov.determinant(), -0.5);
}

} // namespace math
