
#include <utils/debug_utils.hh>
#include <utils/helpers.hh>
#include <utils/math_utils.hh>
#include <utils/multivariate_gaussian.hh>

namespace
{

constexpr double WEAK_TOL = 1e-1;

template<typename C>
// Implements the simple two-pass algorithm for computing the sample mean and covariance.
void compute_sample_mean_cov(const C &container, Eigen::VectorXd &sample_mean, Eigen::MatrixXd &sample_cov)
{
    // Make sure the container is larger than 0 size.
    const int num_samples = container.size();
    IS_GREATER(num_samples, 0);
    const int sample_dim = container.begin()->size();
    sample_mean = Eigen::VectorXd::Zero(sample_dim);
    for (const Eigen::VectorXd& sample : container)
    {
       sample_mean += sample; 
    }
    sample_mean /= static_cast<double>(num_samples);


    sample_cov = Eigen::MatrixXd::Zero(sample_dim, sample_dim);
    for (const Eigen::VectorXd& sample : container)
    {
       const Eigen::VectorXd diff = sample - sample_mean; 
       sample_cov += diff*diff.transpose();
    }
    sample_cov /= static_cast<double>(num_samples-1);
}

} // namespace


void test_1d_gaussian()
{
    constexpr int SEED = 1;
    std::srand(SEED);
    constexpr int dim = 1;

    constexpr int MANY_SAMPLES = 5e4;

    // Draw many samples and confirm zero mean and identity covariance 
    // and confirm sample mean and covariance is as expected.
    math::MultivariateGaussian gaussian(dim, SEED);
    std::array<Eigen::VectorXd, MANY_SAMPLES> samples;
    for (int i = 0; i < MANY_SAMPLES; ++i)
    {
        samples[i] = gaussian.sample(10);
    }

    Eigen::VectorXd sample_mean;
    Eigen::MatrixXd sample_cov;
    compute_sample_mean_cov(samples, sample_mean, sample_cov);
    IS_TRUE(math::is_equal(sample_mean, Eigen::VectorXd::Zero(dim), WEAK_TOL));
    IS_TRUE(math::is_equal(sample_cov, Eigen::MatrixXd::Identity(dim,dim), WEAK_TOL));

    Eigen::VectorXd true_mean = Eigen::VectorXd::Random(dim);
    Eigen::MatrixXd true_cov = make_random_psd(dim, 1e-3);
    gaussian = math::MultivariateGaussian(true_mean, true_cov, SEED);
    for (int i = 0; i < MANY_SAMPLES; ++i)
    {
        samples[i] = gaussian.sample(10);
    }
    compute_sample_mean_cov(samples, sample_mean, sample_cov);
    IS_TRUE(math::is_equal(sample_mean, true_mean, WEAK_TOL));
    IS_TRUE(math::is_equal(sample_cov, true_cov, WEAK_TOL));
}

void test_nd_gaussian(const int dim)
{
    constexpr int SEED = 1;
    std::srand(SEED);

    // Add more samples based on number of dimensions.
    const int num_samples = 5e4 + 1e3*(dim-1);

    // Draw many samples and confirm zero mean and identity covariance 
    // and confirm sample mean and covariance is as expected.
    math::MultivariateGaussian gaussian(dim, SEED);
    std::vector<Eigen::VectorXd> samples(num_samples);
    for (int i = 0; i < num_samples; ++i)
    {
        samples[i] = gaussian.sample(10);
    }

    Eigen::VectorXd sample_mean;
    Eigen::MatrixXd sample_cov;
    compute_sample_mean_cov(samples, sample_mean, sample_cov);
    IS_TRUE(math::is_equal(sample_mean, Eigen::VectorXd::Zero(dim), WEAK_TOL));
    IS_TRUE(math::is_equal(sample_cov, Eigen::MatrixXd::Identity(dim,dim), WEAK_TOL));

    Eigen::VectorXd true_mean = Eigen::VectorXd::Random(dim);
    Eigen::MatrixXd true_cov = make_random_psd(dim, 1e-3);
    gaussian = math::MultivariateGaussian(true_mean, true_cov, SEED);
    for (int i = 0; i < num_samples; ++i)
    {
        samples[i] = gaussian.sample(10);
    }
    compute_sample_mean_cov(samples, sample_mean, sample_cov);
    //WARN("sample mu: "  << sample_mean.transpose());
    //WARN("sample cov: " << sample_cov);
    //WARN("true mu: "  << true_mean.transpose());
    //WARN("true cov: " << true_cov);
    IS_TRUE(math::is_equal(sample_mean, true_mean, WEAK_TOL));
    IS_TRUE(math::is_equal(sample_cov, true_cov, WEAK_TOL));
}

int main()
{
    test_1d_gaussian();

    DOES_THROW(test_nd_gaussian(0));
    test_nd_gaussian(1);
    test_nd_gaussian(2);
    test_nd_gaussian(3);
    test_nd_gaussian(5);

    return 0;
}
