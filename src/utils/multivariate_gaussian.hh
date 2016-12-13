//
// Multivariate Guassian Class.
// Original from Humphrey Hu 
// (https://github.com/Humhu/argus_utils/blob/devel/include/argus_utils/random/MultivariateGaussian.hpp)
// Adapted by Arun Venkatraman
//

#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <cassert>
#include <random>

namespace math
{

class MultivariateGaussian 
{
public:

	// Seeds the engine using a true random number. Sets mean_ to zero
	// and covariance to identity. 
	MultivariateGaussian(const int dim);

	// Seeds the engine using the specified seed. Sets mean_ to zero
	// and _covariance to identity.
	MultivariateGaussian(const int dim, const int seed);
	
	// Sets the mean to mu and covariance to sigma.
	MultivariateGaussian(const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma);
    
	// Sets the mean to mu and covariance to sigma. Seeds the engine using a specified seed. 
	MultivariateGaussian(const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma, const int seed);

	MultivariateGaussian(const MultivariateGaussian& other);
	MultivariateGaussian& operator=(const MultivariateGaussian& other);
    
    void set_mean(const Eigen::VectorXd& mu);
    void set_covariance(const Eigen::MatrixXd& sigma);
	void set_information(const Eigen::MatrixXd& inv_sigma);

	unsigned int get_dimension() const { return mean_.size(); }
	const Eigen::VectorXd& mean() const { return mean_; }
	const Eigen::MatrixXd covariance() const { return llt_.reconstructedMatrix(); }
	const Eigen::MatrixXd cholesky() const { return llt_.matrixL(); }
	
	// Generate a sample truncated at a specified number of standard deviations.
	Eigen::VectorXd sample(double max_var = 3.0);

	// Evaluate the multivariate normal PDF for the specified sample.
    double evaluate_probability(const Eigen::VectorXd& x) const;
    
protected:
	
    std::mt19937 gen_;
	std::normal_distribution<double> distribution_;

    // Mean of the distribution.
	Eigen::VectorXd mean_;
    // Store covariance as its cholesky decomposition.
	Eigen::LLT<Eigen::MatrixXd> llt_;

    // Normalization constant;
	double z_; 

	void initialize_cov(const Eigen::MatrixXd& cov);
};

}
