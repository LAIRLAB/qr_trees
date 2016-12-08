//
// Multivariate Guassian Noise
// Original from Humphrey Hu 
// (https://github.com/Humhu/argus_utils/blob/devel/include/argus_utils/random/MultivariateGaussian.hpp)
// Adapted by Arun Venkatraman
//

#pragma once

#include <utils/debug_utils.hh>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <cassert>
#include <random>

namespace math
{


using MatrixType = Eigen::MatrixXd;
using VectorType = Eigen::VectorXd;

// TODO
inline
double GaussianPDF( const MatrixType& cov, const VectorType& x )
{
	Eigen::LDLT<MatrixType> ldlt( cov );
	MatrixType exponent = -0.5 * x.transpose() * ldlt.solve( x );
	double det = ldlt.vectorD().array().prod();
	double z = 1.0 / ( std::pow( 2*M_PI, x.size()/2.0 )
		               * std::sqrt( det ) );
	return z * std::exp( exponent(0) );
}

/*! \brief Multivariate normal sampling and PDF class. */
class MultivariateGaussian 
{
public:

	using UnivariateNormal = std::normal_distribution<double>;

	/*! \brief Seeds the engine using a true random number. Sets mean_ to z_ero
	 * and _covariance to identity. */
	MultivariateGaussian( unsigned int dim = 1 )
	: distribution_( 0.0, 1.0 ), 
	  mean_( VectorType::Zero( dim ) )
	{
		std::random_device rng;
		generator_.seed( rng );
		initialize_cov( MatrixType::Identity( dim, dim ) );
	}
	
	/*! \brief Seeds the engine using the specified seed. Sets mean_ to z_ero
	 * and _covariance to identity. */
	MultivariateGaussian( unsigned int dim, unsigned long seed )
	: distribution_( 0.0, 1.0 ), 
	  mean_( VectorType::Zero( dim ) )
	{
		generator_.seed( seed );
		initialize_cov( MatrixType::Identity( dim, dim ) );
	}
	
	/*! \brief Seeds the engine using a true random number. */
	MultivariateGaussian( const VectorType& u, const MatrixType& S )
	: distribution_( 0.0, 1.0 ), 
	  mean_( u )
	{
		std::random_device rng;
		generator_.seed( rng );
		initialize_cov( S );
	}
    
	/*! \brief Seeds the engine using a specified seed. */
	MultivariateGaussian( const VectorType& u, const MatrixType& S, unsigned long seed )
	: distribution_( 0.0, 1.0 ), 
	  mean_( u )
	{
		generator_.seed( seed );
		initialize_cov( S );
	}

	MultivariateGaussian( const MultivariateGaussian& other )
	: generator_( other.generator_ ),
	  distribution_( 0.0, 1.0 ),
	  mean_( other.mean_ )
	{}

	MultivariateGaussian& operator=( const MultivariateGaussian& other )
	{
		mean_ = other.mean_;
		z_ = other.z_;
		llt_ = other.llt_;
		generator_ = other.generator_;
		return *this;
	}
    
    void set_mean( const VectorType& u ) 
    { 
    	if( u.size() != mean_.size() )
    	{
    		throw std::runtime_error( "MultivariateGaussian: Invalid mean dimension." );
    	}
    	mean_ = u; 
    }
    void set_covariance( const MatrixType& S )
	{
		if( S.rows() != llt_.matrixL().rows() || S.cols() != llt_.matrixL().cols() )
		{
			throw std::runtime_error( "MultivariateGaussian: Invalid covariance dimensions." );
		}
		initialize_cov( S );
	}

	void set_information( const MatrixType& I )
	{
		if( I.rows() != llt_.matrixL().rows() || I.cols() != llt_.matrixL().cols() )
		{
			throw std::runtime_error( "MultivariateGaussian: Invalid information dimensions." );
		}
		Eigen::LDLT<MatrixType> llti( I );
		MatrixType cov = llti.solve( MatrixType::Identity( I.rows(), I.cols() ) );
		initialize_cov( cov );
	}

	unsigned int get_dimension() const { return mean_.size(); }
	const VectorType& mean() const { return mean_; }
	const MatrixType covariance() const { return llt_.reconstructedMatrix(); }
	const MatrixType cholesky() const { return llt_.matrixL(); }
	
	/*! \brief Generate a sample truncated at a specified number of standard deviations. */
	VectorType sample( double v = 3.0 )
	{
		VectorType samples( mean_.size() );
		for( unsigned int i = 0; i < mean_.size(); i++ )
		{
			double s;
			do
			{
				s = distribution_(generator_); 
			}
			while( std::abs( s ) > v );
			samples(i) = s;
		}
		
		return mean_ + llt_.matrixL()*samples;
	}

	/*! \brief Evaluate the multivariate normal PDF for the specified sample. */
    double evaluate_probability( const VectorType& x ) const
	{
		if( x.size() != mean_.size() )
		{
			throw std::runtime_error( "MultivariateGaussian: Invalid sample dimension." );
		}
		VectorType diff = x - mean_;
		MatrixType exponent = -0.5 * diff.transpose() * llt_.solve( diff );
		return z_ * std::exp( exponent(0) );
	}
    
protected:
	
    std::mt19937 generator_;
	UnivariateNormal distribution_;

	VectorType mean_;
	double z_; // Normalization constant;
	Eigen::LLT<MatrixType> llt_;

	void initialize_cov(const MatrixType& cov )
	{
		if( mean_.size() != cov.rows() || 
		    mean_.size() != cov.cols() )
		{
			throw std::runtime_error( "MultivariateGaussian: mean and covariance dimension mismatch." );
		}

		llt_ = Eigen::LLT<MatrixType>( cov );
		z_ = std::pow( 2*M_PI, -mean_.size()/2.0 )
		     * std::pow( cov.determinant(), -0.5 );
	}
};

}
