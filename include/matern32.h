#ifndef _GPSTATE_MATERN32_H
#define _GPSTATE_MATERN32_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>

#include "KF.h"

// implements a class for solving a Matern 3/2
// calculates Q, Phi, H and R matrices as detailed in Jordan et al


using namespace KF;

namespace gpstate {
namespace matern32 {

class Matern32Solver {
public:
Matern32Solver (const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr, 
	const double& l, const double& varf)
{
	set_pars(l, varf);
	set_data(times, y_i, yerr);
};

void get_varw(const double& varf, const double& ll, double& varw, double& lambda)
{
	double nu = 1.5;
	double lam_pre = std::sqrt(2*nu);
	double var_factor = 2.0 * std::sqrt(M_PI) * std::tgamma( nu+0.5 ) / std::tgamma( nu );
	lambda = lam_pre / ll;
	varw = (varf * var_factor * std::pow( lambda, 2.0*nu ));
};

void set_pars (const double& l, const double& varf) {
	// set initial variance and state, and parameters
	assert(l>0 && varf > 0 && "l and varf ought to be possitive");
	l_ = l;
	varf_ = varf;
	get_varw(varf_, l_, varw_, lambda_);
	initial_mean_variance_Matern32(lambda_, varw_, x1, P1);
};

void set_data (const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr)
{
	times_ = times;
	y_i_ = y_i;
	yerr_ = yerr;
};

void Phi_Matern32(int i, const Eigen::VectorXd& times,  
		double lambda, 
		Eigen::Matrix2d& Phi)
{
	double delta = deltaF(i,times);
	assert(lambda > 0);
	double ef = std::exp(-lambda*delta);

	Phi(0,0) = ef * (lambda * delta + 1.0);
	Phi(0,1) = ef * delta;
	Phi(1,0) = -ef * delta * lambda * lambda;
	Phi(1,1) = ef * (1.0 - lambda * delta);
};

void Q_Matern32(int i, const Eigen::VectorXd& times,  
		double lambda, double varw,
		Eigen::Matrix2d& Q)
{
	double delta = deltaF(i,times);
	double delta2 = delta * delta;
	double lambda2 = lambda * lambda;
	double t1 = 1.0/(4.0 * lambda2);
	double texp = std::exp(-2.0 * delta * lambda);
	assert(lambda > 0);

	Q(0,0) = varw * t1 * (1.0-texp*(2.0*lambda2*delta2 + 2.0*delta*lambda + 1.0)) / lambda;
	Q(1,0) = varw * t1*(2.0 * delta2 * lambda2 * texp);
	Q(0,1) = Q(1,0);
	Q(1,1) = varw * t1 * lambda * (1.0 - texp*(2.0*delta2*lambda2 - 2.0*delta*lambda + 1.0));
};


void H_Matern32(Eigen::RowVector2d& h)
{
	h(0) = 1.0;
	h(1) = 0.0;
};

void R_Matern32(int i, const Eigen::VectorXd& sigma, Eigen::Matrix<double,1,1>& r)
{
	double sigmat = sigma(i);
	r(0) = sigmat * sigmat;
};

void initial_mean_variance_Matern32(const double& lambda, const double& varw,
	Eigen::Vector2d& x1, Eigen::Matrix2d& P1)
{
	x1(0) = 0.0;
	x1(1) = 0.0;
	P1(0,0) = varw / (4.0 * std::pow(lambda,3));
	P1(1,1) = varw / (4.0 * lambda);
	P1(0,1) = 0.0;
	P1(1,0) = 0.0;
};

double KF_log_likelihood()
{
	return KF_log_likelihood_(times_, y_i_, yerr_, lambda_, varw_);
}


// full version of KF_log_likelihood with input lambda and varw
double KF_log_likelihood_(const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr, 
	const double& lambda, const double& varw)
{
  
  int n = times.rows();
  assert( y_i.rows() == n && yerr.rows() == n && "dimension mismatch between y and yerr");
  double ll = 0.0;
  double dll;

  Eigen::Vector2d x_pred, x_filt, x_filt_old;
  Eigen::Matrix2d P_pred, P_filt, P_filt_old;

  H_Matern32(H_i);
  R_Matern32(0, yerr, R_i);

  kalman_recursions(y_i(0), H_i, R_i, x1, P1, dll, x_filt_old, P_filt_old);
  ll += dll;

  for (int j=1; j<n; j++)
  {
  	Phi_Matern32(j, times, lambda,  Phi);
  	Q_Matern32(j, times, lambda, varw, Q);
  	x_pred = Phi * x_filt_old;
  	P_pred = Phi * P_filt_old * Phi.transpose() + Q;
 	H_Matern32(H_i);
   	R_Matern32(j, yerr, R_i);
  	kalman_recursions(y_i(j), H_i, R_i, x_pred, P_pred, dll, x_filt, P_filt);
  	x_filt_old = x_filt;
  	P_filt_old = P_filt;
  	ll += dll;
  }

  return ll;

};

void simulate_Matern32(Eigen::VectorXd& y_sim,std::mt19937 rng){
	simulate_Matern32(times_, y_sim, false, rng);
}

void simulate_Matern32(const Eigen::VectorXd& simTimes, Eigen::VectorXd& y_sim, const bool zeroQ, std::mt19937 rng)
{
	int n = simTimes.rows();
	assert(n>1 && "please simulate at least two points!");

	y_sim = Eigen::VectorXd::Zero(n);
	Eigen::Vector2d x_sim,x_sim_old, MVdev, tmpVec;
	Eigen::Matrix2d Q_chol, R_chol;
	
  	Eigen::MatrixXd transform;


	//std::random_device rd;
	//std::mt19937 rng(rd());
	std::normal_distribution<double> gaussiana;

	double sigma_r;

  	H_Matern32(H_i);
  	R_Matern32(0, yerr_, R_i);

  	sigma_r = std::sqrt(R_i.sum()); // R_i is 1d in our case, make it a scalar

	y_sim(0) = H_i * x1 + sigma_r * gaussiana(rng);
	x_sim_old = x1;

	if (!zeroQ)
	{
		for(int j=1; j<n; j++)
		{
			R_Matern32(j, yerr_, R_i);
		 	Phi_Matern32(j, times_, lambda_, Phi);
			Q_Matern32(j, times_, lambda_, varw_, Q);
			// get MV deviate with covariance matrix Q
			for(int k = 0; k<2; k++)
				Q(k,k)+=1e-10;
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(Q);
			transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
			for (int k = 0; k<2; k++)
				tmpVec(k) = gaussiana(rng);
			MVdev = transform * tmpVec;
			//
			x_sim = Phi * x_sim_old + MVdev;
			sigma_r = std::sqrt(R_i.sum());
			y_sim(j) = H_i*x_sim + sigma_r * gaussiana(rng);
			x_sim_old = x_sim;
		}
	}
	else
	{
		for(int j=1; j<n; j++)
		{
			R_Matern32(j, yerr_, R_i);
		 	Phi_Matern32(j, times_, lambda_, Phi);
		 	x_sim = Phi * x_sim_old;
		 	sigma_r = std::sqrt(R_i.sum());
			y_sim(j) = H_i*x_sim + sigma_r * gaussiana(rng);
			x_sim_old = x_sim;		 
		}	
	}

};

void resample_parameters(Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood, 
	Eigen::VectorXd& logl_array,  Eigen::VectorXd& logvarf_array, 
	const int n_part,  std::mt19937 rng) {
	// draw a sample from a multinomial acroding to probabilities given by weights
	std::cout << "resampling ..." << std::endl;
	std::discrete_distribution<int> dist(weights.data(), weights.data()+weights.size());
	Eigen::VectorXd samples(n_part);
	for (int k = 0; k<n_part; k++)
		samples(k) = dist(rng);
	Eigen::MatrixXd logl_array_new = Eigen::VectorXd::Zero(n_part);
	Eigen::MatrixXd logvarf_array_new = Eigen::VectorXd::Zero(n_part);
	Eigen::VectorXd log_likelihood_new = Eigen::VectorXd::Zero(n_part);
	for (int k = 0; k<n_part; k++)
	{
		logl_array_new.row(k) = logl_array.row(samples(k));
		logvarf_array_new.row(k) = logvarf_array.row(samples(k));
		log_likelihood_new(k) = log_likelihood(samples(k));
	}
	logl_array = logl_array_new;
	logvarf_array = logvarf_array_new;
	log_likelihood = log_likelihood_new;
	std::cout << "resampling done." << std::endl;
}

void construct_covariance(Eigen::VectorXd& logl_array,  
	Eigen::VectorXd& logvarf_array, 
	const int n_part,  
	Eigen::MatrixXd& transform, Eigen::VectorXd& mu){
	std::cout << "computing covariance ..." << std::endl;
	Eigen::MatrixXd mat(n_part,2);

	mu(0) = logl_array.mean();
	mu(1) = logvarf_array.mean();

	for (int k=0; k < n_part; k++)
	{
		mat(k,0) = logl_array(k) - mu(0);
		mat(k,1) = logvarf_array(k) - mu(1);
	}

	// This is the covariance considering weighted observations
	//Eigen::MatrixXd cov = (mat.adjoint() * (weights.asDiagonal() * mat))/(1-sum_weights2);
	Eigen::MatrixXd cov = (mat.adjoint() * mat) / double(mat.rows());
	// Use mu and cov to sample from Multivariate normal
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
	transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();

	std::cout << mu << std::endl;
	std::cout << cov << std::endl;
	std::cout << "computing covariance done." << std::endl;
}

Eigen::VectorXd sample_gaussian_within_limits(
	Eigen::VectorXd& l_lims, Eigen::VectorXd& varf_lims, 
	std::mt19937 rng, 
	Eigen::MatrixXd& transform, Eigen::VectorXd& mu) {
	Eigen::VectorXd deviate(mu.size());
	Eigen::VectorXd tmpVec(mu.size());
	std::normal_distribution<double> gaussiana;
	bool cond = false;
	while (!cond) {
		cond = true;
		for (int kk=0; kk < mu.size(); kk++)
			tmpVec(kk) = gaussiana(rng);
		
		deviate = transform * tmpVec;

		deviate(0) += mu(0); 
		double ll = std::exp( deviate(0) );
		if ((ll < l_lims(0)) || (ll > l_lims(1)))
		{
			cond=false;
			continue;
		}
		deviate(1) += mu(1);
		double varf	 = std::exp( deviate(1) );
		if ((varf < varf_lims(0)) || (varf > varf_lims(1)))
		{
			cond=false;
		}
	}
	return deviate;
}

void move_particles(int i, 
	Eigen::VectorXd& logl_array, Eigen::VectorXd& logvarf_array, 
	Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood,
	Eigen::VectorXd& l_lims,  Eigen::VectorXd& varf_lims, 
	std::mt19937 rng, const int n_part, 
	Eigen::MatrixXd& transform, Eigen::VectorXd& mu
) {
	std::cout << "updating particles..." << std::endl;
	double lambda;
	double ll = 0.0;
	double varf = 0.0;
	double varw = 0.0; 
	
	std::normal_distribution<double> gaussiana;
	std::uniform_real_distribution<double> uniform(0, 1);

	for (int k = 0; k < n_part; k++)
	{
		std::cout << k << " ... \r";
		mu(0) = logl_array(k);
		mu(1) = logvarf_array(k);
		Eigen::VectorXd deviate = sample_gaussian_within_limits(
			l_lims,  varf_lims,
			rng, transform, mu);

		ll = std::exp( deviate(0) );
		varf   = std::exp( deviate(1) );
		get_varw(varf, ll, varw, lambda);

		double res_proposal = KF_log_likelihood_(times_.head(i), y_i_.head(i), yerr_.head(i), lambda, varw);
		double log_like_diff = res_proposal - log_likelihood(k);
		double ratio = std::min(std::exp(log_like_diff),1.0);
		double u_dev = uniform(rng);
		if (u_dev < ratio)
		{
			// we accept the proposal
			logl_array(k) = deviate(0);
			logvarf_array(k) = deviate(1);
			log_likelihood(k) = res_proposal;
		}
	}
	// resampling/remove done, reset weights to uniform values
	weights = Eigen::VectorXd::Ones(n_part) / n_part;
	std::cout << std::endl;
	std::cout << "updating particles done." << std::endl;
}



void SeqMC_Matern32(Eigen::VectorXd& logl_array, Eigen::VectorXd& logvarf_array,
	Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood,
	Eigen::VectorXd& l_lims, Eigen::VectorXd& varf_lims,
	int save_interval, std::mt19937 rng)
{

	assert(logl_array.rows() == logvarf_array.rows() && "prior vectors should have same number of rows!");
	const int n = times_.rows();
	const int n_part = logl_array.rows();
	double gamma_t = 0.1;
	double ess_threshold = n_part * gamma_t;
	double ll, varf, lambda, varw, dll, ESS;

  	Eigen::Vector2d x_pred, x_filt, x_filt_old, x1_s;
  	Eigen::Matrix2d P_pred, P_filt, P_filt_old, P1_s;
  	Eigen::Vector2d *x_filt_old_array = new Eigen::Vector2d[n_part];
  	Eigen::Matrix2d *P_filt_old_array = new Eigen::Matrix2d[n_part];
	Eigen::MatrixXd transform(n_part,2);
	Eigen::VectorXd mu(2);

  	// initia values of H and R matrices
	H_Matern32(H_i);
  	R_Matern32(0, yerr_, R_i);

  	// initial values of weights and likelihood vector
  	weights = Eigen::VectorXd::Ones(n_part) / n_part;
  	log_likelihood = Eigen::VectorXd::Zero(n_part);

  	// first iteration
  	for(int j=0; j<n_part; j++)
  	{
    	ll = std::exp(logl_array(j));
    	varf = std::exp(logvarf_array(j));
    	get_varw(varf, ll, varw, lambda);
    	initial_mean_variance_Matern32(lambda, varw, x1_s, P1_s);
		kalman_recursions(y_i_(0), H_i, R_i, x1_s, P1_s, dll, x_filt_old, P_filt_old);
		x_filt_old_array[j] = x_filt_old;
		P_filt_old_array[j] = P_filt_old;
		log_likelihood(j) += dll;
  	}

  	bool resample;
  	// loop now over all observations
  	for (int i=1; i<n; i++)
  	{
		H_Matern32(H_i);
   		R_Matern32(i,yerr_, R_i);
   		std::cout << "In iteration " << i << std::endl;
 		resample = false;
  		for(int j=0; j<n_part; j++)
  		{
  			//calculate likelihood update using Kalman filter
  			ll = std::exp(logl_array(j));
  			varf = std::exp(logvarf_array(j));
  			get_varw(varf, ll, varw, lambda);
  			Phi_Matern32(i, times_, lambda,  Phi);
  			Q_Matern32(i, times_, lambda, varw, Q);
  			x_filt_old = x_filt_old_array[j];
  			P_filt_old = P_filt_old_array[j];

  			x_pred = Phi * x_filt_old;
  			P_pred = Phi * P_filt_old * Phi.transpose() + Q;
			kalman_recursions(y_i_(i), H_i, R_i, x_pred, P_pred, dll, x_filt, P_filt);
  			x_filt_old_array[j] = x_filt;
  			P_filt_old_array[j] = P_filt;
  			log_likelihood(j) += dll;
  			weights(j) *= std::exp( dll );

  		}
  		// normalizae weights vector such that \Sum_{i=1}^{n_part} w_i = 1
  		weights /= weights.sum();
  		//check ESS. If ESS > threshold, jump to next iteration
  		double sum_weights2 = weights.dot(weights);
  		ESS = (1.0 / sum_weights2);
  		if (ESS < ess_threshold)
  		{
  			std::cout << "In iteration " << i << " we entered a resample/move stage" << std::endl;
  			// draw a sample from a multinomial acroding to probabilities given by weights
 			resample_parameters(weights, log_likelihood, logl_array, logvarf_array, n_part,  rng);
			construct_covariance(logl_array, logvarf_array, n_part,  transform, mu);
			move_particles(i, logl_array, logvarf_array,
				weights, log_likelihood, 
				l_lims, varf_lims,
				rng, n_part,  transform, mu);
			resample = true;
		}
		std::cout << "exp of mean of l: " << std::exp(logl_array.dot(weights)) << std::endl;
		std::cout << "exp of mean of varf: " << std::exp(logvarf_array.dot(weights)) << std::endl;
		std::cout << "mean of log likelihodd: " << log_likelihood.dot(weights) << std::endl;
		if ((i%save_interval == 0) || (resample) || (i==(n-1)))
		{
			std::string path = "data/m32_post_" + std::to_string(i) + ".txt";
			if (resample)
				path = "data/m32_post_re_" + std::to_string(i) + ".txt";
			std::ofstream file(path);
			if (file.is_open())
			{
				file << '#';
				file << "";
				file << "l ";
				file << "varf ";
				file << "weight" << std::endl;				
				for(int m=0; m<n_part; m++)
				{
					file << std::exp(logl_array(m)) << " ";
					file << std::exp(logvarf_array(m)) << " ";
					file << weights(m) << std::endl;
				}
			}


  		}
  	}
	delete[] x_filt_old_array;
	delete[] P_filt_old_array;
	x_filt_old_array = 0;
	P_filt_old_array = 0;
};





private:

  double  lambda_, varf_, varw_, l_;

  Eigen::VectorXd times_, y_i_, yerr_; 
  Eigen::Matrix2d Q, Phi, P1;
  Eigen::RowVector2d H_i;
  Eigen::Matrix<double,1,1> R_i;
  Eigen::Vector2d x1;


}; // class DSHOSolver


}; // namespace dsho
}; // namespace gpstate

#endif //_GPSTATE_MATERN32_H
