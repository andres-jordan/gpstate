#ifndef _GPSTATE_N_DSHOIS_H
#define _GPSTATE_N_DSHOIS_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "KF.h"
#include "ndsho.h"

// implements a class for solving N DSHOs

using namespace KF;

namespace gpstate {
namespace n_dsho {

class N_DSHOSolver_IS : N_DSHOSolver {
public:

N_DSHOSolver_IS (const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr,
	const Eigen::VectorXd& omega0, const Eigen::VectorXd& Qpar, 
	const Eigen::VectorXd& varf) : N_DSHOSolver(times, y_i, yerr, omega0, Qpar, varf)
{
};


double weighted_mean(Eigen::VectorXd x, Eigen::VectorXd& weights) {
	return x.dot(weights) / (weights.sum());
}

double weighted_stdev(Eigen::VectorXd x, Eigen::VectorXd& weights) {
	double mean = weighted_mean(x, weights);
	double var = ((x.array() - mean).square() * weights.array()).sum() / weights.sum();
	double std = std::sqrt(var);
	if (std < 1e-5){
		std = 1e-5;
	}
	return std;
}

double stdev(Eigen::VectorXd x) {
	double var = (x.array() - x.mean()).square().sum() / (x.size() - 1);
	return std::sqrt(var);
}

/*void resample_parameters(Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood, Eigen::MatrixXd& logomega0_array, Eigen::MatrixXd logQpar_array, Eigen::MatrixXd& logvarf_array, const int n_part, const int npars_, std::mt19937 rng) {
	// draw a sample from a multinomial acroding to probabilities given by weights
	std::cout << "resampling ..." << std::endl;
	std::discrete_distribution<int> dist(weights.data(), weights.data()+weights.size());
	Eigen::VectorXd samples(n_part);
	for (int k = 0; k<n_part; k++)
		samples(k) = dist(rng);
	Eigen::MatrixXd logomega0_array_new = Eigen::MatrixXd::Zero(n_part, npars_);
	Eigen::MatrixXd logQpar_array_new = Eigen::MatrixXd::Zero(n_part, npars_);			
	Eigen::MatrixXd logvarf_array_new = Eigen::MatrixXd::Zero(n_part, npars_);
	Eigen::VectorXd log_likelihood_new = Eigen::VectorXd::Zero(n_part);
	for (int k = 0; k<n_part; k++)
	{
		logomega0_array_new.row(k) = logomega0_array.row(samples(k));
		logQpar_array_new.row(k) = logQpar_array.row(samples(k));				
		logvarf_array_new.row(k) = logvarf_array.row(samples(k));
		log_likelihood_new(k) = log_likelihood(samples(k));
	}
	logomega0_array = logomega0_array_new;
	logQpar_array = logQpar_array_new;			
	logvarf_array = logvarf_array_new;
	log_likelihood = log_likelihood_new;
	std::cout << "resampling done." << std::endl;
}*/

void construct_proposal(Eigen::MatrixXd& logomega0_array, Eigen::MatrixXd logQpar_array, Eigen::MatrixXd& logvarf_array, const int n_part, const int npars_, Eigen::VectorXd& stdevs, Eigen::VectorXd& mu){
	std::cout << "computing covariance ..." << std::endl;
	//Eigen::VectorXd mu(3*npars_);

	for(int k=0; k < npars_; k++)
	{
		mu(3*k)=logomega0_array.col(k).mean();
		mu(3*k+1)=logQpar_array.col(k).mean();
		mu(3*k+2)=logvarf_array.col(k).mean();
		stdevs(3*k) = stdev(logomega0_array.col(k));
		stdevs(3*k+1)=stdev(logQpar_array.col(k));
		stdevs(3*k+2)=stdev(logvarf_array.col(k));
	}	
	std::cout << mu << std::endl;
	std::cout << stdevs << std::endl;
	std::cout << "computing covariance done." << std::endl;
}

void construct_proposal(Eigen::VectorXd& weights, Eigen::MatrixXd& logomega0_array, Eigen::MatrixXd logQpar_array, Eigen::MatrixXd& logvarf_array, const int n_part, const int npars_, Eigen::VectorXd& stdevs, Eigen::VectorXd& mu){
	std::cout << "computing covariance ..." << std::endl;
	//Eigen::VectorXd mu(3*npars_);

	for(int k=0; k < npars_; k++)
	{
		mu(3*k)=weighted_mean(logomega0_array.col(k), weights);
		mu(3*k+1)=weighted_mean(logQpar_array.col(k), weights);
		mu(3*k+2)=weighted_mean(logvarf_array.col(k), weights);
		stdevs(3*k) = weighted_stdev(logomega0_array.col(k), weights);
		stdevs(3*k+1)=weighted_stdev(logQpar_array.col(k), weights);
		stdevs(3*k+2)=weighted_stdev(logvarf_array.col(k), weights);
	}
	std::cout << mu << std::endl;
	std::cout << stdevs << std::endl;
	std::cout << "computing covariance done." << std::endl;
}

// to enable rejection sampling, uncomment the next line
// #define REJECTION_SAMPLING_ENABLED
// it is not very efficient

Eigen::VectorXd sample_gaussian_within_limits(
	Eigen::MatrixXd& omega0_lims, Eigen::MatrixXd& Qpar_lims, Eigen::MatrixXd& varf_lims, 
	std::mt19937& rng, const int npars_, Eigen::VectorXd& stdevs, Eigen::VectorXd& mu,
	int& draw_direct_successes, int& draw_reject_successes, int& draw_total_attempts
) {
	Eigen::VectorXd deviate(mu.size());
	std::normal_distribution<double> gaussiana;
	std::uniform_real_distribution<double> uniform(0, 1);
	//std::cout << "  gauss sampling ..." << std::endl;
	bool cond = true;
	int i = 0;
	do {
		cond = true;
		i++;
		draw_total_attempts++;
		// draw from gaussian
		for (int kk=0; kk < mu.size(); kk++) {
			double u = gaussiana(rng);
			double v = u * stdevs(kk) + mu(kk);
			//std::cout << "        drawing... " << u << " * " << stdevs(kk) << " + " << mu(kk) << " = " << v << std::endl;
			deviate(kk) = v;
		}
		//std::cout << "    draw @" << deviate << std::endl;
		
		for (int kk=0; kk < npars_; kk++)
		{
			double omega0 = std::exp( deviate(3*kk) );
			if (omega0 < omega0_lims(0,kk))
				cond=false;
			if (omega0 > omega0_lims(1,kk))
				cond=false;
			if (!cond)
				break;
			double Qpar	 = std::exp( deviate(3*kk + 1) );
			if (Qpar < Qpar_lims(0,kk))
				cond=false;
			if (Qpar > Qpar_lims(1,kk))
				cond=false;
			if (!cond)
				break;
			double varf	 = std::exp( deviate(3*kk + 2) );
			if (varf < varf_lims(0,kk))
				cond=false;
			if (varf > varf_lims(1,kk))
				cond=false;
			if (!cond)
				break;
		}
		if(cond)
			draw_direct_successes++;
	} while(!cond);
	return deviate;
}

void move_particles(int i, 
	Eigen::MatrixXd& logomega0_array, Eigen::MatrixXd logQpar_array, 
	Eigen::MatrixXd& logvarf_array, 
	Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood,
	Eigen::MatrixXd& omega0_lims, Eigen::MatrixXd& Qpar_lims, Eigen::MatrixXd& varf_lims, 
	std::mt19937& rng, const int n_part, const int npars_, Eigen::VectorXd& stdevs, Eigen::VectorXd& mu
) {
	std::cout << "updating particles..." << std::endl;
	Eigen::VectorXd omega0, Qpar, varf, varw;
	omega0 = Eigen::VectorXd::Zero(npars_);
	Qpar = Eigen::VectorXd::Zero(npars_);
	varf = Eigen::VectorXd::Zero(npars_);
	varw = Eigen::VectorXd::Zero(npars_);

	int draw_direct_successes = 0;
	int draw_reject_successes = 0;
	int draw_total_attempts = 0;
	int move_successes = 0;
	
	std::normal_distribution<double> gaussiana;
	std::uniform_real_distribution<double> uniform(0, 1);
	double weightnorm = log_likelihood.maxCoeff();
	for (int k = 0; k < n_part; k++)
	{
		if (k % 40 == 0)
			std::cout << k << " ... \r" << std::flush;
		
		Eigen::VectorXd deviate = sample_gaussian_within_limits(
			omega0_lims, Qpar_lims, varf_lims,
			rng, npars_, stdevs, mu, 
			draw_direct_successes, draw_reject_successes, draw_total_attempts
		);

		for (int kk=0; kk<npars_; kk++)
		{
			omega0(kk) = std::exp( deviate(3*kk) );
			Qpar(kk)	 = std::exp( deviate(3*kk + 1) );
			varf(kk)	 = std::exp( deviate(3*kk + 2) );
		}

		get_varw(varf, omega0, Qpar, varw);
		double res_proposal = KF_log_likelihood_(times_.head(i), y_i_.head(i), yerr_.head(i), omega0, Qpar, varw);
		for (int kk=0; kk<npars_; kk++)
		{
			logomega0_array(k,kk) = deviate(3*kk);
			logQpar_array(k,kk)   = deviate(3*kk+1);
			logvarf_array(k,kk)   = deviate(3*kk+2);
		}
		log_likelihood(k)	= res_proposal;
		double offset = ((deviate - mu).array() / stdevs.array()).square().sum();
		double log_density_proposal = -0.5 * offset - stdevs.array().log().sum();
		weights(k) = std::exp(res_proposal - weightnorm - log_density_proposal);
		//std::cout << "   weight: " << weights(k) << " prop:" << (res_proposal - weightnorm) << " density:" << log_density_proposal << std::endl;
		move_successes++;
	}
	// resampling/remove done, reset weights to uniform values
	weights /= weights.sum();
	std::cout << std::endl;
	std::cout << "updating particles done. acceptance rates:" << draw_direct_successes << ", " << draw_reject_successes << " of " << draw_total_attempts << ". Moves: " << move_successes << " of " << n_part << std::endl;
}

void SeqMC_N_DSHO(Eigen::MatrixXd& logomega0_array, Eigen::MatrixXd logQpar_array, 
				Eigen::MatrixXd& logvarf_array, 
				Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood,
				Eigen::MatrixXd& omega0_lims, Eigen::MatrixXd& Qpar_lims, Eigen::MatrixXd& varf_lims, 
				int save_interval, std::mt19937& rng)
{
	assert(logomega0_array.rows() == logvarf_array.rows() && logQpar_array.rows() == logvarf_array.rows() 
		&&	"prior vectors should have same number of rows!");
	assert(logomega0_array.cols() == logvarf_array.cols() && logQpar_array.cols() == logvarf_array.cols() 
		&&	"prior vectors should have same number of columns!");

	const int n = times_.rows();
	const int n_part = logomega0_array.rows();
	int npars_save = npars_;
	npars_ = logomega0_array.cols();

	double gamma_t = 0.1;
	double ess_threshold = n_part * gamma_t;
	Eigen::VectorXd omega0, Qpar, varf, varw;
	double dll, ESS;

	Eigen::VectorXd x_pred, x_filt, x_filt_old, x1_s;
	Eigen::MatrixXd P_pred, P_filt, P_filt_old, P1_s;
	Eigen::VectorXd *x_filt_old_array = new Eigen::VectorXd[n_part];
	Eigen::MatrixXd *P_filt_old_array = new Eigen::MatrixXd[n_part];
	Eigen::VectorXd stdevs(3*npars_);
	Eigen::VectorXd mu(3*npars_);

	P_filt = Eigen::MatrixXd::Zero(2*npars_, 2*npars_);
	P_filt_old = Eigen::MatrixXd::Zero(2*npars_, 2*npars_);


	omega0 = Eigen::VectorXd::Zero(npars_);
	Qpar = Eigen::VectorXd::Zero(npars_);
	varf = Eigen::VectorXd::Zero(npars_);
	varw = Eigen::VectorXd::Zero(npars_);

	// initialise values of H and R matrices
	//H_N_DSHO(H_i);
	R_N_DSHO(0, yerr_, R_i);

	// initial values of weights and likelihood vector
	weights = Eigen::VectorXd::Ones(n_part) / n_part;
	log_likelihood = Eigen::VectorXd::Zero(n_part);

	// first iteration
	for(int j=0; j<n_part; j++)
	{
		for (int kk = 0; kk<npars_; kk++)
		{
			omega0(kk) = std::exp(logomega0_array(j,kk));
			Qpar(kk) = std::exp(logQpar_array(j,kk));
			varf(kk) = std::exp(logvarf_array(j,kk));
		}
		get_varw(varf, omega0, Qpar, varw);
		initial_mean_variance_N_DSHO(omega0, Qpar, varw, x1_s, P1_s);
		//kalman_recursions(y_i_(0), H_i, R_i, x1_s, P1_s, dll, x_filt_old, P_filt_old);
		kalman_recursions_NDSHO(2*npars_, y_i_(0), R_i, x1_s, P1_s, dll, x_filt_old, P_filt_old);

		x_filt_old_array[j] = x_filt_old;
		P_filt_old_array[j] = P_filt_old;
		log_likelihood(j) += dll;
	}

	// loop now over all observations
	bool resample;
	for (int i=1; i<n; i++)
	{
		//H_N_DSHO(H_i);
 		R_N_DSHO(i,yerr_, R_i);
 		std::cout << "In iteration " << i << std::endl;
 		resample = false;
		for(int j=0; j<n_part; j++)
		{
			//calculate likelihood update using Kalman filter
			for (int kk = 0; kk<npars_; kk++)
			{
				omega0(kk) = std::exp(logomega0_array(j,kk));
				Qpar(kk) = std::exp(logQpar_array(j,kk));
				varf(kk) = std::exp(logvarf_array(j,kk));
			}

			get_varw(varf, omega0, Qpar, varw);
			Phi_N_DSHO(i, times_, omega0, Qpar,	Phi);
			Q_N_DSHO(i, times_, omega0, Qpar, varw, Q);
			x_filt_old = x_filt_old_array[j];
			P_filt_old = P_filt_old_array[j];

			x_pred = Phi * x_filt_old;
			P_pred = Phi * P_filt_old * Phi.transpose() + Q;
			
			//kalman_recursions(y_i_(i), H_i, R_i, x_pred, P_pred, dll, x_filt, P_filt);
			kalman_recursions_NDSHO(2*npars_, y_i_(i), R_i, x_pred, P_pred, dll, x_filt, P_filt);
			x_filt_old_array[j] = x_filt;
			P_filt_old_array[j] = P_filt;
			log_likelihood(j) += dll;
			weights(j) *= std::exp( dll );
		}
		// normalize weights vector such that \Sum_{i=1}^{n_part} w_i = 1
		weights /= weights.sum();
		//check ESS. If ESS > threshold, jump to next iteration
		double sum_weights2 = weights.dot(weights);
		ESS = (1.0 / sum_weights2);
		std::cout << "ess: " << ESS << std::endl;
		if (ESS < ess_threshold)
		{
			std::cout << "In iteration " << i << " we entered a resample/move stage" << std::endl;
			//resample_parameters(, log_likelihood, logomega0_array, logQpar_array, logvarf_array, n_part, npars_, rng);
			construct_proposal(weights, logomega0_array, logQpar_array, logvarf_array, n_part, npars_, stdevs, mu);
			move_particles(i, logomega0_array, logQpar_array, logvarf_array,
				weights, log_likelihood, 
				omega0_lims, Qpar_lims, varf_lims,
				rng, n_part, npars_, stdevs, mu);
			resample = true;
		}
		std::cout << "exp of mean of omega(0), Q(0), varf(0): " << std::exp(logomega0_array.col(0).dot(weights)) << " ";
		std::cout << std::exp(logQpar_array.col(0).dot(weights)) << " ";
		std::cout << std::exp(logvarf_array.col(0).dot(weights)) << std::endl;
		std::cout << "exp of mean of omega(1), Q(1), varf(1): " << std::exp(logomega0_array.col(1).dot(weights)) << " ";	
		std::cout << std::exp(logQpar_array.col(1).dot(weights)) << " ";
		std::cout << std::exp(logvarf_array.col(1).dot(weights)) << std::endl;	
		std::cout << "max of log likelihood: " << log_likelihood.maxCoeff() << std::endl;

		if ((i%save_interval == 0) || (resample) || (i==(n-1)))
		{
			std::string path = "data/ndshois_post_" + std::to_string(i) + ".txt";
			if (resample)
				path = "data/ndshois_post_re_" + std::to_string(i) + ".txt";
			std::ofstream file(path);
			if (file.is_open())
			{
				file << '#';
				for (int m=0; m<npars_; m++)
					file << "omega"+std::to_string(m)+" ";
				for (int m=0; m<npars_; m++)
					file << "Q"+std::to_string(m)+" ";
				for (int m=0; m<npars_; m++)
					file << "varf"+std::to_string(m)+" ";	
				file << "weight" << std::endl;				
				for(int m=0; m<n_part; m++)
				{
					for (int mm=0; mm<npars_; mm++)
						file << std::exp(logomega0_array(m,mm)) << " ";
					for (int mm=0; mm<npars_; mm++)
						file << std::exp(logQpar_array(m,mm)) << " ";
					for (int mm=0; mm<npars_; mm++)
						file << std::exp(logvarf_array(m,mm)) << " ";
					file << weights(m) << std::endl;
				}
			}
		}

	}
	delete[] x_filt_old_array;
	delete[] P_filt_old_array;
	x_filt_old_array = 0;
	P_filt_old_array = 0;
	npars_ = npars_save;
}



};


}; // namespace n_dsho
}; // namespace gpstate

#endif //_GPSTATE_N_DSHOIS_H
