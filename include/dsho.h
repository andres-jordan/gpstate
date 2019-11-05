#ifndef _GPSTATE_DSHO_H
#define _GPSTATE_DSHO_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "KF.h"
#include <iostream>
#include <fstream>

// implements a class for solving a single DSHO
// calculates Q, Phi, H and R matrices as detailed in Jordan et al

using namespace KF;

namespace gpstate {
namespace dsho {

class DSHOSolver {
public:
DSHOSolver (const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr,
	const double& omega0, const double& Qpar, const double& varf)
{
	set_pars(omega0, Qpar, varf);
	set_data(times, y_i, yerr);
};

void get_varw(const double& varf, const double& omega0, const double& Qpar, 
		double& varw)
{
	// varf (input) is total power
	// varw (output) is the variance of the driving Wiener process
	double var_factor = 2.0 * std::pow(omega0,3.0) / (Qpar);	
	varw = (varf * var_factor);
};

void set_pars(const double& omega0, const double& Qpar, const double& varf) {
	// set initial variance and state
	assert(omega0 > 0 && Qpar > 0 && varf > 0 && "omega0, Qpar and varf ought to be positive");
	omega0_ = omega0;
	Qpar_  = Qpar;
	varf_ = varf;
	get_varw(varf_, omega0_, Qpar_, varw_);
	initial_mean_variance_DSHO(omega0_, Qpar_, varw_, x1, P1);
}

void set_data (const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr)
{
	times_ = times;
	y_i_ = y_i;
	yerr_ = yerr;
};


void Phi_DSHO(int i, const Eigen::VectorXd& times,  
		double omega, double Qpar, 
		Eigen::Matrix2d& Phi)
{
	double delta = deltaF(i,times);
	double beta, cc, ss;
	double ef = std::exp(-omega * delta / (2*Qpar));
	if ((std::isfinite(ef)) && (ef > 1e-10))
	{
		if (Qpar > 0.5)
		{
			beta = std::sqrt(Qpar*Qpar - 0.25);
			if (beta < 1e-2)
				beta = 1e-2;
			cc = std::cos(omega * beta * delta / Qpar);
			ss = std::sin(omega * beta * delta / Qpar);
		}
		else{
			beta = std::sqrt(0.25 - Qpar*Qpar);
			if (beta < 1e-2)
				beta = 1e-2;
			cc = std::cosh(omega * beta * delta / Qpar);
			ss = std::sinh(omega * beta * delta / Qpar);

		}

		Phi(0,0) = ef * (cc + ss/(2*beta));
		Phi(0,1) = ef * Qpar * ss / (omega * beta);
		Phi(1,0) = -ef * Qpar * omega * ss / beta;
		Phi(1,1) = ef * (cc-ss/(2*beta));
	}
	else
	{
		Phi(0,0) = 0.0;
		Phi(0,1) = 0.0;
		Phi(1,0) = 0.0;
		Phi(1,1) = 0.0;
	}
};

void Q_DSHO(int i, const Eigen::VectorXd& times,  
		double omega, double Qpar, double varw,
		Eigen::Matrix2d& Q)
{
	double delta = deltaF(i,times);
	double beta,cc,ss,cc2,ss2;
	double texp = std::exp(-omega*delta/Qpar);
	if ((std::isfinite(texp)) && (texp > 1e-10))
	{
		texp *= Qpar;
		if (Qpar > 0.5)
		{
			beta = std::sqrt(Qpar*Qpar - 0.25);
			if (beta < 1e-2)
				beta = 1e-2;
			double argu = omega * beta * delta / Qpar;
			cc = std::cos(argu);
			ss = std::sin(argu);
			cc2 = std::cos(2*argu);
			ss2 = std::sin(2*argu);
		}
		else
		{
			beta = std::sqrt(0.25 - Qpar*Qpar);
			if (beta < 1e-2)
				beta = 1e-2;
			double argu = omega * beta * delta / Qpar;
			cc = std::cosh(argu);
			ss = std::sinh(argu);
			cc2 = std::cosh(2*argu);
			ss2 = std::sinh(2*argu);		
		}

		if (Qpar > 0.5)
		{
			Q(0,0) = varw * texp * ((cc2-1.0) - 2.0*beta*ss2 + 4*std::pow(beta,2)*(std::exp(omega*delta/Qpar)-1))/(8.0*std::pow(omega,3)*std::pow(beta,2));
		}
		else
		{
			Q(0,0) = varw * texp * ((1.0-cc2) - 2.0*beta*ss2 + 4*std::pow(beta,2)*(std::exp(omega*delta/Qpar)-1))/(8.0*std::pow(omega,3)*std::pow(beta,2));
		}
		//
		Q(1,0) = varw * texp * (Qpar*std::pow(ss,2))/(2*std::pow(omega,2)*std::pow(beta,2));
		Q(0,1) = Q(1,0);
		//
		if (Qpar > 0.5)
		{
			Q(1,1) = varw * texp * ((cc2-1.0) + 2.0*beta*ss2 + 4*std::pow(beta,2)*(std::exp(omega*delta/Qpar)-1))/(8.0*omega*std::pow(beta,2));
		}
		else
		{
			Q(1,1) = varw * texp * ((1.0-cc2) + 2.0*beta*ss2 + 4*std::pow(beta,2)*(std::exp(omega*delta/Qpar)-1))/(8.0*omega*std::pow(beta,2));

		}
	}
	else{
		Q(0,0) = varw * Qpar / (2.0 * std::pow(omega,3));
		Q(1,1) = varw * Qpar / (2.0 * omega);
		Q(0,1) = 0.0;
		Q(1,0) = 0.0;
	}
};

void H_DSHO(Eigen::RowVector2d& h)
{
	h(0) = 1.0;
	h(1) = 0.0;
};

void R_DSHO(int i, const Eigen::VectorXd& sigma, Eigen::Matrix<double,1,1>& r)
{
	double sigmat = sigma(i);
	r(0) = sigmat * sigmat;
};

void initial_mean_variance_DSHO(const double& omega, const double& Qpar, const double& varw,
	Eigen::Vector2d& x1, Eigen::Matrix2d& P1)
{
	x1(0) = 0.0;
	x1(1) = 0.0;
	P1(0,0) = varw * Qpar / (2.0 * std::pow(omega,3));
	P1(1,1) = varw * Qpar / (2.0 * omega);
	P1(0,1) = 0.0;
	P1(1,0) = 0.0;
};

double KF_log_likelihood()
{
	return KF_log_likelihood_(times_, y_i_, yerr_, omega0_, Qpar_, varw_);
}


double KF_log_likelihood_(const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr,
	const double& omega0, const double& Qpar, const double& varw)
{
  int n = times.rows();
  assert( y_i.rows() == n && yerr.rows() == n && "dimension mismatch between y and yerr");
  double ll = 0.0;
  double dll;


  Eigen::Vector2d x_pred, x_filt, x_filt_old;
  Eigen::Matrix2d P_pred, P_filt, P_filt_old;

  //H_DSHO(H_i);
  R_DSHO(0, yerr, R_i);

  kalman_recursions_DSHO(y_i(0), R_i, x1, P1, dll, x_filt_old, P_filt_old);

  ll += dll;
  //std::cout << dll << std::endl;

  for (int j=1; j<n; j++)
  {
  	Phi_DSHO(j, times, omega0, Qpar, Phi);
  	Q_DSHO(j, times, omega0, Qpar, varw, Q);
  	x_pred = Phi * x_filt_old;
  	P_pred = Phi * P_filt_old * Phi.transpose() + Q;
 	//H_DSHO(H_i);
   	R_DSHO(j,yerr, R_i);
  	kalman_recursions_DSHO(y_i(j), R_i, x_pred, P_pred, dll, x_filt, P_filt);
  	//if (std::isnan(dll))
  	//{
  	//	std::cout << "dll: "<<dll << std::endl;
  	//	std::cout << "pars:" << omega0 << " " << Qpar << " " << varw << std::endl;
  	//	std::cout << "Phi:" <<Phi << " " << "Q:" << Q << " " << std::endl;
  	//	std::cout << "x_filt_old:" <<x_filt_old << " " << "P_filt_old:" << P_filt_old << " " << std::endl;  		
  	//	std::exit(0);
  	//}
  	x_filt_old = x_filt;
  	P_filt_old = P_filt;
  	ll += dll;
  }

  return ll;

};

void simulate_DSHO(Eigen::VectorXd& y_sim){
	simulate_DSHO(times_, y_sim, false);
}

void simulate_DSHO(const Eigen::VectorXd& simTimes, 
			Eigen::VectorXd& y_sim, 
			const bool zeroQ)
{
	int n = simTimes.rows();
	assert(n>1 && "please simulate at least two points!");

	y_sim = Eigen::VectorXd::Zero(n);
	Eigen::Vector2d x_sim,x_sim_old, MVdev, tmpVec;
	Eigen::Matrix2d Q_chol, R_chol;
	
  	Eigen::MatrixXd transform;


	std::random_device rd;
	std::mt19937 rng(rd());
	std::normal_distribution<double> gaussiana;

	double sigma_r;

  	H_DSHO(H_i);
  	R_DSHO(0, yerr_, R_i);

  	sigma_r = std::sqrt(R_i.sum()); // R_i is 1d in our case, make it a scalar

	y_sim(0) = H_i * x1 + sigma_r * gaussiana(rng);
	x_sim_old = x1;
	if (!zeroQ)
	{
		for(int j=1; j<n; j++)
		{
			R_DSHO(j, yerr_, R_i);
		 	Phi_DSHO(j, times_, omega0_, Qpar_, Phi);
			Q_DSHO(j, times_, omega0_, Qpar_, varw_, Q);
			for(int k = 0; k<2; k++)
				Q(k,k)+=1e-10;
			// get MV deviate with covariance matrix Q
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
			R_DSHO(j, yerr_, R_i);
		 	Phi_DSHO(j, times_, omega0_, Qpar_, Phi);
		 	x_sim = Phi * x_sim_old;
		 	sigma_r = std::sqrt(R_i.sum());
			y_sim(j) = H_i*x_sim + sigma_r * gaussiana(rng);
			x_sim_old = x_sim;		 
		}	
	}

};


void SeqMC_DSHO(Eigen::VectorXd& logomega0_array, Eigen::VectorXd logQpar_array, Eigen::VectorXd& logvarf_array, 
	Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood, 
				Eigen::Vector2d& omega0_lims, Eigen::Vector2d& Qpar_lims, Eigen::Vector2d& varf_lims,
				int save_interval)
{

	assert(logomega0_array.rows() == logvarf_array.rows() && logQpar_array.rows() == logvarf_array.rows() 
		&&  "prior vectors should have same number of rows!");
	const int n = times_.rows();
	const int n_part = logomega0_array.rows();
	double gamma_t = 0.1;
	double ess_threshold = n_part * gamma_t;
	double omega0, Qpar, varf, varw, dll, ESS;

  	Eigen::Vector2d x_pred, x_filt, x_filt_old, x1_s;
  	Eigen::Matrix2d P_pred, P_filt, P_filt_old, P1_s;
  	Eigen::Vector2d *x_filt_old_array = new Eigen::Vector2d[n_part];
  	Eigen::Matrix2d *P_filt_old_array = new Eigen::Matrix2d[n_part];
  	Eigen::MatrixXd transform;

  	// initia values of H and R matrices
	//H_DSHO(H_i);
  	R_DSHO(0, yerr_, R_i);


  	// initial values of weights and likelihood vector
  	weights = Eigen::VectorXd::Ones(n_part) / n_part;
  	log_likelihood = Eigen::VectorXd::Zero(n_part);

  	// first iteration
  	for(int j=0; j<n_part; j++)
  	{
    	omega0 = std::exp(logomega0_array(j));
    	Qpar = std::exp(logQpar_array(j));
    	varf = std::exp(logvarf_array(j));
    	get_varw(varf, omega0, Qpar, varw);
    	initial_mean_variance_DSHO(omega0, Qpar, varw, x1_s, P1_s);
		//kalman_recursions(y_i_(0), H_i, R_i, x1_s, P1_s, dll, x_filt_old, P_filt_old);
		kalman_recursions_DSHO(y_i_(0), R_i, x1_s, P1_s, dll, x_filt_old, P_filt_old);
		x_filt_old_array[j] = x_filt_old;
		P_filt_old_array[j] = P_filt_old;
		log_likelihood(j) += dll;
  	}

  	// loop now over all observations
  	bool resample;
  	for (int i=1; i<n; i++)
  	{
		//H_DSHO(H_i);
   		R_DSHO(i,yerr_, R_i);
   		resample=false;
   		std::cout << "In iteration " << i << std::endl;
  		for(int j=0; j<n_part; j++)
  		{
  			//calculate likelihood update using Kalman filter
  			omega0 = std::exp(logomega0_array(j));
	    	Qpar = std::exp(logQpar_array(j));
  			varf = std::exp(logvarf_array(j));
			get_varw(varf, omega0, Qpar, varw);
  			Phi_DSHO(i, times_, omega0, Qpar,  Phi);
  			Q_DSHO(i, times_, omega0, Qpar, varw, Q);
  			x_filt_old = x_filt_old_array[j];
  			P_filt_old = P_filt_old_array[j];

  			x_pred = Phi * x_filt_old;
  			P_pred = Phi * P_filt_old * Phi.transpose() + Q;
			//kalman_recursions(y_i_(i), H_i, R_i, x_pred, P_pred, dll, x_filt, P_filt);
			kalman_recursions_DSHO(y_i_(i), R_i, x_pred, P_pred, dll, x_filt, P_filt);
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
  			std::random_device rd;
			std::mt19937 rng(rd());
			std::discrete_distribution<int> dist(weights.data(), weights.data()+weights.size());
			std::uniform_real_distribution<double> uniform(0, 1);
			std::normal_distribution<double> gaussiana;
			Eigen::VectorXd samples(n_part);
			for (int k = 0; k<n_part; k++)
					samples(k) = dist(rng);
			Eigen::VectorXd logomega0_array_new = Eigen::VectorXd::Zero(n_part);
			Eigen::VectorXd logQpar_array_new = Eigen::VectorXd::Zero(n_part);			
			Eigen::VectorXd logvarf_array_new = Eigen::VectorXd::Zero(n_part);
			Eigen::VectorXd log_likelihood_new = Eigen::VectorXd::Zero(n_part);
			for (int k = 0; k<n_part; k++)
			{
				logomega0_array_new(k) = logomega0_array(samples(k));
				logQpar_array_new(k) = logQpar_array(samples(k));				
				logvarf_array_new(k) = logvarf_array(samples(k));
				log_likelihood_new(k) = log_likelihood(samples(k));
			}
			logomega0_array = logomega0_array_new;
			logQpar_array = logQpar_array_new;			
			logvarf_array = logvarf_array_new;
			log_likelihood = log_likelihood_new;
			// now perform the move stage 
			Eigen::MatrixXd mat(n_part,3);
			Eigen::VectorXd mu(3);
			mu(0)=logomega0_array.mean();
			mu(1)=logQpar_array.mean();
			mu(2)=logvarf_array.mean();
			for (int k=0; k < n_part; k++)
			{
				mat(k,0) = logomega0_array(k) - mu(0);
				mat(k,1) = logQpar_array(k) - mu(1);				
				mat(k,2) = logvarf_array(k) - mu(2);
			}
			//Eigen::MatrixXd cov = (mat.adjoint() * (weights.asDiagonal() * mat))/(1-sum_weights2);
			Eigen::MatrixXd cov = (mat.adjoint() * mat) / double(mat.rows());
			// Use mu and cov to sample from Multivariate normal
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
			transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();

			for (int k = 0; k < n_part; k++)
			{
				Eigen::VectorXd deviate(mu.size());
				Eigen::VectorXd tmpVec(mu.size());
				bool cond = true;
				do
				{
					cond=true;
					for (int kk=0; kk < mu.size(); kk++)
						tmpVec(kk) = gaussiana(rng);
					deviate = transform * tmpVec;
					//deviate(0) += logomega0_array(k);
					//deviate(1) += logQpar_array(k);
					//deviate(2) += logvarf_array(k);
					deviate(0) += mu(0);
					deviate(1) += mu(1);
					deviate(2) += mu(2);
	  				omega0 = std::exp( deviate(0) );
	  				Qpar   = std::exp( deviate(1) );
  					varf   = std::exp( deviate(2) );
  					if (omega0 < omega0_lims(0))
  						cond=false;
  					if (omega0 > omega0_lims(1))
   						cond=false; 						
  					if (Qpar < Qpar_lims(0))
  						cond=false;
   					if (Qpar > Qpar_lims(1))
  						cond=false;
  					if (varf < varf_lims(0))
  						cond=false;
   					if (varf > varf_lims(1))
  						cond=false;  					
				} while(!cond);
				// calculate likelihood for this proposal using kalman filter

  				get_varw(varf, omega0, Qpar, varw);
				double res_proposal = KF_log_likelihood_(times_.head(i), y_i_.head(i), yerr_.head(i), omega0, Qpar, varw);
				double log_like_diff = res_proposal - log_likelihood(k);
				double ratio = std::min(std::exp(log_like_diff),1.0);
				double u_dev = uniform(rng);
				if (u_dev < ratio)
				{
					// we accept the proposal
					logomega0_array(k) = deviate(0);
					logQpar_array(k)   = deviate(1);
					logvarf_array(k)   = deviate(2);
					log_likelihood(k)  = res_proposal;
				}
			}
		// resampling/remove done, reset weights to uniform values
		weights = Eigen::VectorXd::Ones(n_part) / n_part;
		resample = true;
  		}
	std::cout << "exp of mean of omega: " << std::exp(logomega0_array.dot(weights)) << std::endl;
	std::cout << "exp of mean of Q: " << std::exp(logQpar_array.dot(weights)) << std::endl;
	std::cout << "exp of mean of varf: " << std::exp(logvarf_array.dot(weights)) << std::endl;

	std::cout << "max of log likelihodd: " << log_likelihood.maxCoeff() << std::endl;

	 if ((i%save_interval == 0) || (resample) || (i==(n-1)))
	 {
	 	std::string path = "data/dsho_post_" + std::to_string(i) + ".txt";
	 	if (resample)
	 		path = "data/dsho_post_re_" + std::to_string(i) + ".txt";
	 	std::ofstream file(path);
	 	if (file.is_open())
  		{
    		file << "#omega0 Qpar varf weight" << std::endl;
    		for(int m=0; m<n_part; m++)
    		{
        		file <<  std::exp(logomega0_array(m)) <<" "<< std::exp(logQpar_array(m)) << " " << std::exp(logvarf_array(m)) << " " << weights(m) << std::endl;
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

  double omega0_, Qpar_, varf_, varw_;

  Eigen::VectorXd times_, y_i_, yerr_; 
  Eigen::Matrix2d Q, Phi, P1;
  Eigen::RowVector2d H_i;
  Eigen::Matrix<double,1,1> R_i;
  Eigen::Vector2d x1;


}; // class DSHOSolver


}; // namespace dsho
}; // namespace gpstate

#endif //_GPSTATE_DSHO_H