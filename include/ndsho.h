#ifndef _GPSTATE_N_DSHO_H
#define _GPSTATE_N_DSHO_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "KF.h"

// implements a class for solving N DSHOs
// calculates Q, Phi, H and R matrices as detailed in Jordan et al


using namespace KF;

namespace gpstate {
namespace n_dsho {

class N_DSHOSolver {
public:
N_DSHOSolver (const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr,
	const Eigen::VectorXd& omega0, const Eigen::VectorXd& Qpar, 
	const Eigen::VectorXd& varf)
{
	set_pars(omega0, Qpar, varf);
	set_data(times, y_i, yerr);
};

void get_varw(const Eigen::VectorXd& varf, const Eigen::VectorXd& omega0, 
	const Eigen::VectorXd& Qpar, Eigen::VectorXd& varw)
{
	// varf (input) is power at omega0
	// varw (output) is the variance of the driving Wiener process
	for (int i=0; i<npars_; i++)
	{
		double var_factor = 2.0 * std::pow(omega0(i),3.0) / (Qpar(i));
		varw(i) = (varf(i) * var_factor);
	}
};

void set_pars(const Eigen::VectorXd& omega0, const Eigen::VectorXd& Qpar, 
	const Eigen::VectorXd& varf) {
	// set initial variance and state
	npars_ = omega0.rows();
	assert(omega0.rows() == Qpar.rows() && omega0.rows() == varf.rows() 
		&& "parameter vectors ought to have same number of entries");
	for(int i=0; i<npars_; i++){
		assert(omega0(i) > 0 && Qpar(i) > 0 && varf(i) > 0 && "omega0, Qpar and varf ought to be positive");
	}
	omega0_ = omega0;
	Qpar_	= Qpar;
	varf_ = varf;
	varw_ = Eigen::VectorXd::Zero(npars_);
	get_varw(varf_, omega0_, Qpar_, varw_);
	initial_mean_variance_N_DSHO(omega0_, Qpar_, varw_, x1, P1);
	Q = Eigen::MatrixXd::Zero(2*npars_,2*npars_);
	Phi = Eigen::MatrixXd::Zero(2*npars_, 2*npars_);
}

void set_data (const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr)
{
	times_ = times;
	y_i_ = y_i;
	yerr_ = yerr;
};


void Phi_N_DSHO(int i, const Eigen::VectorXd& times,	
		const Eigen::VectorXd& omega, const Eigen::VectorXd& Qpar, 
		Eigen::MatrixXd& Phi)
{
	double delta = deltaF(i,times);
	double beta, cc, ss;
	for (int i=0; i<npars_; i++)
	{
		int l = 2*i;
		double ef = std::exp(-omega(i) * delta / (2*Qpar(i)));
		if ((std::isfinite(ef)) && (ef > 1e-10))
		{	
			if (Qpar(i) >= 0.5)
			{
				beta = std::sqrt(Qpar(i)*Qpar(i) - 0.25);
				if (beta < 1e-2)
					beta = 1e-2;
				cc = std::cos(omega(i) * beta * delta / Qpar(i));
				ss = std::sin(omega(i) * beta * delta / Qpar(i));
			}
			else
			{
				beta = std::sqrt(0.25 - Qpar(i)*Qpar(i));
				if (beta < 1e-2)
					beta = 1e-2;
				cc = std::cosh(omega(i) * beta * delta / Qpar(i));
				ss = std::sinh(omega(i) * beta * delta / Qpar(i));

			}
			Phi(l,l) = ef * (cc + ss/(2*beta));
			Phi(l,l+1) = ef * Qpar(i) * ss / (omega(i) * beta);
			Phi(l+1,l) = -ef * Qpar(i) * omega(i) * ss / beta;
			Phi(l+1,l+1) = ef * (cc-ss/(2*beta));
		}
		else
		{
			Phi(l,l) = 0.0;
			Phi(l,l+1) = 0.0;
			Phi(l+1,l) = 0.0;
			Phi(l+1,l+1) = 0.0;
		}
	}	
};

void Q_N_DSHO(int i, const Eigen::VectorXd& times,	
		const Eigen::VectorXd& omega, 
		const Eigen::VectorXd& Qpar, const Eigen::VectorXd& varw,
		Eigen::MatrixXd& Q)
{
	double delta = deltaF(i,times);
	double beta,cc,ss,cc2,ss2;
	for (int i=0; i<npars_; i++)
	{
		int l = 2*i;
		double texp =	std::exp(-omega(i)*delta/Qpar(i));
		if ((std::isfinite(texp)) && (texp > 1e-10))
		{
			texp *= Qpar(i);
			if (Qpar(i) > 0.5)
			{
				beta = std::sqrt(Qpar(i)*Qpar(i) - 0.25);
				if (beta < 1e-2)
					beta = 1e-2;
				double argu = omega(i) * beta * delta / Qpar(i);
				cc = std::cos(argu);
				ss = std::sin(argu);
				cc2 = std::cos(2*argu);
				ss2 = std::sin(2*argu);
			}
			else
			{
				beta = std::sqrt(0.25 - Qpar(i)*Qpar(i));
				if (beta < 1e-2)
					beta = 1e-2;
				double argu = omega(i) * beta * delta / Qpar(i);
				cc = std::cosh(argu);
				ss = std::sinh(argu);
				cc2 = std::cosh(2*argu);
				ss2 = std::sinh(2*argu);		
			}
			if (Qpar(i) > 0.5)
			{
				Q(l,l) = varw(i) * texp * ((cc2-1.0) - 2.0*beta*ss2 + 4*std::pow(beta,2)*(std::exp(omega(i)*delta/Qpar(i))-1))/(8.0*std::pow(omega(i),3)*std::pow(beta,2));
			}
			else
			{
				Q(l,l) = varw(i) * texp * ((1.0-cc2) - 2.0*beta*ss2 + 4*std::pow(beta,2)*(std::exp(omega(i)*delta/Qpar(i))-1))/(8.0*std::pow(omega(i),3)*std::pow(beta,2));			
			}
			Q(l+1,l) = varw(i) * texp * (Qpar(i)*std::pow(ss,2))/(2*std::pow(omega(i),2)*std::pow(beta,2));
			Q(l,l+1) = Q(l+1,l);
			if (Qpar(i) > 0.5)
			{		
				Q(l+1,l+1) = varw(i) * texp * ((cc2-1.0) + 2.0*beta*ss2 + 4*std::pow(beta,2)*(std::exp(omega(i)*delta/Qpar(i))-1))/(8.0*omega(i)*std::pow(beta,2));
			}
			else
			{
				Q(l+1,l+1) = varw(i) * texp * ((1.0-cc2) + 2.0*beta*ss2 + 4*std::pow(beta,2)*(std::exp(omega(i)*delta/Qpar(i))-1))/(8.0*omega(i)*std::pow(beta,2));			
			}
		}
		else
		{
			Q(l,l)=0.0;
			Q(l+1,l)=0.0;
			Q(l,l+1)=0.0;
			Q(l+1,l+1)=0.0;			
		}
	}
};

void H_N_DSHO(Eigen::RowVectorXd& h)
{
	h = Eigen::RowVectorXd::Zero(2*npars_);
	for (int i=0; i<npars_; i++)
		h(2*i) = 1.0;	
};

void R_N_DSHO(int i, const Eigen::VectorXd& sigma, Eigen::Matrix<double,1,1>& r)
{
	double sigmat = sigma(i);
	r(0) = sigmat * sigmat;
};

void initial_mean_variance_N_DSHO(const Eigen::VectorXd& omega, 
	const Eigen::VectorXd& Qpar, const Eigen::VectorXd& varw,
	Eigen::VectorXd& x1, Eigen::MatrixXd& P1)
{
	x1 = Eigen::VectorXd::Zero(2*npars_);
	P1 = Eigen::MatrixXd::Zero(2*npars_,2*npars_);
	for (int i=0; i<npars_; i++)
	{
		P1(2*i,2*i) = varw(i) * Qpar(i) / (2.0 * std::pow(omega(i),3));
		P1(2*i+1,2*i+1) = varw(i) * Qpar(i) / (2.0 * omega(i));
	}
};

double KF_log_likelihood()
{
	return KF_log_likelihood_(times_, y_i_, yerr_, omega0_, Qpar_, varw_);
}


double KF_log_likelihood_(const Eigen::VectorXd& times, 
	const Eigen::VectorXd& y_i, const Eigen::VectorXd& yerr,
	const Eigen::VectorXd& omega0, const Eigen::VectorXd& Qpar, 
	const Eigen::VectorXd& varw)
{
	int n = times.rows();
	assert( y_i.rows() == n && yerr.rows() == n && "dimension mismatch between y and yerr");
	double ll = 0.0;
	double dll;


	Eigen::VectorXd x_pred, x_filt, x_filt_old;
	Eigen::MatrixXd P_pred, P_filt, P_filt_old, Tmp;

	Tmp = Eigen::MatrixXd::Zero(2*npars_, 2*npars_);
	P_filt_old = Eigen::MatrixXd::Zero(2*npars_, 2*npars_);
	P_filt = Eigen::MatrixXd::Zero(2*npars_, 2*npars_);
	x_filt = Eigen::VectorXd::Zero(2*npars_);
	x_pred = Eigen::VectorXd::Zero(2*npars_);
	P_pred = Eigen::MatrixXd::Zero(2*npars_, 2*npars_);


	//H_N_DSHO(H_i);
	R_N_DSHO(0, yerr, R_i);

	kalman_recursions_NDSHO(2*npars_, y_i(0), R_i, x1, P1, dll, x_filt_old, P_filt_old);

	ll += dll;
	for (int j=1; j<n; j++)
	{
		Phi_N_DSHO(j, times, omega0, Qpar, Phi);
		Q_N_DSHO(j, times, omega0, Qpar, varw, Q);
		for (int i=0; i<2*npars_; i+=2)
		{
			x_pred(i) = Phi(i,i)*x_filt_old(i) + Phi(i,i+1)*x_filt_old(i+1);
			x_pred(i+1) = Phi(i+1,i)*x_filt_old(i) + Phi(i+1,i+1)*x_filt_old(i+1);
		}
		//x_pred = Phi * x_filt_old;
		for (int i=0; i<2*npars_; i+=2)
		{	Tmp(i,i) = P_filt_old(i,i) * Phi(i,i) + P_filt_old(i,i+1)*Phi(i,i+1);
			Tmp(i,i+1) = P_filt_old(i,i) * Phi(i+1,i) + P_filt_old(i,i+1) * Phi(i+1,i+1);
			Tmp(i+1, i) = P_filt_old(i+1,i) * Phi(i,i) + P_filt_old(i+1,i+1) * Phi(i,i+1);
			Tmp(i+1, i+1) = P_filt_old(i+1,i) * Phi(i+1,i) + P_filt_old(i+1,i+1) * Phi(i+1,i+1);
		}		
		for (int i=0; i<2*npars_; i+=2)
		{
			P_pred(i,i) = Phi(i,i) * Tmp(i,i) + Phi(i,i+1) * Tmp(i+1,i) + Q(i,i);
	 		P_pred(i,i+1) = Phi(i,i) * Tmp(i,i+1) + Phi(i, i+1) * Tmp(i+1, i+1) + Q(i,i+1);
			P_pred(i+1,i) = Phi(i+1,i) * Tmp(i,i) + Phi(i+1,i+1) * Tmp(i+1,i) + Q(i+1,i);
			P_pred(i+1,i+1) = Phi(i+1,i) * Tmp(i,i+1) + Phi(i+1,i+1) * Tmp(i+1,i+1) + Q(i+1,i+1);
		}
		//P_pred = Phi * P_filt_old * Phi.transpose() + Q;
 	//H_N_DSHO(H_i);
	 	R_N_DSHO(j,yerr, R_i);
		kalman_recursions_NDSHO(2*npars_, y_i(j), R_i, x_pred, P_pred, dll, x_filt, P_filt);
		x_filt_old = x_filt;
		P_filt_old = P_filt;
		ll += dll;
	}

	return ll;

};

void simulate_N_DSHO(Eigen::VectorXd& y_sim, std::mt19937 rng){
	simulate_N_DSHO(times_, y_sim, false, rng);
}


void simulate_N_DSHO(const Eigen::VectorXd& simTimes, 
					Eigen::VectorXd& y_sim, const bool zeroQ, std::mt19937 rng)
{
	int n = simTimes.rows();
	assert(n>1 && "please simulate at least two points!");

	y_sim = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd x_sim,x_sim_old, MVdev, tmpVec;
	Eigen::MatrixXd Q_chol, R_chol;
	
		Eigen::MatrixXd transform;

		tmpVec = Eigen::VectorXd::Zero(2*npars_);

	//std::random_device rd;
	//std::mt19937 rng(rd());
	std::normal_distribution<double> gaussiana;


	double sigma_r;

		H_N_DSHO(H_i);
		R_N_DSHO(0, yerr_, R_i);

		sigma_r = std::sqrt(R_i.sum()); // R_i is 1d in our case, make it a scalar

	y_sim(0) = H_i * x1 + sigma_r * gaussiana(rng);
	x_sim_old = x1;
	if (!zeroQ)
	{
		for(int j=1; j<n; j++)
		{
			R_N_DSHO(j, yerr_, R_i);
		 	Phi_N_DSHO(j, times_, omega0_, Qpar_, Phi);
			Q_N_DSHO(j, times_, omega0_, Qpar_, varw_, Q);
			for(int k = 0; k<2*npars_; k++)
				Q(k,k)+=1e-10;
			// get MV deviate with covariance matrix Q
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(Q);
			transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
			for (int k = 0; k<2*npars_; k++)
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
			R_N_DSHO(j, yerr_, R_i);
		 	Phi_N_DSHO(j, times_, omega0_, Qpar_, Phi);
		 	x_sim = Phi * x_sim_old;
		 	sigma_r = std::sqrt(R_i.sum());
			y_sim(j) = H_i*x_sim + sigma_r * gaussiana(rng);
			x_sim_old = x_sim;		 
		}	
	}

};

void resample_parameters(Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood, Eigen::MatrixXd& logomega0_array, Eigen::MatrixXd logQpar_array, Eigen::MatrixXd& logvarf_array, const int n_part, const int npars_, std::mt19937 rng) {
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
}

void construct_covariance(Eigen::MatrixXd& logomega0_array, 
	Eigen::MatrixXd logQpar_array, Eigen::MatrixXd& logvarf_array, 
	const int n_part, const int npars_, 
	Eigen::MatrixXd& transform, Eigen::VectorXd& mu){
	std::cout << "computing covariance ..." << std::endl;
	Eigen::MatrixXd mat(n_part,3*npars_);
	//Eigen::VectorXd mu(3*npars_);

	for(int k=0; k < npars_; k++)
	{
		mu(3*k)=logomega0_array.col(k).mean();
		mu(3*k+1)=logQpar_array.col(k).mean();
		mu(3*k+2)=logvarf_array.col(k).mean();
	}	

	for (int k=0; k < n_part; k++)
	{
		for (int kk=0; kk<npars_; kk++)
		{
			mat(k,3*kk) = logomega0_array(k,kk) - mu(3*kk);
			mat(k,3*kk+1) = logQpar_array(k,kk) - mu(3*kk+1);				
			mat(k,3*kk+2) = logvarf_array(k,kk) - mu(3*kk+2);
		}
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
	Eigen::MatrixXd& omega0_lims, Eigen::MatrixXd& Qpar_lims, Eigen::MatrixXd& varf_lims, 
	std::mt19937 rng, const int npars_, Eigen::MatrixXd& transform, Eigen::VectorXd& mu
) {
	Eigen::VectorXd deviate(mu.size());
	Eigen::VectorXd tmpVec(mu.size());
	std::normal_distribution<double> gaussiana;
	bool cond = false;
	while(!cond) {
		cond = true;
		for (int kk=0; kk < mu.size(); kk++)
			tmpVec(kk) = gaussiana(rng);
		
		deviate = transform * tmpVec;

		for (int kk=0; kk < npars_; kk++)
		{
			deviate(3*kk) += mu(3*kk); 
			double omega0 = std::exp( deviate(3*kk) );
			if ((omega0 < omega0_lims(0,kk)) || (omega0 > omega0_lims(1,kk)))
			{
				cond=false;
				continue;
			}
			deviate(3*kk+1) += mu(3*kk+1);
			double Qpar	 = std::exp( deviate(3*kk + 1) );
			if ((Qpar < Qpar_lims(0,kk)) || (Qpar > Qpar_lims(1,kk)))
			{
				cond=false;
				continue;
			}
			deviate(3*kk+2) += mu(3*kk+2);
			double varf	 = std::exp( deviate(3*kk + 2) );
			if ((varf < varf_lims(0,kk)) || (varf > varf_lims(1,kk)))
			{
				cond=false;
			}
		}
	} 
	return deviate;
}

void move_particles(int i, 
	Eigen::MatrixXd& logomega0_array, Eigen::MatrixXd logQpar_array, 
	Eigen::MatrixXd& logvarf_array, 
	Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood,
	Eigen::MatrixXd& omega0_lims, Eigen::MatrixXd& Qpar_lims, Eigen::MatrixXd& varf_lims, 
	std::mt19937 rng, const int n_part, const int npars_, Eigen::MatrixXd& transform, Eigen::VectorXd& mu
) {
	std::cout << "updating particles..." << std::endl;
	Eigen::VectorXd omega0, Qpar, varf, varw;
	omega0 = Eigen::VectorXd::Zero(npars_);
	Qpar  = Eigen::VectorXd::Zero(npars_);
	varf  = Eigen::VectorXd::Zero(npars_);
	varw  = Eigen::VectorXd::Zero(npars_);
	
	std::normal_distribution<double> gaussiana;
	std::uniform_real_distribution<double> uniform(0, 1);
	for (int k = 0; k < n_part; k++)
	{
		std::cout << k << " ... \r";
		mu(0) = logomega0_array(k);
		mu(1) = logQpar_array(k);
		mu(2) = logvarf_array(k);
		Eigen::VectorXd deviate = sample_gaussian_within_limits(
			omega0_lims, Qpar_lims, varf_lims,
			rng, npars_, transform, mu);

		for (int kk=0; kk<npars_; kk++)
		{
			omega0(kk) = std::exp( deviate(3*kk) );
			Qpar(kk)	 = std::exp( deviate(3*kk + 1) );
			varf(kk)	 = std::exp( deviate(3*kk + 2) );
		}

		get_varw(varf, omega0, Qpar, varw);
		double res_proposal = KF_log_likelihood_(times_.head(i), y_i_.head(i), yerr_.head(i), omega0, Qpar, varw);
		double log_like_diff = res_proposal - log_likelihood(k);
		double ratio = std::min(std::exp(log_like_diff),1.0);
		double u_dev = uniform(rng);
		if (u_dev < ratio)
		{
			// we accept the proposal
			for (int kk=0; kk<npars_; kk++)
			{
				logomega0_array(k,kk) = deviate(3*kk);
				logQpar_array(k,kk)	 = deviate(3*kk+1);
				logvarf_array(k,kk)	 = deviate(3*kk+2);
			}
			log_likelihood(k)	= res_proposal;
		}
	}
	// resampling/remove done, reset weights to uniform values
	weights = Eigen::VectorXd::Ones(n_part) / n_part;
	std::cout << std::endl;
	std::cout << "updating particles done." << std::endl;
}

void SeqMC_N_DSHO(Eigen::MatrixXd& logomega0_array, Eigen::MatrixXd logQpar_array, 
				Eigen::MatrixXd& logvarf_array, 
				Eigen::VectorXd& weights, Eigen::VectorXd& log_likelihood,
				Eigen::MatrixXd& omega0_lims, Eigen::MatrixXd& Qpar_lims, Eigen::MatrixXd& varf_lims, 
				int save_interval, std::mt19937 rng)
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
	Eigen::MatrixXd transform(n_part,3*npars_);
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
		if (ESS < ess_threshold)
		{
			std::cout << "In iteration " << i << " we entered a resample/move stage" << std::endl;
			resample_parameters(weights, log_likelihood, logomega0_array, logQpar_array, logvarf_array, n_part, npars_, rng);
			construct_covariance(logomega0_array, logQpar_array, logvarf_array, n_part, npars_, transform, mu);
			move_particles(i, logomega0_array, logQpar_array, logvarf_array,
				weights, log_likelihood, 
				omega0_lims, Qpar_lims, varf_lims,
				rng, n_part, npars_, transform, mu);
			resample = true;
		}
		std::cout << "exp of mean of omega(0), Q(0), varf(0): " << std::exp(logomega0_array.col(0).dot(weights)) << " ";
		std::cout << std::exp(logQpar_array.col(0).dot(weights)) << " ";
		std::cout << std::exp(logvarf_array.col(0).dot(weights)) << std::endl;
		if (npars_ > 1)
		{
			std::cout << "exp of mean of omega(1), Q(1), varf(1): " << std::exp(logomega0_array.col(1).dot(weights)) << " ";	
			std::cout << std::exp(logQpar_array.col(1).dot(weights)) << " ";
			std::cout << std::exp(logvarf_array.col(1).dot(weights)) << std::endl;	
		}
		std::cout << "max of log likelihood: " << log_likelihood.maxCoeff() << std::endl;

		if ((i%save_interval == 0) || (resample) || (i==(n-1)))
		{
			std::string path = "data/ndsho_post_" + std::to_string(i) + ".txt";
			if (resample)
				path = "data/ndsho_post_re_" + std::to_string(i) + ".txt";
			if (npars_ == 1)
			{
				path = "data/dsho_post_" + std::to_string(i) + ".txt";
				if (resample)
					path = "data/dsho_post_re_" + std::to_string(i) + ".txt";
			}
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
};


private:

	int npars_;
	Eigen::VectorXd omega0_, Qpar_, varf_, varw_;

	Eigen::VectorXd times_, y_i_, yerr_; 
	Eigen::MatrixXd Q, Phi, P1;
	Eigen::RowVectorXd H_i;
	Eigen::Matrix<double,1,1> R_i;
	Eigen::VectorXd x1;
	
	
	


}; // class DSHOSolver


}; // namespace n_dsho
}; // namespace gpstate

#endif //_GPSTATE_N_DSHO_H
