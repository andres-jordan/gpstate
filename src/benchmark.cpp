#include <iostream>
#include <sys/time.h>
#include <Eigen/Core>

#include "celerite/celerite.h"
//#include "celerite/carma.h"
#include "celerite/utils.h"
#include "../include/KF.h"
#include "../include/dsho.h"

// This code benchmarks a single DSHO, celerite vs gpstate

// Timer for the benchmark.
double get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return double(now.tv_usec) * 1.0e-6 + double(now.tv_sec);
}

int main (int argc, char* argv[])
{
  srand(42);

  size_t N_max = pow(2, 19);
  if (argc >= 2) N_max = atoi(argv[1]);
  size_t niter = 5;
  if (argc >= 3) niter = atoi(argv[2]);
  size_t niter_celerite = niter;
  if (argc >= 4) niter_celerite = atoi(argv[3]);
  std::cout << "N:" << N_max << " gpstate:" << niter << " celerite:" << niter_celerite << std::endl;

 // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N_max),
                  yerr = Eigen::VectorXd::Random(N_max),
                  y, diag;
  yerr.array() *= 0.1;
  yerr.array() += 1.0;
  diag = yerr.array() * yerr.array();
  std::sort(x.data(), x.data() + x.size());
  y = sin(x.array());

  //set up the DSHO parameters
  size_t nterms=3;
  double omega0=1.0;
  double Q = 1.0;
  double varf = 1.0;
  int flag;

  // translate into corresponding CARMA parameters for use in celerite
  Eigen::VectorXd carma_arparams(nterms);
  Eigen::VectorXd carma_maparams(nterms-1);
  carma_arparams << omega0*omega0, omega0/Q, 1.0;
  carma_maparams << 1.0, 0.0;

  Eigen::VectorXd alpha_real, beta_real;
  Eigen::VectorXd alpha_complex_real(1), alpha_complex_imag(1),
                    beta_complex_real(1), beta_complex_imag(1);

  double temp = std::sqrt(4.0*Q*Q - 1.0);
  double S0 = varf* std::pow(Q,-2) * std::sqrt(M_PI) / std::sqrt(2);
  alpha_complex_real(0) = S0 * omega0 * Q;
  alpha_complex_imag(0) = S0 * omega0 * Q/temp;
  beta_complex_real(0) = 0.5*omega0 / Q;
  beta_complex_imag(0) = 0.5*temp*omega0 / Q;


//f = np.sqrt(4.0 * Q**2-1)
 //       return (
 //           S0 * w0 * Q,
 //           S0 * w0 * Q / f,
 //           0.5 * w0 / Q,
 //           0.5 * w0 / Q * f,
 //       )

  // the following pieces of code verify that the log likelihoods are
  // consistent between celerite and gpstate
  celerite::solver::CholeskySolver<double> solver;
  solver.compute(0.0, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
  double celerite_ll = -0.5*(solver.dot_solve(y) + solver.log_determinant() + x.rows() * log(2.0 * M_PI));
  std::cout << celerite_ll << std::endl;

  gpstate::dsho::DSHOSolver dsho(x,y,yerr,omega0, Q, 1.0);
  double log_likelihood = dsho.KF_log_likelihood();
  std::cout << "logL: " << log_likelihood << std::endl;



  // now benchmark, knowing that we do get consistent log-likelohood values
  double strt;

for (size_t N = 64; N <= N_max; N *= 2) {

  double celerite_time = 0.0;
  double gpstate_time = 0.0;

  if (niter_celerite > 0) {
    //celerite::solver::CholeskySolver<double> solver;
    //solver.compute(0.0, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
    //celerite_ll = -0.5*(solver.dot_solve(y) + solver.log_determinant() + x.rows() * log(2.0 * M_PI));
    //std::cout << "logL(celerite)" << celerite_ll << std::endl;

    for (size_t i = 0; i < niter_celerite; ++i) {
      strt = get_timestamp();
      solver.compute(0.0, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x.head(N), diag.head(N));
      celerite_ll = -0.5*(solver.dot_solve(y.head(N)) + solver.log_determinant() + x.head(N).rows() * log(2.0 * M_PI));
      celerite_time += get_timestamp()-strt;
    }
  }
  if (niter > 0) {
    //gpstate::dsho::DSHOSolver dsho1(x,y,yerr,omega0, Q, 1.0);
    //log_likelihood = dsho1.KF_log_likelihood();
    //std::cout << "logL: " << log_likelihood << std::endl;

    for (size_t i = 0; i < niter; ++i) {
      strt = get_timestamp();
      gpstate::dsho::DSHOSolver dsho(x.head(N),y.head(N),yerr.head(N),omega0, Q, 1.0);
      double log_likelihood = dsho.KF_log_likelihood();
      gpstate_time += get_timestamp()-strt;
      }
    }
   // Print the results.
   if ((niter > 0) & (niter_celerite > 0)){
     std::cout << N;
     std::cout << " ";
     std::cout << celerite_time / niter_celerite;
     std::cout << " ";
     std::cout << gpstate_time / niter;
     std::cout << "\n";
   }
}
}
