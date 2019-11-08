#include <iostream>
#include <sys/time.h>
#include <Eigen/Core>
#include <fstream>

#include "celerite/celerite.h"
#include "celerite/carma.h"
#include "celerite/utils.h"
#include "../include/KF.h"
#include "../include/ndsho.h"
#include "../include/dsho.h"


using namespace Eigen;
using namespace std;

#define TWOPI 6.283185307179586

// This program benchmarks a single DSHO using celerite and gpstate

// Timer for the benchmark.
double get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return double(now.tv_usec) * 1.0e-6 + double(now.tv_sec);
}

// Function to read a whitespace separated file
std::vector<double> load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            //cout << std::stod(cell) << endl;
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return values;
}



int main (int argc, char* argv[])
{
  srand(42);

  // we will benchmark models with up to N_max DSHO components
  size_t N_max = 64;
  //N_max=1;
  if (argc >= 2) N_max = atoi(argv[1]);
  // dfor each sample size, timeing will be done niter times
  size_t niter = 5;
  if (argc >= 3) niter = atoi(argv[2]);

  std::random_device rd;
  std::mt19937 rng(rd());
  std::normal_distribution<double> gaussian(0.0,0.05);

  int vecsize = 10000;
	// map data to VectorXds
	VectorXd yi(vecsize);
  VectorXd times = Eigen::VectorXd::Random(vecsize);
  times.array() += 1.0;
  times.array() *= (10*0.5);
  std::sort(times.data(), times.data() + times.size());
  double omega0 = TWOPI;

  for (int j=0; j<times.size(); j++){
      //times(j) = 0.001*double(j);
      yi(j) = sin(omega0 * times(j)) + gaussian(rng);
  }

	// this is the observational error vector
	VectorXd yierr = VectorXd::Ones(yi.size());
	yierr.array() *= 0.05;



  // Generate some fake data.
  //size_t N_points = 4096;
  //Eigen::VectorXd x = Eigen::VectorXd::Random(N_points),
  //                yerr = Eigen::VectorXd::Random(N_points),
  //                y, diag;
  //yerr.array() *= 0.1;
  //yerr.array() += 1.0;
  //diag = yerr.array() * yerr.array();
  //std::sort(x.data(), x.data() + x.size());
  //y = sin(x.array());

  //std::vector<double> values = load_csv("two_comp_dsho.txt");
  //std::vector<double> values = load_csv("GPtest10_dsho_trim.txt");


  //VectorXd times = Map<VectorXd, 0, InnerStride<2> > (values.data(), 10000);
  //VectorXd yi = Map<VectorXd, 0, InnerStride<2> > (values.data()+1, 10000);
  //VectorXd yierr = VectorXd::Ones(yi.size());
  //yierr *= 0.01;
  VectorXd diagi = yierr.array()*yierr.array();



  //set up the single DSHO parameters we use as a base
  //double omega0 = 0.785398;
  double Q = 10.0;
  double varf = 0.05;

  double celerite_time = 0.0;
  double gpstate_time = 0.0;
  double strt;

  Eigen::VectorXd alpha_real, beta_real;

  for (size_t N = 1; N <= N_max; N +=1) {
    // define arrays
    // The DSHO is a CARMA model, with parameters as written below
    Eigen::VectorXd omega0_arr(N), Q_arr(N), varf_arr(N);
    Eigen::VectorXd alpha_complex_real_arr(N), alpha_complex_imag_arr(N), beta_complex_real_arr(N), beta_complex_imag_arr(N);
    double log_likelihood=0.0, celerite_ll=0.0;

    int nterms = 3;

    for (size_t i=0; i<N; i++)
    {
      if (i == 0)
      {
        omega0_arr(i) = omega0; //0.785398; //0.1234234;
      }
      else
      {
        omega0_arr(i) = omega0 + static_cast<double>(i);
      }
      Q_arr(i) = Q;
      varf_arr(i) = varf;
      Eigen::VectorXd carma_arparams(nterms);
      Eigen::VectorXd carma_maparams(nterms-1);
      carma_arparams << omega0_arr(i)*omega0_arr(i), omega0_arr(i)/Q, 1.0;
      carma_maparams << 1.0, 0.0;

      // version that used to work!
      double temp = std::sqrt(4.0*Q_arr(i)*Q_arr(i) - 1.0);
      //double S0 = varf_arr(i)* std::pow(Q_arr(i),-2) * std::sqrt(M_PI) / std::sqrt(2);
      double S0 = varf_arr(i)* std::pow(omega0_arr(i),-4) * std::sqrt(M_PI) / std::sqrt(2);
      alpha_complex_real_arr(i) = S0 * omega0_arr(i) * Q_arr(i);
      alpha_complex_imag_arr(i) = S0 * omega0_arr(i) * Q_arr(i) / temp;
      beta_complex_real_arr(i) = 0.5 * omega0_arr(i) / Q_arr(i);
      beta_complex_imag_arr(i) = 0.5 * temp * omega0_arr(i) / Q_arr(i);
    }


    celerite::solver::CholeskySolver<double> solver;

    for (size_t i = 0; i < niter; ++i) {
      strt = get_timestamp();
      //int flag = solver.compute(alpha_real_arr, beta_real_arr, alpha_complex_real_arr, alpha_complex_imag_arr, beta_complex_real_arr, beta_complex_imag_arr, x, diag);
      //celerite_ll = -0.5*(solver.dot_solve(y) + solver.log_determinant() + x.rows() * log(2.0 * M_PI));

      solver.compute(0.0, alpha_real, beta_real, alpha_complex_real_arr, alpha_complex_imag_arr, beta_complex_real_arr, beta_complex_imag_arr, times, diagi);
      celerite_ll = -0.5*(solver.dot_solve(yi) + solver.log_determinant() + times.rows() * log(2.0 * M_PI));
      celerite_time += get_timestamp()-strt;
    }

    //gpstate::n_dsho::N_DSHOSolver ndsho(x,y,yerr,omega0_arr, Q_arr, varf_arr);
    gpstate::n_dsho::N_DSHOSolver ndsho(times,yi,yierr,omega0_arr, Q_arr, varf_arr);
    for (size_t i = 0; i < niter; ++i) {
      strt = get_timestamp();
      log_likelihood = ndsho.KF_log_likelihood();
      gpstate_time += get_timestamp()-strt;
    }


    // Print the results.
    std::cout << N;
    std::cout << " ";
    std::cout << celerite_time / niter;
    std::cout << " ";
    std::cout << gpstate_time / niter;
    std::cout << " ";
    std::cout << celerite_ll;
    std::cout << " ";
    std::cout << log_likelihood;
    std::cout << " ";

    std::cout << "\n";
  }
}
