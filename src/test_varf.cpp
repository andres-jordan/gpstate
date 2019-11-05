#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>

#include "../include/KF.h"
#include "../include/dsho.h"
#include "../include/matern32.h"
#include "../include/ndsho.h"


using namespace Eigen;
using namespace std;

// function to read a space separated file

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

int main()
{

std::random_device rd;
std::mt19937 rng(rd());


// read file with two component damped simple harmonic oscillator simulated data
//std::vector<double> values = load_csv("two_comp_dsho.txt");
//cout << values.size() << endl;


// map data to VectorXds
//VectorXd times = Map<VectorXd, 0, InnerStride<2> > (values.data(), 10000);
//VectorXd yi = Map<VectorXd, 0, InnerStride<2> > (values.data()+1, 10000);
int vecsize = 10000;
VectorXd yi(vecsize);
VectorXd times = Eigen::VectorXd::Random(vecsize);
times.array() += 1.0;
times.array() *= (100*0.5);
std::sort(times.data(), times.data() + times.size());

// this is the observational error vector
VectorXd yerr = VectorXd::Ones(yi.size());

double obs_err = 0.05;
yerr.array() *= obs_err;

std::uniform_real_distribution<double> uniform(0, 5);

// Now we run a bunch of tests

// create a Matern 3/2 solver and get likelihood of data
gpstate::matern32::Matern32Solver m32(times, yi, yerr,std::sqrt(10), 1.0);
double log_likelihood = m32.KF_log_likelihood();
cout << "logL: " << log_likelihood << endl;

// simulate a Matern 3/2 on the times given by VectorXd times
VectorXd y_sim;
m32.simulate_Matern32(y_sim, rng);

// set parameters for a 2 component DSHO model
Eigen::Vector2d omegas,Qpars,varfs;
omegas << 1, 3*0.15915494309;
Qpars << 10, 10;
varfs << 1, 1;

// set parameters for a single DSHO model
double omega0=12;
double Q=1;
double varf=1;

// simulate a single DSHO with parameters omega0, Q, varf
// we simulate for various values of Q, and measure the simulated variance
// objective is to verify that the variance specified by 'varf'
// corresponds indeed to the actual simulated variance
// (in other words, we are verifying the normalizations of our GPs)
gpstate::dsho::DSHOSolver dsho(times,yi,yerr,omega0, Q, varf);
dsho.simulate_DSHO(y_sim);

double mean = y_sim.mean();
Eigen::VectorXd tmp = y_sim.array()-mean;
double variance  = tmp.dot(tmp) / y_sim.rows();
std::cout << "var1:" << variance << std::endl;

Q = 10;
dsho.set_pars(omega0,Q,varf);
dsho.simulate_DSHO(y_sim);
mean = y_sim.mean();
tmp = y_sim.array()-mean;
variance  = tmp.dot(tmp) / y_sim.rows();
std::cout << "var2:" << variance << std::endl;

Q = 0.1;
dsho.set_pars(omega0,Q,varf);
dsho.simulate_DSHO(y_sim);
mean = y_sim.mean();
tmp = y_sim.array()-mean;
variance  = tmp.dot(tmp) / y_sim.rows();
std::cout << "var3:" << variance << std::endl;

Q=10;
omega0=6;
dsho.set_pars(omega0,Q,varf);
dsho.simulate_DSHO(y_sim);
mean = y_sim.mean();
tmp = y_sim.array()-mean;
variance  = tmp.dot(tmp) / y_sim.rows();
std::cout << "var4:" << variance << std::endl;

omega0=120;
dsho.set_pars(omega0,Q,varf);
dsho.simulate_DSHO(y_sim);
mean = y_sim.mean();
tmp = y_sim.array()-mean;
variance  = tmp.dot(tmp) / y_sim.rows();
std::cout << "var5:" << variance << std::endl;
log_likelihood = dsho.KF_log_likelihood();
cout << "logL: " << log_likelihood << endl;

//std::exit(0);

Eigen::Matrix<double,1,1> omega_one, Qpar_one, varf_one;
omega_one << 120;
Qpar_one << 10;
varf_one << 1;

// Verify that the ndsho implementation is consistent with the dsho one
gpstate::n_dsho::N_DSHOSolver ndsho(times,yi,yerr,omega_one, Qpar_one, varf_one);
ndsho.simulate_N_DSHO(y_sim, rng);
mean = y_sim.mean();
tmp = y_sim.array()-mean;
variance  = tmp.dot(tmp) / y_sim.rows();
std::cout << "var6:" << variance << std::endl;
log_likelihood = ndsho.KF_log_likelihood();
cout << "logL: " << log_likelihood << endl;

// write simulated vector to file GPtest.txt
std::ofstream file("GPtest.txt");
if (file.is_open())
  {
    file << "#time y_sim" << endl;
    for(int i=0; i<times.rows(); i++)
    {
        file <<  times(i) <<" "<< y_sim(i) << endl;
    }
  }


}
