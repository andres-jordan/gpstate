#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

#include "../include/KF.h"
#include "../include/dsho.h"
#include "../include/matern32.h"
#include "../include/ndsho.h"


using namespace Eigen;
using namespace std;

#define TWOPI 6.283185307179586

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

int main() {

	// read file with two component damped simple harmonic oscillator simulated data
	//std::vector<double> values = load_csv("two_comp_dsho.txt");
	//cout << values.size() << endl;

	// map data to VectorXds
	//VectorXd times = Map<VectorXd, 0, InnerStride<2> > (values.data(), 10000);
	//VectorXd yi = Map<VectorXd, 0, InnerStride<2> > (values.data()+1, 10000);


  int vecsize = 10000;
	// map data to VectorXds
	VectorXd yi(vecsize);
  VectorXd times = Eigen::VectorXd::Random(vecsize);
  times.array() += 1.0;
  times.array() *= (100*0.5);
  std::sort(times.data(), times.data() + times.size());


	// this is the observational error vector
	VectorXd yerr = VectorXd::Ones(yi.size());
	yerr.array() *= 0.05;

	VectorXd y_sim, y_sim_one;

	for(int j=0; j < 24; j++) {
		//auto const seed = std::random_device()();
		int seed = 123;
		std::mt19937 rng(seed);
		cout << seed << endl;
		// set parameters for a 2 component DSHO model
		Eigen::Vector2d omegas,Qpars,varfs;
		Eigen::Matrix<double,1,1> omega_one, Qpar_one, varf_one;
		double omega0 = 8; // this component stays fixed
		double period_list[6] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
		std::cout << j/6 << j%6 << std::endl;
		double omega1 = TWOPI / period_list[j%6];
		double Q0 = 10.0;
		double Q1 = 10.0;
		double varf0 = 1.0;
		double varf1 = 1.0;
		if ((j/6 == 1) || (j/6 ==3))
		{
			varf0 = 0.01;
			varf1 = 0.01;
		}
		if (j/6 > 1)
		{
			Q1 = 1.0;
		}
		omegas << omega0, omega1;
		Qpars << Q0, Q1;
		varfs << varf0, varf1;

		omega_one << omega1;
		Qpar_one << Q1;
		varf_one << varf1;

		// Simulate a 2 component DSHO model
		gpstate::n_dsho::N_DSHOSolver ndsho(times,yi,yerr,omegas, Qpars, varfs);
		ndsho.simulate_N_DSHO(y_sim, rng);

		// For efficiency, just output one of the components as well (the varying one)
		gpstate::n_dsho::N_DSHOSolver ndsho_one(times,yi,yerr,omega_one, Qpar_one, varf_one);
		ndsho_one.simulate_N_DSHO(y_sim_one, rng);

		// write simulated vector to file GPtest.txt
		std::ofstream file("GPtest" + std::to_string(j+1) + "_ndsho.txt");
		assert(file.is_open());
		file << "# N_DSHO simulated light curve" << endl;
		file << "# input parameters: " << endl;
		file << "# omega: " << omega0 << ", " << omega1 << endl;
		file << "# Q: " << Q0 << ", " << Q1 << endl;
		file << "# varf: " << varf0 << ", " << varf1 << endl;
		file << "# time y_sim" << endl;
		for(int i=0; i<times.rows(); i++)
			file <<  times(i) <<" "<< y_sim(i) << endl;

		std::ofstream file_one("GPtest" + std::to_string(j+1) + "_dsho.txt");
		file_one << "# DSHO simulated light curve" << endl;
		file_one << "# input parameters: " << endl;
		file_one << "# omega: " << omega_one << endl;
		file_one << "# Q: " << Qpar_one << endl;
		file_one << "# varf: " << varf_one << endl;
		file_one << "# time y_sim" << endl;
		for(int i=0; i<times.rows(); i++)
			file_one <<  times(i) <<" "<< y_sim_one(i) << endl;


	}
}
