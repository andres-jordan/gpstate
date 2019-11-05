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

	VectorXd y_sim;

	double varf_list[2] = {1.0, 0.1*0.1};
	for (int k=0; k<2; k++){
		for(int j=0; j < 6; j++) {
			double varf = varf_list[k];
			//auto const seed = std::random_device()();
			int seed = 123;
			std::mt19937 rng(seed);
			cout << seed << endl;
			double lambda = std::pow(2,j);

			// Simulate a Matern 3/2 model
			gpstate::matern32::Matern32Solver m32(times,yi,yerr, lambda, varf);
			m32.simulate_Matern32(y_sim, rng);

			double mean = y_sim.mean();
			Eigen::VectorXd tmp = y_sim.array()-mean;
			double variance  = tmp.dot(tmp) / y_sim.rows();
			std::cout << "var:" << variance << " " << std::to_string(k*6 + j+1) << std::endl;

			// write simulated vector to file GPtest.txt
			std::ofstream file("GPtest" + std::to_string(k*6 + j+1) + "_m32.txt");
			assert(file.is_open());
			file << "# Matern 3/2 simulated light curve" << endl;
			file << "# input parameters: " << endl;
			file << "# omega: " << lambda << endl;
			file << "# varf: " << varf  << endl;
			file << "#time y_sim" << endl;
			for(int i=0; i<times.rows(); i++)
				file <<  times(i) <<" "<< y_sim(i) << endl;

		}
	}

}
