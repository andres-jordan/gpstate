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


int main() {

  std::random_device rd;
  std::mt19937 rng(rd());
  std::normal_distribution<double> gaussian(0.0,0.05);

  int vecsize = 10000;
	// map data to VectorXds
	VectorXd times(vecsize), yi(vecsize);

  double omega0 = TWOPI;

  for (int j=0; j<times.size(); j++){
      times(j) = 0.001*double(j);
      yi(j) = sin(omega0 * times(j)) + gaussian(rng);
  }

	// this is the observational error vector
	VectorXd yerr = VectorXd::Ones(yi.size());
	yerr.array() *= 0.05;

	VectorXd y_sim(vecsize);

	std::ofstream file("GPtest_sin.txt");
	assert(file.is_open());
	//file << "# Sin simulated light curve" << endl;
	//file << "# input parameters: " << endl;
	//file << "# omega: " << omega0 << endl;
	//file << "# time y_sim" << endl;
	for(int i=0; i<times.rows(); i++)
		file <<  times(i) <<" "<< yi(i) << endl;

  for (int j=0; j<times.size(); j++){
    yi(j) += sin((omega0 + 1) * times(j));
  }
	std::ofstream file2("GPtest_dsin.txt");
	assert(file2.is_open());
	for(int i=0; i<times.rows(); i++)
		file2 <<  times(i) <<" "<< yi(i) << endl;


}
