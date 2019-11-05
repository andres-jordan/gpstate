#ifndef _KF_H_
#define _KF_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <random>

// This file implements the Kalman recursions. Various versions 
// are optimized for estimation of (N)DSHO and 1d measurement vectors

typedef Eigen::Matrix< double, 2, 1 > Vector1d;

namespace KF{


// version for general matrices (1d y[i])
void kalman_recursions(const double& y_i, const Eigen::RowVectorXd& H_i, 
                      const Eigen::MatrixXd& R_i, const Eigen::VectorXd& x_pred_i, 
                      const Eigen::MatrixXd& P_pred_i, double& dll,
                      Eigen::VectorXd& x_filt_i, Eigen::MatrixXd& P_filt_i)
{

  Eigen::Matrix<double, 1, 1> innov, S, yy;
  Eigen::VectorXd K;
  yy(0) = y_i;
  if (!(std::isnan(y_i)))
  {
    innov = yy - H_i * x_pred_i;
    S = H_i * P_pred_i * H_i.transpose() + R_i; // innovation covariance
    double Ss = S.sum();
    double Sinv = 1.0 / Ss; 
    //tmpMat = H_i * P_pred_i.transpose();
    //K = S.transpose().colPivHouseholderQr().solve(tmpMat);
    //K.transposeInPlace();
    K = P_pred_i * H_i.transpose() * Sinv;
    x_filt_i = x_pred_i + K*innov;
    //P_filt_i = P_pred_i - K * S * K.transpose();
    P_filt_i = P_pred_i - Ss * K * K.transpose();
    //tmpVec = S.colPivHouseholderQr().solve(innov);
    //dll = -0.5*innov.dot(tmpVec) - 0.5*std::log(2 * M_PI * S.determinant());
    dll = -0.5 * Sinv * (innov.dot(innov)) - 0.5 * std::log(2.0 * M_PI * Ss);
  }
  else
  {
    x_filt_i = x_pred_i;
    P_filt_i = P_pred_i;
    dll=0;
  }
}

// general version, optimized for NDSHO
void kalman_recursions_NDSHO(const int N, const double& y_i,  
                      const Eigen::MatrixXd& R_i, const Eigen::VectorXd& x_pred_i, 
                      const Eigen::MatrixXd& P_pred_i, double& dll,
                      Eigen::VectorXd& x_filt_i, Eigen::MatrixXd& P_filt_i)
{

  double innov, S;
  Eigen::VectorXd K(N);
  //P_filt_i = Eigen::MatrixXd::Zero(N,N);
  int i;
  if (!(std::isnan(y_i)))
  {
    innov = y_i;
    for (i=0; i<N; i+=2)
      innov -= x_pred_i(i);
    S = R_i(0);
    for (i=0; i<N; i+=2)
      S += P_pred_i(i,i);
    //S = H_i * P_pred_i * H_i.transpose() + R_i; // innovation covariance
    double Sinv = 1.0 / S; 
    //tmpMat = H_i * P_pred_i.transpose();
    //K = S.transpose().colPivHouseholderQr().solve(tmpMat);
    //K.transposeInPlace();
    for (i=0; i<N; i+=2)
    {
      K(i) = P_pred_i(i,i)*Sinv;
      K(i+1) = P_pred_i(i,i+1)*Sinv;
    }
    //K = P_pred_i * H_i.transpose() * Sinv;
    x_filt_i = x_pred_i + K*innov;
    //P_filt_i = P_pred_i - K * S * K.transpose();
    for (i=0; i<N; i+=2)
    {
      double tmp = S*K(i)*K(i+1);
      P_filt_i(i,i) = P_pred_i(i,i) - S*K(i)*K(i);
      P_filt_i(i,i+1) = P_pred_i(i,i+1) - tmp;
      P_filt_i(i+1,i) = P_pred_i(i+1,i) - tmp;
      P_filt_i(i+1,i+1) = P_pred_i(i+1,i+1) - S*K(i+1)*K(i+1);  
    }

    //P_filt_i = P_pred_i - S * K * K.transpose();
    //tmpVec = S.colPivHouseholderQr().solve(innov);
    //dll = -0.5*innov.dot(tmpVec) - 0.5*std::log(2 * M_PI * S.determinant());
    dll = -0.5 * (Sinv * innov * innov +std::log(2.0 * M_PI * S));
  }
  else
  {
    x_filt_i = x_pred_i;
    P_filt_i = P_pred_i;
    dll=0;
  }
}


// version for 2d matrices, optimized for DSHO
void kalman_recursions_DSHO(const double& y_i,  
                      const Eigen::Matrix<double, 1,1>& R_i, const Eigen::Vector2d& x_pred_i, 
                      const Eigen::Matrix2d& P_pred_i, double& dll,
                      Eigen::Vector2d& x_filt_i, Eigen::Matrix2d& P_filt_i)
{
  //Eigen::Matrix<double, 1, 1> innov, S, yy;
  double innov, S;
  Eigen::Vector2d K;
  //yy(0) = y_i;
  if (!(std::isnan(y_i)))
  {
    innov = y_i - x_pred_i(0);
    S = P_pred_i(0,0) + R_i(0);// innovation covariance
    double Sinv = 1.0 / S; 
    K(0) = P_pred_i(0,0)*Sinv;
    K(1) = P_pred_i(1,0)*Sinv;
    //K = P_pred_i * H_i.transpose() * Sinv;
    x_filt_i = x_pred_i + K*innov;
    double tmp = S*K(0)*K(1);
    P_filt_i(0,0) = P_pred_i(0,0) - S*K(0)*K(0);
    P_filt_i(0,1) = P_pred_i(0,1) - tmp;
    P_filt_i(1,0) = P_pred_i(1,0) - tmp;
    P_filt_i(1,1) = P_pred_i(1,1) - S*K(1)*K(1);  
    //P_filt_i = P_pred_i - S * K * K.transpose();
    //std::cout << innov << " " << Sinv << " " << innov.dot(innov) << " " << Ss << " " << 0.5*std::log(2.0 * M_PI * Ss) << std::endl; 
    dll = -0.5 * (Sinv * innov * innov + std::log(2.0 * M_PI * S));
    //std::cout << dll << std::endl;
  }
  else
  {
    x_filt_i = x_pred_i;
    P_filt_i = P_pred_i;
    dll=0;
  }
}



// version for general 2d matrices
void kalman_recursions(const double& y_i, const Eigen::RowVector2d& H_i, 
                      const Eigen::Matrix<double, 1,1>& R_i, const Eigen::Vector2d& x_pred_i, 
                      const Eigen::Matrix2d& P_pred_i, double& dll,
                      Eigen::Vector2d& x_filt_i, Eigen::Matrix2d& P_filt_i)
{
  Eigen::Matrix<double, 1, 1> innov, S, yy;
  Eigen::Vector2d K;
  K.setZero();
  //K(0) = 0.0;
  //K(1) = 0.0;
  yy(0) = y_i;
  if (!(std::isnan(y_i)))
  {
    innov = yy - H_i * x_pred_i;
    S = H_i * P_pred_i * H_i.transpose() + R_i ;// innovation covariance
    double Ss = S.sum();
    double Sinv = 1.0 / Ss; 
    K = P_pred_i * H_i.transpose() * Sinv;
    x_filt_i = x_pred_i + K*innov;
    P_filt_i = P_pred_i - Ss * K * K.transpose();
    //std::cout << innov << " " << Sinv << " " << innov.dot(innov) << " " << Ss << " " << 0.5*std::log(2.0 * M_PI * Ss) << std::endl; 
    dll = -0.5 * Sinv * (innov.dot(innov)) - 0.5 * std::log(2.0 * M_PI * Ss);
    //std::cout << dll << std::endl;
  }
  else
  {
    x_filt_i = x_pred_i;
    P_filt_i = P_pred_i;
    dll=0;
  }
}



double deltaF(int i, const Eigen::VectorXd& times)
{
  assert(i>0);
  return times(i)-times(i-1);
};


} // namespace KF

#endif // _KF_H_
