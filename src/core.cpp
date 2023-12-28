#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>
#include <omp.h>

using namespace Eigen;
using namespace std;

double sgn(double val) {
    return (double(0) < val) - (val < double(0));
}

void sthreshmat(MatrixXd & x,
                double tau,
                const MatrixXd & t){

    MatrixXd tmp1(x.cols(), x.cols());
    MatrixXd tmp2(x.cols(), x.cols());

    tmp1 = x.array().unaryExpr(ptr_fun(sgn));
    tmp2 = (x.cwiseAbs() - tau*t).cwiseMax(0.0);

    x = tmp1.cwiseProduct(tmp2);

    return;
}

SparseMatrix<double> accord(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S, 
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diag,
    Ref<VectorXd> hist_norm_diff,
    Ref<VectorXd> hist_hn
    ) {

    int p = S.cols();

    int num_threads = omp_get_max_threads(); // Set the number of threads
    omp_set_num_threads(num_threads);

    SparseMatrix<double> X(p, p), Xn(p, p);     // current and next estimate
    SparseMatrix<double> Step(p, p);            // Xn - X
    MatrixXd W(p, p), Wn(p, p);                 // X*S

    MatrixXd grad_h1(p, p);                     // gradient of h1
    MatrixXd tmp(p, p);
    ArrayXd y(p);                               // diagonal elements

    double Q, hn, h1, h1n, norm_diff;
    double c_ = 0.5;
    double tau_;

    X.setIdentity();                            // initial guess: X = I

    #pragma omp parallel
    {
        W = X * S;
        grad_h1 = W + lam2 * X;                     // gradient
        h1 = 0.5 * (SparseMatrix<double>(X.transpose())*W).trace() + 0.5*lam2*pow(X.norm(),2);

        int itr_count;                              // iteration counts

        itr_count = 0;
        while (true) {

            tau_ = 1.0;

            while (true) {

                tmp = MatrixXd(X) - tau_*grad_h1;

                if (penalize_diag == true) {
                    y = tmp.diagonal().array() - tau_*LambdaMat.diagonal().array();
                } else {
                    y = tmp.diagonal().array();
                }

                y = 0.5 * (y+(y.pow(2.0)+4*tau_*Eigen::VectorXd::Ones(p).array()).sqrt());
                sthreshmat(tmp, tau_, LambdaMat);
                tmp.diagonal() = y;
                Xn = tmp.sparseView();

                // backtracking line search bounded from below
                if (tmp.diagonal().minCoeff() > 0) {

                    Step = Xn - X;
                    Wn = Xn * S;

                    h1n = 0.5*(SparseMatrix<double>(Xn.transpose())*Wn).trace() + 0.5*lam2*pow(Xn.norm(),2);
                    Q = h1 + Step.cwiseProduct(grad_h1).sum() + (0.5/tau_)*Step.squaredNorm();

                    if ((tau_ <= tau) || (h1n <= Q)) {
                        break;
                    }
                }

                tau_ *= c_;
            }
        
            grad_h1 = Wn + lam2 * Xn;
            hn = h1n - Xn.diagonal().array().log().sum() + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();

            norm_diff = Step.norm();
            hist_norm_diff(itr_count) = norm_diff;
            hist_hn(itr_count) = hn;

            itr_count += 1;

            if (norm_diff < epstol || itr_count >= maxitr) {
                if (itr_count <= maxitr) {
                    hist_norm_diff(itr_count) = -1;
                    hist_hn(itr_count) = -1;
                }
                break;
            } else {
                h1 = h1n;
                X = Xn;
                W = Wn;
            }
        }
        
        Xn = 0.5*(SparseMatrix<double>(Xn.diagonal().asDiagonal()) * Xn) + 0.5*(SparseMatrix<double>(Xn.transpose()) * Xn.diagonal().asDiagonal());
    }
    
    return Xn;
}