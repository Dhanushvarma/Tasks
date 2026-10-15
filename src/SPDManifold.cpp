/*
 * Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

// associated header
#include "Tasks/SPDManifold.h"

// includes
// std
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

// Eigen
#include <Eigen/Eigenvalues>

namespace tasks
{

namespace spd
{

namespace
{

/** Square root and inverse square root of an SPD matrix, its eigenvalues bounded below */
struct SqrtSPD
{
  explicit SqrtSPD(const Eigen::MatrixXd & A)
  {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    Eigen::VectorXd l = es.eigenvalues();
    const double lMin = 1e-12 * std::max(l.maxCoeff(), 1.);
    l = l.cwiseMax(lMin).cwiseSqrt();
    const Eigen::MatrixXd & V = es.eigenvectors();
    sqrt = V * l.asDiagonal() * V.transpose();
    invSqrt = V * l.cwiseInverse().asDiagonal() * V.transpose();
  }

  /** invSqrt B invSqrt, symmetrized */
  Eigen::MatrixXd whiten(const Eigen::MatrixXd & B) const
  {
    Eigen::MatrixXd C = invSqrt * B * invSqrt;
    return 0.5 * (C + C.transpose());
  }

  /** sqrt (U diag(f) U^T) sqrt, symmetrized */
  Eigen::MatrixXd color(const Eigen::MatrixXd & U, const Eigen::VectorXd & f) const
  {
    Eigen::MatrixXd S = sqrt * U * f.asDiagonal() * U.transpose() * sqrt;
    return 0.5 * (S + S.transpose());
  }

  Eigen::MatrixXd sqrt;
  Eigen::MatrixXd invSqrt;
};

/** Eigenvalues of an SPD matrix, bounded below so that their logarithm is finite */
Eigen::VectorXd positiveEigenvalues(const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> & es)
{ return es.eigenvalues().cwiseMax(std::numeric_limits<double>::min()); }

} // namespace

int vecSize(int D)
{ return D * (D + 1) / 2; }

int matSize(int n)
{
  int D = static_cast<int>(std::lround((std::sqrt(8. * n + 1.) - 1.) / 2.));
  if(n < 1 || vecSize(D) != n)
  {
    std::ostringstream str;
    str << "spd: " << n << " is not the size of the vectorization of a symmetric matrix";
    throw std::domain_error(str.str());
  }
  return D;
}

Eigen::VectorXd vec(const Eigen::MatrixXd & S)
{
  const Eigen::Index D = S.rows();
  Eigen::VectorXd v(vecSize(static_cast<int>(D)));
  Eigen::Index i = 0;
  for(Eigen::Index k = 0; k < D; ++k) { v(i++) = S(k, k); }
  for(Eigen::Index offset = 1; offset < D; ++offset)
  {
    for(Eigen::Index k = 0; k + offset < D; ++k) { v(i++) = std::sqrt(0.5) * (S(k, k + offset) + S(k + offset, k)); }
  }
  return v;
}

Eigen::MatrixXd unvec(const Eigen::VectorXd & v)
{
  const Eigen::Index D = matSize(static_cast<int>(v.size()));
  Eigen::MatrixXd S(D, D);
  Eigen::Index i = 0;
  for(Eigen::Index k = 0; k < D; ++k) { S(k, k) = v(i++); }
  for(Eigen::Index offset = 1; offset < D; ++offset)
  {
    for(Eigen::Index k = 0; k + offset < D; ++k) { S(k, k + offset) = S(k + offset, k) = std::sqrt(0.5) * v(i++); }
  }
  return S;
}

bool isSPD(const Eigen::MatrixXd & M, double tol)
{
  if(M.rows() != M.cols() || M.size() == 0 || !M.allFinite()) { return false; }
  if((M - M.transpose()).cwiseAbs().maxCoeff() > tol * std::max(M.cwiseAbs().maxCoeff(), 1.)) { return false; }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
  return es.eigenvalues().minCoeff() > 0.;
}

Eigen::MatrixXd log(const Eigen::MatrixXd & A, const Eigen::MatrixXd & B, double * distance)
{
  SqrtSPD sqrtA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sqrtA.whiten(B));
  Eigen::VectorXd logC = positiveEigenvalues(es).array().log();
  if(distance) { *distance = logC.norm(); }
  return sqrtA.color(es.eigenvectors(), logC);
}

Eigen::MatrixXd exp(const Eigen::MatrixXd & A, const Eigen::MatrixXd & L)
{
  SqrtSPD sqrtA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sqrtA.whiten(L));
  return sqrtA.color(es.eigenvectors(), es.eigenvalues().array().exp());
}

double distance(const Eigen::MatrixXd & A, const Eigen::MatrixXd & B)
{
  SqrtSPD sqrtA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sqrtA.whiten(B), Eigen::EigenvaluesOnly);
  return positiveEigenvalues(es).array().log().matrix().norm();
}

Eigen::MatrixXd geodesic(const Eigen::MatrixXd & A,
                         const Eigen::MatrixXd & B,
                         double s,
                         Eigen::MatrixXd * dS,
                         Eigen::MatrixXd * ddS)
{
  SqrtSPD sqrtA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sqrtA.whiten(B));
  const Eigen::ArrayXd logC = positiveEigenvalues(es).array().log();
  // C^s = U diag(c_i^s) U^T, its derivatives with respect to s multiply each c_i^s by log(c_i)
  const Eigen::ArrayXd powC = (s * logC).exp();
  const Eigen::MatrixXd & U = es.eigenvectors();
  if(dS) { *dS = sqrtA.color(U, (powC * logC).matrix()); }
  if(ddS) { *ddS = sqrtA.color(U, (powC * logC * logC).matrix()); }
  return sqrtA.color(U, powC.matrix());
}

} // namespace spd

} // namespace tasks
