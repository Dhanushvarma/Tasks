/*
 * Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

// includes
// Eigen
#include <Eigen/Core>

#include <tasks/config.hh>

namespace tasks
{

/** Operations on the manifold of symmetric positive definite (SPD) matrices with the affine-invariant metric (Pennec et
 * al., "A Riemannian framework for tensor computing", IJCV 2006), as used to track manipulability ellipsoids (Jaquier
 * et al., "Geometry-aware manipulability learning, tracking, and transfer", IJRR 2021) */
namespace spd
{

/** Size D (D + 1) / 2 of the Mandel vectorization of a D x D symmetric matrix */
TASKS_DLLAPI int vecSize(int D);

/** Size D of the symmetric matrix whose Mandel vectorization has n entries
 *
 * @throw std::domain_error if n is not D (D + 1) / 2 for some D
 */
TASKS_DLLAPI int matSize(int n);

/** Mandel vectorization of a symmetric matrix: the diagonal, then sqrt(2) times each superdiagonal, from the first
 * superdiagonal to the last. The Euclidean norm of the vector is the Frobenius norm of the matrix */
TASKS_DLLAPI Eigen::VectorXd vec(const Eigen::MatrixXd & S);

/** Symmetric matrix of a Mandel vectorization, inverse of vec()
 *
 * @throw std::domain_error if the size of v is not D (D + 1) / 2 for some D
 */
TASKS_DLLAPI Eigen::MatrixXd unvec(const Eigen::VectorXd & v);

/** True if M is square, symmetric (within tol relative to its largest entry) and positive definite */
TASKS_DLLAPI bool isSPD(const Eigen::MatrixXd & M, double tol = 1e-9);

/** Logarithmic map \f$ \mathrm{Log}_A(B) = A^{1/2} \log(A^{-1/2} B A^{-1/2}) A^{1/2} \f$: the tangent vector at A of
 * the geodesic from A to B
 *
 * The eigenvalues of A are bounded below by 1e-12 times its largest eigenvalue (and 1e-12), so a singular A gives a
 * finite result.
 *
 * @param distance If not null, receives the affine-invariant distance between A and B
 */
TASKS_DLLAPI Eigen::MatrixXd log(const Eigen::MatrixXd & A, const Eigen::MatrixXd & B, double * distance = nullptr);

/** Exponential map \f$ \mathrm{Exp}_A(L) = A^{1/2} \exp(A^{-1/2} L A^{-1/2}) A^{1/2} \f$, inverse of log() */
TASKS_DLLAPI Eigen::MatrixXd exp(const Eigen::MatrixXd & A, const Eigen::MatrixXd & L);

/** Affine-invariant distance \f$ \| \log(A^{-1/2} B A^{-1/2}) \|_F \f$ */
TASKS_DLLAPI double distance(const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);

/** Point at s of the geodesic from A (s = 0) to B (s = 1), \f$ A^{1/2} (A^{-1/2} B A^{-1/2})^s A^{1/2} \f$
 *
 * @param dS If not null, receives the derivative of the point with respect to s
 * @param ddS If not null, receives the second derivative of the point with respect to s
 */
TASKS_DLLAPI Eigen::MatrixXd geodesic(const Eigen::MatrixXd & A,
                                      const Eigen::MatrixXd & B,
                                      double s,
                                      Eigen::MatrixXd * dS = nullptr,
                                      Eigen::MatrixXd * ddS = nullptr);

} // namespace spd

} // namespace tasks
