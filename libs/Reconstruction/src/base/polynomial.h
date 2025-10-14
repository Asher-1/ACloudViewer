// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_BASE_POLYNOMIAL_H_
#define COLMAP_SRC_BASE_POLYNOMIAL_H_

#include <Eigen/Core>

namespace colmap {

// All polynomials are assumed to be the form:
//
//   sum_{i=0}^N polynomial(i) x^{N-i}.
//
// and are given by a vector of coefficients of size N + 1.
//
// The implementation is based on COLMAP's old polynomial functionality and is
// inspired by Ceres-Solver's/Theia's implementation to support complex
// polynomials. The companion matrix implementation is based on NumPy.

// Evaluate the polynomial for the given coefficients at x using the Horner
// scheme. This function is templated such that the polynomial may be evaluated
// at real and/or imaginary points.
template <typename T>
T EvaluatePolynomial(const Eigen::VectorXd& coeffs, const T& x);

// Find the root of polynomials of the form: a * x + b = 0.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindLinearPolynomialRoots(const Eigen::VectorXd& coeffs,
                               Eigen::VectorXd* real,
                               Eigen::VectorXd* imag);

// Find the roots of polynomials of the form: a * x^2 + b * x + c = 0.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindQuadraticPolynomialRoots(const Eigen::VectorXd& coeffs,
                                  Eigen::VectorXd* real,
                                  Eigen::VectorXd* imag);

// Find the roots of a polynomial using the Durand-Kerner method, based on:
//
//    https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method
//
// The Durand-Kerner is comparatively fast but often unstable/inaccurate.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindPolynomialRootsDurandKerner(const Eigen::VectorXd& coeffs,
                                     Eigen::VectorXd* real,
                                     Eigen::VectorXd* imag);

// Find the roots of a polynomial using the companion matrix method, based on:
//
//    R. A. Horn & C. R. Johnson, Matrix Analysis. Cambridge,
//    UK: Cambridge University Press, 1999, pp. 146-7.
//
// Compared to Durand-Kerner, this method is slower but more stable/accurate.
// The real and/or imaginary variable may be NULL if the output is not needed.
bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs,
                                        Eigen::VectorXd* real,
                                        Eigen::VectorXd* imag);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T EvaluatePolynomial(const Eigen::VectorXd& coeffs, const T& x) {
    T value = 0.0;
    for (Eigen::VectorXd::Index i = 0; i < coeffs.size(); ++i) {
        value = value * x + coeffs(i);
    }
    return value;
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_POLYNOMIAL_H_
