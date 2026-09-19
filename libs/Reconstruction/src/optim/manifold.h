// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

#include <cmath>

#include "ceres/ceres.h"

inline void SetQuaternionManifold(ceres::Problem* problem, double* quat_xyzw) {
#if CERES_VERSION_MAJOR >= 3 || \
        (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
    problem->SetManifold(quat_xyzw, new ceres::EigenQuaternionManifold);
#else
    problem->SetParameterization(quat_xyzw,
                                 new ceres::EigenQuaternionParameterization);
#endif
}

// Fork note: the codebase-wide qvec convention stores quaternions in the
// [w, x, y, z] memory order, while ceres::EigenQuaternionManifold assumes
// Eigen's internal [x, y, z, w] order. This wrapper reorders the ambient
// coordinates around the standard Eigen manifold so the tangent-space
// updates act on the correct rotation axis components. Without it, every
// manifold-regularized pose block is parameterized with a permuted
// rotation, corrupting the LM steps (visible as dense-Cholesky failures in
// the rig bundle adjustment).
class EigenQuaternionManifoldWxyz : public ceres::Manifold {
public:
    int AmbientSize() const override { return 4; }
    int TangentSize() const override { return 3; }

    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override {
        double x_xyzw[4] = {x[1], x[2], x[3], x[0]};
        double out_xyzw[4];
        if (!impl_.Plus(x_xyzw, delta, out_xyzw)) {
            return false;
        }
        x_plus_delta[0] = out_xyzw[3];
        x_plus_delta[1] = out_xyzw[0];
        x_plus_delta[2] = out_xyzw[1];
        x_plus_delta[3] = out_xyzw[2];
        return true;
    }

    bool PlusJacobian(const double* x, double* jacobian_wxyz) const override {
        double x_xyzw[4] = {x[1], x[2], x[3], x[0]};
        std::vector<double> jacobian_xyzw(4 * 3);
        if (!impl_.PlusJacobian(x_xyzw, jacobian_xyzw.data())) {
            return false;
        }
        // Ambient row order: wxyz index i reads xyzw row kPerm[i].
        static constexpr int kPerm[4] = {3, 0, 1, 2};
        for (int i = 0; i < 4; ++i) {
            for (int t = 0; t < 3; ++t) {
                jacobian_wxyz[i * 3 + t] = jacobian_xyzw[kPerm[i] * 3 + t];
            }
        }
        return true;
    }

    bool Minus(const double* y,
               const double* x,
               double* y_minus_x) const override {
        double y_xyzw[4] = {y[1], y[2], y[3], y[0]};
        double x_xyzw[4] = {x[1], x[2], x[3], x[0]};
        return impl_.Minus(y_xyzw, x_xyzw, y_minus_x);
    }

    bool MinusJacobian(const double* x, double* jacobian_wxyz) const override {
        double x_xyzw[4] = {x[1], x[2], x[3], x[0]};
        std::vector<double> jacobian_xyzw(3 * 4);
        if (!impl_.MinusJacobian(x_xyzw, jacobian_xyzw.data())) {
            return false;
        }
        // Ambient column order: wxyz index j reads xyzw col kPerm[j].
        static constexpr int kPerm[4] = {3, 0, 1, 2};
        for (int t = 0; t < 3; ++t) {
            for (int j = 0; j < 4; ++j) {
                jacobian_wxyz[t * 4 + j] = jacobian_xyzw[t * 4 + kPerm[j]];
            }
        }
        return true;
    }

private:
    ceres::EigenQuaternionManifold impl_;
};

inline void SetQuaternionManifoldWxyz(ceres::Problem* problem,
                                      double* quat_wxyz) {
#if CERES_VERSION_MAJOR >= 3 || \
        (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
    problem->SetManifold(quat_wxyz, new EigenQuaternionManifoldWxyz);
#else
    // The hand-written ceres::QuaternionParameterization assumes the
    // [w, x, y, z] order natively.
    problem->SetParameterization(quat_wxyz,
                                 new ceres::QuaternionParameterization);
#endif
}

inline void SetSubsetManifold(int size,
                              const std::vector<int>& constant_params,
                              ceres::Problem* problem,
                              double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
        (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
    problem->SetManifold(params,
                         new ceres::SubsetManifold(size, constant_params));
#else
    problem->SetParameterization(
            params, new ceres::SubsetParameterization(size, constant_params));
#endif
}

template <int size>
inline void SetSphereManifold(ceres::Problem* problem, double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
        (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
    problem->SetManifold(params, new ceres::SphereManifold<size>);
#else
    problem->SetParameterization(
            params, new ceres::HomogeneousVectorParameterization(size));
#endif
}

// Use an exponential function to ensure the variable to be strictly positive
// Generally applicable for scale parameters (e.g. in colmap::Sim3d)
#if CERES_VERSION_MAJOR >= 3 || \
        (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
template <int AmbientSpaceDimension>
class PositiveExponentialManifold : public ceres::Manifold {
public:
    static_assert(ceres::DYNAMIC == Eigen::Dynamic,
                  "ceres::DYNAMIC needs to be the same as Eigen::Dynamic.");

    PositiveExponentialManifold() : size_{AmbientSpaceDimension} {}
    explicit PositiveExponentialManifold(int size) : size_{size} {
        if (AmbientSpaceDimension != Eigen::Dynamic) {
            CHECK_EQ(AmbientSpaceDimension, size)
                    << "Specified size by template parameter differs from the "
                       "supplied "
                       "one.";
        } else {
            CHECK_GT(size_, 0) << "The size of the manifold needs to be a "
                                  "positive integer.";
        }
    }

    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override {
        for (int i = 0; i < size_; ++i) {
            x_plus_delta[i] = x[i] * std::exp(delta[i]);
        }
        return true;
    }

    bool PlusJacobian(const double* x, double* jacobian) const override {
        for (int i = 0; i < size_; ++i) {
            jacobian[size_ * i + i] = x[i];
        }
        return true;
    }

    virtual bool Minus(const double* y,
                       const double* x,
                       double* y_minus_x) const override {
        for (int i = 0; i < size_; ++i) {
            y_minus_x[i] = std::log(y[i] / x[i]);
        }
        return true;
    }

    virtual bool MinusJacobian(const double* x,
                               double* jacobian) const override {
        for (int i = 0; i < size_; ++i) {
            jacobian[size_ * i + i] = 1.0 / x[i];
        }
        return true;
    }

    int AmbientSize() const override {
        return AmbientSpaceDimension == ceres::DYNAMIC ? size_
                                                       : AmbientSpaceDimension;
    }
    int TangentSize() const override { return AmbientSize(); }

private:
    const int size_{};
};
#else
class PositiveExponentialParameterization
    : public ceres::LocalParameterization {
public:
    explicit PositiveExponentialParameterization(int size) : size_{size} {
        CHECK_GT(size_, 0)
                << "The size of the manifold needs to be a positive integer.";
    }
    ~PositiveExponentialParameterization() {}

    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override {
        for (int i = 0; i < size_; ++i) {
            x_plus_delta[i] = x[i] * std::exp(delta[i]);
        }
        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override {
        for (int i = 0; i < size_; ++i) {
            jacobian[size_ * i + i] = x[i];
        }
        return true;
    }

    int GlobalSize() const override { return size_; }
    int LocalSize() const override { return size_; }

private:
    const int size_{};
};

#endif

template <int size>
inline void SetPositiveExponentialManifold(ceres::Problem* problem,
                                           double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
        (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
    problem->SetManifold(params, new PositiveExponentialManifold<size>);
#else
    problem->SetParameterization(params,
                                 new PositiveExponentialParameterization(size));
#endif
}

inline int ParameterBlockTangentSize(ceres::Problem* problem,
                                     const double* param) {
#if CERES_VERSION_MAJOR >= 3 || \
        (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
    return problem->ParameterBlockTangentSize(param);
#else
    return problem->ParameterBlockLocalSize(param);
#endif
}

}  // namespace colmap
