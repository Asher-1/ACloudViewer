// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CLOUDVIEWER_EIGEN_H
#define CLOUDVIEWER_EIGEN_H

#include "CVCoreLib.h"

#ifdef _MSVC_LANG
#define CPP_VERSION _MSVC_LANG
#else
#define CPP_VERSION __cplusplus
#endif

// EIGEN
// Ensure EIGEN_HAS_CXX17_OVERALIGN is set before including Eigen headers
// This allows C++17 std::allocator's automatic alignment handling even when
// EIGEN_MAX_ALIGN_BYTES=0 (for PCL compatibility)
#ifndef EIGEN_HAS_CXX17_OVERALIGN
#ifdef CPP_VERSION
#if CPP_VERSION >= 201703L
#define EIGEN_HAS_CXX17_OVERALIGN 1
#endif
#endif
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

// SYSTEM
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if !EIGEN_VERSION_AT_LEAST(3, 4, 0) || CPP_VERSION < 201703L

#include <initializer_list>
#ifndef EIGEN_ALIGNED_ALLOCATOR
#define EIGEN_ALIGNED_ALLOCATOR Eigen::aligned_allocator
#endif

// Equivalent to EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION but with support for
// initializer lists, which is a C++11 feature and not supported by the Eigen.
// The initializer list extension is inspired by Theia and StackOverflow code.
#define EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(...)                   \
    namespace std {                                                          \
    template <>                                                              \
    class vector<__VA_ARGS__, std::allocator<__VA_ARGS__>>                   \
        : public vector<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__>> { \
        typedef vector<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__>>    \
                vector_base;                                                 \
                                                                             \
    public:                                                                  \
        typedef __VA_ARGS__ value_type;                                      \
        typedef vector_base::allocator_type allocator_type;                  \
        typedef vector_base::size_type size_type;                            \
        typedef vector_base::iterator iterator;                              \
        explicit vector(const allocator_type& a = allocator_type())          \
            : vector_base(a) {}                                              \
        template <typename InputIterator>                                    \
        vector(InputIterator first,                                          \
               InputIterator last,                                           \
               const allocator_type& a = allocator_type())                   \
            : vector_base(first, last, a) {}                                 \
        vector(const vector& c) : vector_base(c) {}                          \
        explicit vector(size_type num, const value_type& val = value_type()) \
            : vector_base(num, val) {}                                       \
        vector(iterator start, iterator end) : vector_base(start, end) {}    \
        vector& operator=(const vector& x) {                                 \
            vector_base::operator=(x);                                       \
            return *this;                                                    \
        }                                                                    \
        vector(initializer_list<__VA_ARGS__> list)                           \
            : vector_base(list.begin(), list.end()) {}                       \
    };                                                                       \
    }  // namespace std

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Vector2d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Vector4d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Vector4f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Matrix2d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Matrix2f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Matrix4d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Matrix4f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Affine3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Affine3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Quaterniond)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Quaternionf)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Matrix<float, 3, 4>)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Matrix<double, 3, 4>)

#endif

#undef CPP_VERSION

namespace Eigen {

/// Extending Eigen namespace by adding frequently used matrix type
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<uint8_t, 3, 1> Vector3uint8;

/// Use Eigen::DontAlign for matrices inside classes which are exposed in the
/// CloudViewer headers https://github.com/isl-org/CloudViewer/issues/653
typedef Eigen::Matrix<double, 6, 6, Eigen::DontAlign> Matrix6d_u;
typedef Eigen::Matrix<double, 4, 4, Eigen::DontAlign> Matrix4d_u;
typedef Eigen::Matrix<double, 3, 1, Eigen::DontAlign> Vector3d_u;
typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vector3f_u;
typedef Eigen::Matrix<double, 4, 1, Eigen::DontAlign> Vector4d_u;
typedef Eigen::Matrix<float, 4, 1, Eigen::DontAlign> Vector4f_u;

}  // namespace Eigen

namespace cloudViewer {

namespace utility {

using Matrix4d_allocator = Eigen::aligned_allocator<Eigen::Matrix4d>;
using Matrix6d_allocator = Eigen::aligned_allocator<Eigen::Matrix6d>;
using Vector2d_allocator = Eigen::aligned_allocator<Eigen::Vector2d>;
using Vector3uint8_allocator = Eigen::aligned_allocator<Eigen::Vector3uint8>;
using Vector4i_allocator = Eigen::aligned_allocator<Eigen::Vector4i>;
using Vector4d_allocator = Eigen::aligned_allocator<Eigen::Vector4d>;
using Vector6d_allocator = Eigen::aligned_allocator<Eigen::Vector6d>;

/// Genretate a skew-symmetric matrix from a vector 3x1.
Eigen::Matrix3d CV_CORE_LIB_API SkewMatrix(const Eigen::Vector3d& vec);

/// Function to transform 6D motion vector to 4D motion matrix
/// Reference:
/// https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html#TutorialGeoTransform
Eigen::Matrix4d CV_CORE_LIB_API
TransformVector6dToMatrix4d(const Eigen::Vector6d& input);

/// Function to transform 4D motion matrix to 6D motion vector
/// this is consistent with the matlab function in
/// the Aerospace Toolbox
/// Reference:
/// https://github.com/qianyizh/ElasticReconstruction/blob/master/Matlab_Toolbox/Core/mrEvaluateRegistration.m
Eigen::Vector6d CV_CORE_LIB_API
TransformMatrix4dToVector6d(const Eigen::Matrix4d& input);

/// Function to solve Ax=b
std::tuple<bool, Eigen::VectorXd> CV_CORE_LIB_API
SolveLinearSystemPSD(const Eigen::MatrixXd& A,
                     const Eigen::VectorXd& b,
                     bool prefer_sparse = false,
                     bool check_symmetric = false,
                     bool check_det = false,
                     bool check_psd = false);

/// Function to solve Jacobian system
/// Input: 6x6 Jacobian matrix and 6-dim residual vector.
/// Output: tuple of is_success, 4x4 extrinsic matrices.
std::tuple<bool, Eigen::Matrix4d> CV_CORE_LIB_API
SolveJacobianSystemAndObtainExtrinsicMatrix(const Eigen::Matrix6d& JTJ,
                                            const Eigen::Vector6d& JTr);

/// Function to solve Jacobian system
/// Input: 6nx6n Jacobian matrix and 6n-dim residual vector.
/// Output: tuple of is_success, n 4x4 motion matrices.
std::tuple<bool, std::vector<Eigen::Matrix4d, Matrix4d_allocator>>
        CV_CORE_LIB_API SolveJacobianSystemAndObtainExtrinsicMatrixArray(
                const Eigen::MatrixXd& JTJ, const Eigen::VectorXd& JTr);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> CV_CORE_LIB_API
ComputeJTJandJTr(std::function<void(int, VecType&, double&, double&)> f,
                 int iteration_num,
                 bool verbose = true);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> CV_CORE_LIB_API ComputeJTJandJTr(
        std::function<
                void(int,
                     std::vector<VecType, Eigen::aligned_allocator<VecType>>&,
                     std::vector<double>&,
                     std::vector<double>&)> f,
        int iteration_num,
        bool verbose = true);

Eigen::Matrix3d CV_CORE_LIB_API RotationMatrixX(double radians);
Eigen::Matrix3d CV_CORE_LIB_API RotationMatrixY(double radians);
Eigen::Matrix3d CV_CORE_LIB_API RotationMatrixZ(double radians);

/// Color conversion from double [0,1] to uint8_t 0-255; this does proper
/// clipping and rounding
Eigen::Vector3uint8 CV_CORE_LIB_API ColorToUint8(const Eigen::Vector3d& color);
/// Color conversion from uint8_t 0-255 to double [0,1]
Eigen::Vector3d CV_CORE_LIB_API ColorToDouble(uint8_t r, uint8_t g, uint8_t b);
Eigen::Vector3d CV_CORE_LIB_API ColorToDouble(const Eigen::Vector3uint8& rgb);

/// Function to compute the covariance matrix of a set of points.
template <typename IdxType>
Eigen::Matrix3d CV_CORE_LIB_API
ComputeCovariance(const std::vector<Eigen::Vector3d>& points,
                  const std::vector<IdxType>& indices);

/// Function to compute the mean and covariance matrix of a set of points.
template <typename IdxType>
std::tuple<Eigen::Vector3d, Eigen::Matrix3d> CV_CORE_LIB_API
ComputeMeanAndCovariance(const std::vector<Eigen::Vector3d>& points,
                         const std::vector<IdxType>& indices);

/// Function to compute the mean and covariance matrix of a set of points.
/// \tparam RealType Either float or double.
/// \tparam IdxType Either size_t or int.
/// \param points Contiguous memory with the 3D points.
/// \param indices The indices for which the mean and covariance will be
/// computed. \return The mean and covariance matrix.
template <typename RealType, typename IdxType>
std::tuple<Eigen::Vector3d, Eigen::Matrix3d> CV_CORE_LIB_API
ComputeMeanAndCovariance(const RealType* const points,
                         const std::vector<IdxType>& indices);

}  // namespace utility
}  // namespace cloudViewer

#endif  // CLOUDVIEWER_EIGEN_H
