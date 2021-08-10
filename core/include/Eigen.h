// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#ifndef CV_EIGEN_HEADER
#define CV_EIGEN_HEADER

#include "CVCoreLib.h"

// EIGEN
#include <Eigen/Core>  // for EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <Eigen/Geometry>
#include <Eigen/StdVector>

// SYSTEM
#include <initializer_list>
#include <memory>  // for std::allocate_shared, std::dynamic_pointer_cast, cloudViewer::make_shared, std::shared_ptr, std::static_pointer_cast, std::weak_ptr
#include <tuple>
#include <type_traits>  // for std::enable_if_t, std::false_type, std::true_type
#include <utility>      // for std::forward
#include <vector>

/**
 * \brief Macro to signal a class requires a custom allocator
 *
 *  It's an implementation detail to have pcl::has_custom_allocator work, a
 *  thin wrapper over Eigen's own macro
 *
 * \see pcl::has_custom_allocator, pcl::make_shared
 * \ingroup common
 */

#ifdef SIMD_ENABLED
#define CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW \
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW           \
    using _custom_allocator_type_trait = void;
#else
#define CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW using _custom_type_trait = void;
#endif

#ifdef SIMD_ENABLED
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

#define EIGEN_STL_UMAP(KEY, VALUE)                                     \
    std::unordered_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>, \
                       Eigen::aligned_allocator<std::pair<KEY const, VALUE>>>
#define EIGEN_STL_UMAP_HASH(KEY, VALUE, HASH)                \
    std::unordered_map<KEY, VALUE, HASH, std::equal_to<KEY>, \
                       Eigen::aligned_allocator<std::pair<KEY const, VALUE>>>
#else
// Equivalent to EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION but with support for
// initializer lists, which is a C++11 feature and not supported by the Eigen.
// The initializer list extension is inspired by Theia and StackOverflow code.
#define EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(...) \
    namespace std {}  // namespace std

#define EIGEN_STL_UMAP(KEY, VALUE)                                     \
    std::unordered_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>, \
                       std::allocator<std::pair<KEY const, VALUE>>>
#define EIGEN_STL_UMAP_HASH(KEY, VALUE, HASH)                \
    std::unordered_map<KEY, VALUE, HASH, std::equal_to<KEY>, \
                       std::allocator<std::pair<KEY const, VALUE>>>
#endif

// EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Eigen::Vector2d)
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

namespace Eigen {

template <typename X, typename... Args>
inline std::shared_ptr<X> make_shared(Args&&... args) {
    return std::shared_ptr<X>(new X(std::forward<Args>(args)...));
}

template <typename X, typename... Args>
inline std::unique_ptr<X> make_unique(Args&&... args) {
    return std::unique_ptr<X>(new X(std::forward<Args>(args)...));
}

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
template <typename...>
using void_t = void;  // part of std in c++17
template <typename, typename = void_t<>>
struct has_custom_allocator : std::false_type {};
template <typename T>
struct has_custom_allocator<T, void_t<typename T::_custom_allocator_type_trait>>
    : std::true_type {};

template <typename T, typename... Args>
std::enable_if_t<has_custom_allocator<T>::value, std::shared_ptr<T>>
make_shared(Args&&... args) {
    return std::allocate_shared<T>(Eigen::aligned_allocator<T>(),
                                   std::forward<Args>(args)...);
}

template <typename T, typename... Args>
std::enable_if_t<!has_custom_allocator<T>::value, std::shared_ptr<T>>
make_shared(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

namespace utility {

using Matrix4d_allocator = Eigen::aligned_allocator<Eigen::Matrix4d>;
using Matrix6d_allocator = Eigen::aligned_allocator<Eigen::Matrix6d>;
using Vector2d_allocator = Eigen::aligned_allocator<Eigen::Vector2d>;
using Vector3uint8_allocator = Eigen::aligned_allocator<Eigen::Vector3uint8>;
using Vector4i_allocator = Eigen::aligned_allocator<Eigen::Vector4i>;
using Vector4d_allocator = Eigen::aligned_allocator<Eigen::Vector4d>;
using Vector6d_allocator = Eigen::aligned_allocator<Eigen::Vector6d>;

/// Genretate a skew-symmetric matrix from a vector 3x1.
Eigen::Matrix3d CV_CORE_LIB_API SkewMatrix(const Eigen::Vector3d &vec);

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
std::tuple<MatType, VecType, double> ComputeJTJandJTr(
        std::function<void(int, VecType&, double&, double&)> f,
        int iteration_num,
        bool verbose = true);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> ComputeJTJandJTr(
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

}  // namespace utility
}  // namespace cloudViewer

#endif  // CV_EIGEN_HEADER
