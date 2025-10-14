// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_POINTS_TO_NUMPY_HEADER
#define ECV_POINTS_TO_NUMPY_HEADER

#include <CVConst.h>

#include "eCV_python.h"
#include "utility/Matrix.h"

class ccHObject;
class ccPointCloud;

namespace cloudViewer {
namespace utility {

/// Class IJsonConvertible defines the behavior of a class that can convert
/// itself to/from a json::Value.
class ECV_PYTHON_LIB_API Points2Numpy {
public:
    explicit Points2Numpy();

    void setInputCloud(const ccPointCloud* cloud);

    bool getOutputData(Matrix<PointCoordinateType>& out);

    void batchConvertToNumpy(
            const ccHObject* cloud,
            std::vector<Matrix<PointCoordinateType>>& numpyContainer);

private:
    //! Associated cloud
    const ccPointCloud* m_cc_cloud;

    bool m_partialVisibility;
    unsigned m_visibilityNum;
};

}  // namespace utility
}  // namespace cloudViewer

#endif  // ECV_POINTS_TO_NUMPY_HEADER
