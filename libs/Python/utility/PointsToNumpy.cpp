// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "utility/PointsToNumpy.h"

// CV_CORE_LIB
#include <Eigen.h>

// CV_DB_LIB
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>

// SYSTEM
#include <assert.h>

namespace cloudViewer {
namespace utility {

Points2Numpy::Points2Numpy()
    : m_cc_cloud(nullptr), m_partialVisibility(false), m_visibilityNum(0) {}

void Points2Numpy::setInputCloud(const ccPointCloud* cloud) {
    assert(cloud);
    m_cc_cloud = cloud;
    // count the number of visible points
    if (m_cc_cloud->isVisibilityTableInstantiated()) {
        m_visibilityNum = 0;
        assert(m_cc_cloud->getTheVisibilityArray().size() ==
               m_cc_cloud->size());
        m_partialVisibility = true;
        unsigned count = m_cc_cloud->size();
        for (unsigned i = 0; i < count; ++i) {
            if (m_cc_cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE) {
                ++m_visibilityNum;
            }
        }
    } else {
        m_visibilityNum = 0;
        m_partialVisibility = false;
    }
}

bool Points2Numpy::getOutputData(Matrix<PointCoordinateType>& out) {
    if (!m_cc_cloud) {
        return false;
    }

    assert(m_cc_cloud);
    out.clear();

    unsigned pointCount =
            m_partialVisibility ? m_visibilityNum : m_cc_cloud->size();
    size_t bits = 3;
    if (m_cc_cloud->hasColors()) {
        bits += 3;
    }
    if (m_cc_cloud->hasNormals()) {
        bits += 3;
    }

    std::vector<size_t> shape;
    shape.push_back(pointCount);
    shape.push_back(bits);
    out.resize(shape);

    try {
        unsigned index = 0;
        for (unsigned i = 0; i < pointCount; ++i) {
            if (m_partialVisibility) {
                if (m_cc_cloud->getTheVisibilityArray().at(i) ==
                    POINT_VISIBLE) {
                    const CCVector3* P = m_cc_cloud->getPoint(i);
                    out[index * bits + 0] =
                            static_cast<PointCoordinateType>(P->x);
                    out[index * bits + 1] =
                            static_cast<PointCoordinateType>(P->y);
                    out[index * bits + 2] =
                            static_cast<PointCoordinateType>(P->z);

                    size_t offset = 3;

                    if (m_cc_cloud->hasColors()) {
                        const ecvColor::Rgb& rgb = m_cc_cloud->getPointColor(i);
                        out[index * bits + 3] =
                                static_cast<PointCoordinateType>(rgb.r);
                        out[index * bits + 4] =
                                static_cast<PointCoordinateType>(rgb.g);
                        out[index * bits + 5] =
                                static_cast<PointCoordinateType>(rgb.b);
                        offset += 3;
                    }

                    if (m_cc_cloud->hasNormals()) {
                        const CCVector3& normal = m_cc_cloud->getPointNormal(i);
                        out[index * bits + offset] = normal.x;
                        out[index * bits + offset + 1] = normal.y;
                        out[index * bits + offset + 2] = normal.z;
                    }

                    index++;
                }
            } else {
                const CCVector3* P = m_cc_cloud->getPoint(i);
                out[i * bits + 0] = static_cast<PointCoordinateType>(P->x);
                out[i * bits + 1] = static_cast<PointCoordinateType>(P->y);
                out[i * bits + 2] = static_cast<PointCoordinateType>(P->z);

                size_t offset = 3;
                if (m_cc_cloud->hasColors()) {
                    const ecvColor::Rgb& rgb = m_cc_cloud->getPointColor(i);
                    out[i * bits + 3] = static_cast<PointCoordinateType>(rgb.r);
                    out[i * bits + 4] = static_cast<PointCoordinateType>(rgb.g);
                    out[i * bits + 5] = static_cast<PointCoordinateType>(rgb.b);
                    offset += 3;
                }

                if (m_cc_cloud->hasNormals()) {
                    const CCVector3& normal = m_cc_cloud->getPointNormal(i);
                    out[i * bits + offset] = normal.x;
                    out[i * bits + offset + 1] = normal.y;
                    out[i * bits + offset + 2] = normal.z;
                }
            }
        }
    } catch (...) {
        // any error (memory, etc.)
        CVLog::Warning(
                "[Points2Numpy::getOutputData] some unknown errors occured!");
        return false;
    }

    return true;
}

void Points2Numpy::batchConvertToNumpy(
        const ccHObject* entity,
        std::vector<Matrix<PointCoordinateType>>& numpyContainer) {
    unsigned childNumber = static_cast<unsigned>(entity->getChildrenNumber());
    assert(entity && childNumber != 0);
    for (unsigned i = 0; i < childNumber; ++i) {
        ccHObject* ent = entity->getChild(i);
        if (!ent || !ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
            CVLog::Warning(
                    "[DeepSemanticSegmentation::compute] failed: all entities "
                    "should be pointcloud!");
            numpyContainer.clear();
            return;
        }
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent);

        if (!cloud) {
            numpyContainer.clear();
            return;
        }

        setInputCloud(cloud);
        Matrix<PointCoordinateType> matrixPy;
        if (!getOutputData(matrixPy)) {
            numpyContainer.clear();
            return;
        }

        numpyContainer.push_back(matrixPy);
    }
}

}  // namespace utility
}  // namespace cloudViewer
