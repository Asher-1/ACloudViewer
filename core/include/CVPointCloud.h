//##########################################################################
//#                                                                        #
//#                               cloudViewer                              #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the  #
//#  License.                                                              #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef CV_LIB_POINT_CLOUD_HEADER
#define CV_LIB_POINT_CLOUD_HEADER

// Local
#include "GenericIndexedCloudPersist.h"
#include "PointCloudTpl.h"

namespace cloudViewer {

//! A storage-efficient point cloud structure that can also handle an unlimited
//! number of scalar fields
class CV_CORE_LIB_API PointCloud
    : public PointCloudTpl<GenericIndexedCloudPersist> {
public:
    //! Default constructor
    PointCloud() = default;

    //! Default destructor
    ~PointCloud() override = default;

    //! Reserves memory to store the normals
    bool reserveNormals(unsigned newCount) {
        if (m_normals.capacity() < newCount) {
            try {
                m_normals.reserve(newCount);
            } catch (const std::bad_alloc&) {
                return false;
            }
        }

        return true;
    }

    // inherited from PointCloudTpl
    bool resize(unsigned newCount) override {
        if (!PointCloudTpl<GenericIndexedCloudPersist>::resize(newCount)) {
            return false;
        }

        // resize the normals as well
        if (m_normals.capacity() != 0) {
            try {
                m_normals.resize(newCount);
            } catch (const std::bad_alloc&) {
                return false;
            }
        }

        return true;
    }

    //! Adds a normal
    /** \param N a 3D normal
     **/
    inline void addNormal(const CCVector3& N) {
        assert(m_normals.size() < m_normals.capacity());
        m_normals.push_back(N);
    }

    //! Returns the set of normals
    std::vector<CCVector3>& normals() { return m_normals; }

    //! Returns the set of normals (const version)
    const std::vector<CCVector3>& normals() const { return m_normals; }

    // inherited from cloudViewer::GenericIndexedCloud
    bool normalsAvailable() const override {
        return !m_normals.empty() && m_normals.size() >= size();
    }
    const CCVector3* getNormal(unsigned pointIndex) const override {
        return &m_normals[pointIndex];
    }

protected:
    //! Point normals (if any)
    std::vector<CCVector3> m_normals;
};

}  // namespace cloudViewer

#endif  // CC_LIB_POINT_CLOUD_HEADER
