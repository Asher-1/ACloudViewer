// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvHObject.h"
#include "ecvPlanarEntityInterface.h"

namespace cloudViewer {
class GenericIndexedCloudPersist;
}

class ccMesh;
class ccPolyline;
class ccPointCloud;

//! Facet
/** Composite object: point cloud + 2D1/2 contour polyline + 2D1/2 surface mesh
 **/
class ECV_DB_LIB_API ccFacet : public ccHObject,
                               public ccPlanarEntityInterface {
public:
    //! Default constructor
    /** \param maxEdgeLength max edge length (if possible - ignored if 0)
            \param name name
    **/
    ccFacet(PointCoordinateType maxEdgeLength = 0,
            QString name = QString("Facet"));

    //! Destructor
    virtual ~ccFacet() override;

    //! Creates a facet from a set of points
    /** The facet boundary can either be the convex hull (maxEdgeLength = 0)
            or the concave hull (maxEdgeLength > 0).
            \param cloud cloud from which to create the facet
            \param maxEdgeLength max edge length (if possible - ignored if 0)
            \param transferOwnership if true and the input cloud is a
    ccPointCloud, it will be 'kept' as 'origin points' \param planeEquation to
    input a custom plane equation \return a facet (or 0 if an error occurred)
    **/
    static ccFacet* Create(cloudViewer::GenericIndexedCloudPersist* cloud,
                           PointCoordinateType maxEdgeLength = 0,
                           bool transferOwnership = false,
                           const PointCoordinateType* planeEquation = nullptr);

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::FACET;
    }
    virtual bool isSerializable() const override { return true; }

    //! Sets the facet unique color
    /** \param rgb RGB color
     **/
    void setColor(const ecvColor::Rgb& rgb);

    // inherited from ccPlanarEntityInterface
    inline CCVector3 getNormal() const override {
        return CCVector3(m_planeEquation);
    }

    //! Returns associated RMS
    inline double getRMS() const { return m_rms; }
    //! Returns associated surface area
    inline double getSurface() const { return m_surface; }
    //! Returns plane equation
    inline const PointCoordinateType* getPlaneEquation() const {
        return m_planeEquation;
    }
    //! Inverts the facet normal
    void invertNormal();
    //! Returns the facet center
    inline const CCVector3& getCenter() const { return m_center; }

    //! Returns polygon mesh (if any)
    inline ccMesh* getPolygon() { return m_polygonMesh; }
    //! Returns polygon mesh (if any)
    inline const ccMesh* getPolygon() const { return m_polygonMesh; }

    //! Returns contour polyline (if any)
    inline ccPolyline* getContour() { return m_contourPolyline; }
    //! Returns contour polyline (if any)
    inline const ccPolyline* getContour() const { return m_contourPolyline; }

    //! Returns contour vertices (if any)
    inline ccPointCloud* getContourVertices() { return m_contourVertices; }
    //! Returns contour vertices (if any)
    inline const ccPointCloud* getContourVertices() const {
        return m_contourVertices;
    }

    //! Returns origin points (if any)
    inline ccPointCloud* getOriginPoints() { return m_originPoints; }
    //! Returns origin points (if any)
    inline const ccPointCloud* getOriginPoints() const {
        return m_originPoints;
    }

    //! Sets polygon mesh
    inline void setPolygon(ccMesh* mesh) { m_polygonMesh = mesh; }
    //! Sets contour polyline
    inline void setContour(ccPolyline* poly) { m_contourPolyline = poly; }
    //! Sets contour vertices
    inline void setContourVertices(ccPointCloud* cloud) {
        m_contourVertices = cloud;
    }
    //! Sets origin points
    inline void setOriginPoints(ccPointCloud* cloud) { m_originPoints = cloud; }

    //! Gets normal vector mesh
    std::shared_ptr<ccMesh> getNormalVectorMesh(bool update = false);

    //! Clones this facet
    ccFacet* clone() const;
    bool clone(ccFacet* facet) const;

    virtual bool IsEmpty() const override;
    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;
    virtual ccBBox GetAxisAlignedBoundingBox() const override;
    virtual ecvOrientedBBox GetOrientedBoundingBox() const override;
    virtual ccFacet& Transform(const Eigen::Matrix4d& transformation) override;
    virtual ccFacet& Translate(const Eigen::Vector3d& translation,
                               bool relative = true) override;
    virtual ccFacet& Scale(const double s,
                           const Eigen::Vector3d& center) override;
    virtual ccFacet& Rotate(const Eigen::Matrix3d& R,
                            const Eigen::Vector3d& center) override;

    //! Copy constructor
    /** \param poly polyline to clone
     **/
    ccFacet(const ccFacet& poly);

    ccFacet& operator+=(const ccFacet& polyline);
    ccFacet& operator=(const ccFacet& polyline);
    ccFacet operator+(const ccFacet& polyline) const;

    /// \brief Assigns each line in the LineSet the same color.
    ///
    /// \param color Specifies the color to be applied.
    ccFacet& PaintUniformColor(const Eigen::Vector3d& color) {
        setColor(ecvColor::Rgb::FromEigen(color));
        return (*this);
    }

protected:
    // inherited from ccDrawable
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

    //! Creates internal representation (polygon, polyline, etc.)
    bool createInternalRepresentation(
            cloudViewer::GenericIndexedCloudPersist* points,
            const PointCoordinateType* planeEquation = nullptr);

    //! for python interface use
    std::shared_ptr<ccMesh> m_arrow;

    //! Facet
    ccMesh* m_polygonMesh;
    //! Facet contour
    ccPolyline* m_contourPolyline;
    //! Shared vertices (between polygon and contour)
    ccPointCloud* m_contourVertices;
    //! Origin points
    ccPointCloud* m_originPoints;

    //! Plane equation - as usual in CC plane equation is ax + by + cz = d
    PointCoordinateType m_planeEquation[4];

    //! Facet centroid
    CCVector3 m_center;

    //! RMS (relatively to m_associatedPoints)
    double m_rms;

    //! Surface (m_polygon)
    double m_surface;

    //! Max length
    PointCoordinateType m_maxEdgeLength;

    // inherited from ccHObject
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;

    // ccHObject interface
    virtual void applyGLTransformation(const ccGLMatrix& trans) override;
};
