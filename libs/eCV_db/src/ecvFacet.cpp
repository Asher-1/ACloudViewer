//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#include "ecvFacet.h"

#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "ecvPolyline.h"

// CORE_DB_LIB
#include <Delaunay2dMesh.h>
#include <DistanceComputationTools.h>
#include <Logging.h>
#include <MeshSamplingTools.h>
#include <PointProjectionTools.h>

static const char DEFAULT_POLYGON_MESH_NAME[] = "2D polygon";
static const char DEFAULT_CONTOUR_NAME[] = "Contour";
static const char DEFAULT_CONTOUR_POINTS_NAME[] = "Contour points";
static const char DEFAULT_ORIGIN_POINTS_NAME[] = "Origin points";

ccFacet::ccFacet(PointCoordinateType maxEdgeLength /*=0*/,
                 QString name /*=QString("Facet")*/)
    : ccHObject(name),
      ccPlanarEntityInterface(getUniqueID()),
      m_arrow(nullptr),
      m_polygonMesh(nullptr),
      m_contourPolyline(nullptr),
      m_contourVertices(nullptr),
      m_originPoints(nullptr),
      m_center(0, 0, 0),
      m_rms(0.0),
      m_surface(0.0),
      m_maxEdgeLength(maxEdgeLength) {
    m_planeEquation[0] = 0;
    m_planeEquation[1] = 0;
    m_planeEquation[2] = 1;
    m_planeEquation[3] = 0;

    setVisible(true);
    lockVisibility(false);
}

ccFacet::ccFacet(const ccFacet& poly)
    : ccHObject(poly.getName()),
      ccPlanarEntityInterface(getUniqueID()),
      m_arrow(nullptr),
      m_polygonMesh(nullptr),
      m_contourPolyline(nullptr),
      m_contourVertices(nullptr),
      m_originPoints(nullptr),
      m_center(0, 0, 0),
      m_rms(0.0),
      m_surface(0.0),
      m_maxEdgeLength(0) {
    poly.clone(this);
}

ccFacet::~ccFacet() {}

ccFacet* ccFacet::clone() const {
    ccFacet* facet = new ccFacet(m_maxEdgeLength, m_name);

    if (!clone(facet)) {
        return nullptr;
    }

    return facet;
}

bool ccFacet::clone(ccFacet* facet) const {
    if (!facet || this->IsEmpty()) {
        return false;
    }

    // clone contour
    if (m_contourPolyline) {
        assert(m_contourVertices);
        facet->m_contourPolyline = new ccPolyline(*m_contourPolyline);
        facet->m_contourVertices = dynamic_cast<ccPointCloud*>(
                facet->m_contourPolyline->getAssociatedCloud());

        if (!facet->m_contourPolyline || !facet->m_contourVertices) {
            // not enough memory?!
            CVLog::Warning(
                    QString("[ccFacet::clone][%1] Failed to clone countour!")
                            .arg(getName()));
            delete facet;
            return false;
        }

        // the copy constructor of ccFacet creates a new cloud (the copy of this
        // facet's 'contour points') but set it by default as a child of the
        // polyline (while we want the opposite in a facet)
        facet->m_contourPolyline->detachChild(facet->m_contourVertices);

        facet->m_contourPolyline->setLocked(m_contourPolyline->isLocked());
        facet->m_contourVertices->setEnabled(m_contourVertices->isEnabled());
        facet->m_contourVertices->setVisible(m_contourVertices->isVisible());
        facet->m_contourVertices->setLocked(m_contourVertices->isLocked());
        facet->m_contourVertices->setName(m_contourVertices->getName());
        facet->m_contourVertices->addChild(facet->m_contourPolyline);
        facet->addChild(facet->m_contourVertices);
    }

    // clone mesh
    if (m_polygonMesh) {
        facet->m_polygonMesh =
                m_polygonMesh->cloneMesh(facet->m_contourVertices);
        if (!facet->m_polygonMesh) {
            // not enough memory?!
            CVLog::Warning(
                    QString("[ccFacet::clone][%1] Failed to clone polygon!")
                            .arg(getName()));
            delete facet;
            return false;
        }

        facet->m_polygonMesh->setLocked(m_polygonMesh->isLocked());
        facet->m_polygonMesh->setName(m_polygonMesh->getName());
        if (facet->m_contourVertices)
            facet->m_contourVertices->addChild(facet->m_polygonMesh);
        else
            facet->addChild(facet->m_polygonMesh);
    }

    if (m_originPoints) {
        facet->m_originPoints =
                dynamic_cast<ccPointCloud*>(m_originPoints->clone());
        if (!facet->m_originPoints) {
            CVLog::Warning(QString("[ccFacet::clone][%1] Failed to clone "
                                   "origin points!")
                                   .arg(getName()));
            // delete facet;
            // return 0;
        } else {
            facet->m_originPoints->setLocked(m_originPoints->isLocked());
            facet->m_originPoints->setName(m_originPoints->getName());
            facet->addChild(facet->m_originPoints);
        }
    }

    if (m_arrow) {
        if (!facet->m_arrow) {
            facet->m_arrow = std::shared_ptr<ccMesh>();
            facet->m_arrow->CreateInternalCloud();
        }
        *facet->m_arrow = *m_arrow;
    }

    facet->m_center = m_center;
    facet->m_rms = m_rms;
    facet->m_surface = m_surface;
    facet->m_showNormalVector = m_showNormalVector;
    memcpy(facet->m_planeEquation, m_planeEquation,
           sizeof(PointCoordinateType) * 4);
    facet->setVisible(isVisible());
    facet->lockVisibility(isVisibilityLocked());

    return true;
}

ccFacet* ccFacet::Create(cloudViewer::GenericIndexedCloudPersist* cloud,
                         PointCoordinateType maxEdgeLength /*=0*/,
                         bool transferOwnership /*=false*/,
                         const PointCoordinateType* planeEquation /*=0*/) {
    assert(cloud);

    // we need at least 3 points to compute a mesh or a plane! ;)
    if (!cloud || cloud->size() < 3) {
        CVLog::Error(
                "[ccFacet::Create] Need at least 3 points to create a valid "
                "facet!");
        return nullptr;
    }

    // create facet structure
    ccFacet* facet = new ccFacet(maxEdgeLength, "facet");
    if (!facet->createInternalRepresentation(cloud, planeEquation)) {
        delete facet;
        return nullptr;
    }

    ccPointCloud* pc = dynamic_cast<ccPointCloud*>(cloud);
    if (pc) {
        facet->setName(pc->getName() + QString(".facet"));
        if (facet->getPolygon()) {
            facet->getPolygon()->setOpacity(0.5f);
            facet->getPolygon()->setTempColor(ecvColor::darkGrey);
        }
        if (facet->getContour()) {
            facet->getContour()->setColor(ecvColor::green);
            facet->getContour()->showColors(true);
        }

        if (transferOwnership) {
            pc->setName(DEFAULT_ORIGIN_POINTS_NAME);
            pc->setEnabled(false);
            pc->setLocked(true);
            facet->addChild(pc);
            facet->setOriginPoints(pc);
        }
    }

    return facet;
}

bool ccFacet::createInternalRepresentation(
        cloudViewer::GenericIndexedCloudPersist* points,
        const PointCoordinateType* planeEquation /*=0*/) {
    assert(points);
    if (!points) return false;
    unsigned ptsCount = points->size();
    if (ptsCount < 3) return false;

    cloudViewer::Neighbourhood Yk(points);

    // get corresponding plane
    if (!planeEquation) {
        planeEquation = Yk.getLSPlane();
        if (!planeEquation) {
            CVLog::Warning(
                    "[ccFacet::createInternalRepresentation] Failed to compute "
                    "the LS plane passing through the input points!");
            return false;
        }
    }
    memcpy(m_planeEquation, planeEquation, sizeof(PointCoordinateType) * 4);

    // we project the input points on a plane
    std::vector<cloudViewer::PointProjectionTools::IndexedCCVector2> points2D;
    // local base
    CCVector3 X;
    CCVector3 Y;

    if (!Yk.projectPointsOn2DPlane<
                cloudViewer::PointProjectionTools::IndexedCCVector2>(
                points2D, nullptr, &m_center, &X, &Y)) {
        CVLog::Error(
                "[ccFacet::createInternalRepresentation] Not enough memory!");
        return false;
    }

    // compute resulting RMS
    m_rms = cloudViewer::DistanceComputationTools::
            computeCloud2PlaneDistanceRMS(points, m_planeEquation);

    // update the points indexes (not done by
    // Neighbourhood::projectPointsOn2DPlane)
    {
        for (unsigned i = 0; i < ptsCount; ++i) {
            points2D[i].index = i;
        }
    }

    // try to get the points on the convex/concave hull to build the contour and
    // the polygon
    {
        std::list<cloudViewer::PointProjectionTools::IndexedCCVector2*>
                hullPoints;
        if (!cloudViewer::PointProjectionTools::extractConcaveHull2D(
                    points2D, hullPoints, m_maxEdgeLength * m_maxEdgeLength)) {
            CVLog::Error(
                    "[ccFacet::createInternalRepresentation] Failed to compute "
                    "the convex hull of the input points!");
        }

        unsigned hullPtsCount = static_cast<unsigned>(hullPoints.size());

        // create vertices
        m_contourVertices = new ccPointCloud();
        {
            if (!m_contourVertices->reserve(hullPtsCount)) {
                delete m_contourVertices;
                m_contourVertices = nullptr;
                CVLog::Error(
                        "[ccFacet::createInternalRepresentation] Not enough "
                        "memory!");
                return false;
            }

            // projection on the LS plane (in 3D)
            for (std::list<cloudViewer::PointProjectionTools::
                                   IndexedCCVector2*>::const_iterator it =
                         hullPoints.begin();
                 it != hullPoints.end(); ++it) {
                m_contourVertices->addPoint(m_center + X * (*it)->x +
                                            Y * (*it)->y);
            }
            m_contourVertices->setName(DEFAULT_CONTOUR_POINTS_NAME);
            m_contourVertices->setLocked(true);
            m_contourVertices->setEnabled(false);
            addChild(m_contourVertices);
        }

        // we create the corresponding (3D) polyline
        {
            m_contourPolyline = new ccPolyline(m_contourVertices);
            if (m_contourPolyline->reserve(hullPtsCount)) {
                m_contourPolyline->addPointIndex(0, hullPtsCount);
                m_contourPolyline->setClosed(true);
                m_contourPolyline->setVisible(true);
                m_contourPolyline->setLocked(true);
                m_contourPolyline->setName(DEFAULT_CONTOUR_NAME);
                m_contourVertices->addChild(m_contourPolyline);
                m_contourVertices->setEnabled(true);
                m_contourVertices->setVisible(false);
            } else {
                delete m_contourPolyline;
                m_contourPolyline = nullptr;
                CVLog::Warning(
                        "[ccFacet::createInternalRepresentation] Not enough "
                        "memory to create the contour polyline!");
            }
        }

        // we create the corresponding (2D) mesh
        std::vector<CCVector2> hullPointsVector;
        try {
            hullPointsVector.reserve(hullPoints.size());
            for (std::list<cloudViewer::PointProjectionTools::
                                   IndexedCCVector2*>::const_iterator it =
                         hullPoints.begin();
                 it != hullPoints.end(); ++it) {
                hullPointsVector.push_back(**it);
            }
        } catch (...) {
            CVLog::Warning(
                    "[ccFacet::createInternalRepresentation] Not enough memory "
                    "to create the contour mesh!");
        }

        // if we have computed a concave hull, we must remove triangles falling
        // outside!
        bool removePointsOutsideHull = (m_maxEdgeLength > 0);

        if (!hullPointsVector.empty() &&
            cloudViewer::Delaunay2dMesh::Available()) {
            // compute the facet surface
            cloudViewer::Delaunay2dMesh dm;
            std::string errorStr;
            if (dm.buildMesh(hullPointsVector,
                             cloudViewer::Delaunay2dMesh::USE_ALL_POINTS,
                             errorStr)) {
                if (removePointsOutsideHull)
                    dm.removeOuterTriangles(hullPointsVector, hullPointsVector);
                unsigned triCount = dm.size();
                assert(triCount != 0);

                m_polygonMesh = new ccMesh(m_contourVertices);
                if (m_polygonMesh->reserve(triCount)) {
                    // import faces
                    for (unsigned i = 0; i < triCount; ++i) {
                        const cloudViewer::VerticesIndexes* tsi =
                                dm.getTriangleVertIndexes(i);
                        m_polygonMesh->addTriangle(tsi->i1, tsi->i2, tsi->i3);
                    }
                    m_polygonMesh->setVisible(true);
                    m_polygonMesh->enableStippling(true);

                    // unique normal for facets
                    if (m_polygonMesh->reservePerTriangleNormalIndexes()) {
                        NormsIndexesTableType* normsTable =
                                new NormsIndexesTableType();
                        normsTable->reserve(1);
                        CCVector3 N(m_planeEquation);
                        normsTable->addElement(
                                ccNormalVectors::GetNormIndex(N.u));
                        m_polygonMesh->setTriNormsTable(normsTable);
                        for (unsigned i = 0; i < triCount; ++i)
                            m_polygonMesh->addTriangleNormalIndexes(
                                    0, 0, 0);  // all triangles will have the
                                               // same normal!
                        m_polygonMesh->showNormals(true);
                        m_polygonMesh->setLocked(true);
                        m_polygonMesh->setName(DEFAULT_POLYGON_MESH_NAME);
                        m_contourVertices->addChild(m_polygonMesh);
                        m_contourVertices->setEnabled(true);
                        m_contourVertices->setVisible(false);
                    } else {
                        CVLog::Warning(
                                "[ccFacet::createInternalRepresentation] Not "
                                "enough memory to create the polygon mesh's "
                                "normals!");
                    }

                    // update facet surface
                    m_surface = cloudViewer::MeshSamplingTools::computeMeshArea(
                            m_polygonMesh);
                } else {
                    delete m_polygonMesh;
                    m_polygonMesh = nullptr;
                    CVLog::Warning(
                            "[ccFacet::createInternalRepresentation] Not "
                            "enough memory to create the polygon mesh!");
                }
            } else {
                CVLog::Warning(QString("[ccFacet::createInternalRepresentation]"
                                       " Failed to create the polygon mesh "
                                       "(third party lib. said '%1'")
                                       .arg(QString::fromStdString(errorStr)));
            }
        }
    }

    return true;
}

void ccFacet::setColor(const ecvColor::Rgb& rgb) {
    if (m_contourVertices && m_contourVertices->setRGBColor(rgb)) {
        m_contourVertices->showColors(true);
        if (m_polygonMesh) m_polygonMesh->showColors(true);
    }

    if (m_contourPolyline) {
        m_contourPolyline->setColor(rgb);
        m_contourPolyline->showColors(true);
    }
    showColors(true);
}

std::shared_ptr<ccMesh> ccFacet::getNormalVectorMesh(bool update) {
    // const auto boundingbox = GetAxisAlignedBoundingBox();
    // double scale = std::max(0.01, boundingbox.GetMaxExtent() * 0.2);
    if (normalVectorIsShown() && update || !m_arrow) {
        PointCoordinateType scale = 1.0;
        // the surface might be 0 if Delaunay 2.5D triangulation is not
        // supported
        if (m_surface > 0) {
            scale = sqrt(m_surface);
        } else {
            scale = sqrt(m_contourPolyline->computeLength());
        }
        m_arrow = ccMesh::CreateArrow(0.02 * scale, 0.05 * scale, 0.9 * scale,
                                      0.1 * scale);
        // m_arrow->ComputeVertexNormals();
        m_arrow->setTempColor(m_contourPolyline->getColor());
        m_arrow->showColors(true);

        Eigen::Matrix4d transformation;
        ccGLMatrix mat = ccGLMatrix::FromToRotation(CCVector3(0, 0, PC_ONE),
                                                    getNormal());
        transformation = ccGLMatrixd::ToEigenMatrix4(mat);
        m_arrow->Transform(transformation);

        transformation = Eigen::Matrix4d::Identity();
        transformation.block<3, 1>(0, 3) = CCVector3d::fromArray(getCenter());
        m_arrow->Transform(transformation);
    }

    return m_arrow;
}

void ccFacet::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (!MACRO_Draw3D(context)) return;

    if (!isRedraw()) {
        return;
    }

    // show normal vector
    if (m_contourPolyline) {
        PointCoordinateType scale = 1.0;
        if (normalVectorIsShown()) {
            if (m_surface > 0)  // the surface might be 0 if Delaunay 2.5D
                                // triangulation is not supported
            {
                scale = sqrt(m_surface);
            } else {
                scale = sqrt(m_contourPolyline->computeLength());
            }
        }

        glDrawNormal(context, m_center, scale, &m_contourPolyline->getColor());
    }
}

bool ccFacet::toFile_MeOnly(QFile& out) const {
    if (!ccHObject::toFile_MeOnly(out)) return false;

    // we can't save the origin points here (as it will be automatically saved
    // as a child) so instead we save it's unique ID (dataVersion>=32) WARNING:
    // the cloud must be saved in the same BIN file! (responsibility of the
    // caller)
    {
        uint32_t originPointsUniqueID =
                (m_originPoints ? (uint32_t)m_originPoints->getUniqueID() : 0);
        if (out.write((const char*)&originPointsUniqueID, 4) < 0)
            return WriteError();
    }

    // we can't save the contour points here (as it will be automatically saved
    // as a child) so instead we save it's unique ID (dataVersion>=32) WARNING:
    // the cloud must be saved in the same BIN file! (responsibility of the
    // caller)
    {
        uint32_t contourPointsUniqueID =
                (m_contourVertices ? (uint32_t)m_contourVertices->getUniqueID()
                                   : 0);
        if (out.write((const char*)&contourPointsUniqueID, 4) < 0)
            return WriteError();
    }

    // we can't save the contour polyline here (as it will be automatically
    // saved as a child) so instead we save it's unique ID (dataVersion>=32)
    // WARNING: the polyline must be saved in the same BIN file! (responsibility
    // of the caller)
    {
        uint32_t contourPolyUniqueID =
                (m_contourPolyline ? (uint32_t)m_contourPolyline->getUniqueID()
                                   : 0);
        if (out.write((const char*)&contourPolyUniqueID, 4) < 0)
            return WriteError();
    }

    // we can't save the polygon mesh here (as it will be automatically saved as
    // a child) so instead we save it's unique ID (dataVersion>=32) WARNING: the
    // mesh must be saved in the same BIN file! (responsibility of the caller)
    {
        uint32_t polygonMeshUniqueID =
                (m_polygonMesh ? (uint32_t)m_polygonMesh->getUniqueID() : 0);
        if (out.write((const char*)&polygonMeshUniqueID, 4) < 0)
            return WriteError();
    }

    // plane equation (dataVersion>=32)
    if (out.write((const char*)&m_planeEquation,
                  sizeof(PointCoordinateType) * 4) < 0)
        return WriteError();

    // center (dataVersion>=32)
    if (out.write((const char*)m_center.u, sizeof(PointCoordinateType) * 3) < 0)
        return WriteError();

    // RMS (dataVersion>=32)
    if (out.write((const char*)&m_rms, sizeof(double)) < 0) return WriteError();

    // surface (dataVersion>=32)
    if (out.write((const char*)&m_surface, sizeof(double)) < 0)
        return WriteError();

    // Max edge length (dataVersion>=31)
    if (out.write((const char*)&m_maxEdgeLength, sizeof(PointCoordinateType)) <
        0)
        return WriteError();

    return true;
}

bool ccFacet::fromFile_MeOnly(QFile& in,
                              short dataVersion,
                              int flags,
                              LoadedIDMap& oldToNewIDMap) {
    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    if (dataVersion < 32) return false;

    // origin points (dataVersion>=32)
    // as the cloud will be saved automatically (as a child)
    // we only store its unique ID --> we hope we will find it at loading time
    {
        uint32_t origPointsUniqueID = 0;
        if (in.read((char*)&origPointsUniqueID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the cloud unique ID in the
        //'m_originPoints' pointer!!!
        *(uint32_t*)(&m_originPoints) = origPointsUniqueID;
    }

    // contour points
    // as the cloud will be saved automatically (as a child)
    // we only store its unique ID --> we hope we will find it at loading time
    {
        uint32_t contourPointsUniqueID = 0;
        if (in.read((char*)&contourPointsUniqueID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the cloud unique ID in the
        //'m_contourVertices' pointer!!!
        *(uint32_t*)(&m_contourVertices) = contourPointsUniqueID;
    }

    // contour points
    // as the polyline will be saved automatically (as a child)
    // we only store its unique ID --> we hope we will find it at loading time
    {
        uint32_t contourPolyUniqueID = 0;
        if (in.read((char*)&contourPolyUniqueID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the polyline unique ID in the
        //'m_contourPolyline' pointer!!!
        *(uint32_t*)(&m_contourPolyline) = contourPolyUniqueID;
    }

    // polygon mesh
    // as the mesh will be saved automatically (as a child)
    // we only store its unique ID --> we hope we will find it at loading time
    {
        uint32_t polygonMeshUniqueID = 0;
        if (in.read((char*)&polygonMeshUniqueID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the polyline unique ID in the
        //'m_contourPolyline' pointer!!!
        *(uint32_t*)(&m_polygonMesh) = polygonMeshUniqueID;
    }

    // plane equation (dataVersion>=32)
    if (in.read((char*)&m_planeEquation, sizeof(PointCoordinateType) * 4) < 0)
        return ReadError();

    // center (dataVersion>=32)
    if (in.read((char*)m_center.u, sizeof(PointCoordinateType) * 3) < 0)
        return ReadError();

    // RMS (dataVersion>=32)
    if (in.read((char*)&m_rms, sizeof(double)) < 0) return ReadError();

    // surface (dataVersion>=32)
    if (in.read((char*)&m_surface, sizeof(double)) < 0) return ReadError();

    // Max edge length (dataVersion>=31)
    if (in.read((char*)&m_maxEdgeLength, sizeof(PointCoordinateType)) < 0)
        return WriteError();

    return true;
}

void ccFacet::applyGLTransformation(const ccGLMatrix& trans) {
    ccHObject::applyGLTransformation(trans);

    // move/rotate the center to its new location
    trans.apply(m_center);

    // apply the rotation to the normal of the plane equation
    trans.applyRotation(m_planeEquation);

    // compute new d-parameter from the updated values
    CCVector3 n(m_planeEquation);
    m_planeEquation[3] = n.dot(m_center);
}

void ccFacet::invertNormal() {
    for (int i = 0; i < 4; ++i) {
        m_planeEquation[i] = -m_planeEquation[i];
    }
}

bool ccFacet::IsEmpty() const {
    return (!m_polygonMesh || m_polygonMesh->size() == 0 ||
            !m_contourPolyline || m_contourPolyline->size() == 0);
}

Eigen::Vector3d ccFacet::GetMinBound() const {
    if (getPolygon()) {
        return getPolygon()->GetMinBound();
    } else {
        return Eigen::Vector3d();
    }
}

Eigen::Vector3d ccFacet::GetMaxBound() const {
    if (getPolygon()) {
        return getPolygon()->GetMaxBound();
    } else {
        return Eigen::Vector3d();
    }
}

Eigen::Vector3d ccFacet::GetCenter() const {
    if (getPolygon()) {
        return getPolygon()->GetCenter();
    } else {
        return Eigen::Vector3d();
    }
}

ccBBox ccFacet::GetAxisAlignedBoundingBox() const {
    if (getPolygon()) {
        return getPolygon()->GetAxisAlignedBoundingBox();
    } else {
        return ccBBox();
    }
}

ecvOrientedBBox ccFacet::GetOrientedBoundingBox() const {
    if (getPolygon()) {
        return getPolygon()->GetOrientedBoundingBox();
    } else {
        return ecvOrientedBBox();
    }
}

ccFacet& ccFacet::Transform(const Eigen::Matrix4d& transformation) {
    if (!getPolygon()) {
        return *this;
    }

    getPolygon()->Transform(transformation);
    return *this;
}

ccFacet& ccFacet::Translate(const Eigen::Vector3d& translation, bool relative) {
    if (!getPolygon()) {
        return *this;
    }

    getPolygon()->Translate(translation, relative);
    return *this;
}

ccFacet& ccFacet::Scale(const double s, const Eigen::Vector3d& center) {
    if (!getPolygon()) {
        return *this;
    }

    getPolygon()->Scale(s, center);
    return *this;
}

ccFacet& ccFacet::Rotate(const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& center) {
    if (!getPolygon()) {
        return *this;
    }

    getPolygon()->Rotate(R, center);
    return *this;
}

ccFacet& ccFacet::operator+=(const ccFacet& facet) {
    cloudViewer::utility::LogWarning("ccFace does not support '+=' operator!");
    return (*this);
}

ccFacet& ccFacet::operator=(const ccFacet& facet) {
    if (facet.IsEmpty()) return (*this);
    if (this == &facet) {
        return (*this);
    }

    facet.clone(this);
    return (*this);
}

ccFacet ccFacet::operator+(const ccFacet& facet) const {
    cloudViewer::utility::LogWarning("ccFace does not support '=' operator!");
    return ccFacet();
}
