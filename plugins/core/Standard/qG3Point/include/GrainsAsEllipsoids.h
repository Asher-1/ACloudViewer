// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// qCC_db
#include <ecvAdvancedTypes.h>
#include <ecvBBox.h>
#include <ecvColorTypes.h>
#include <ecvCustomObject.h>
#include <ecvDisplayTools.h>
#include <ecvHObject.h>
#include <ecvSerializableObject.h>

#include <QObject>
#include <memory>
#include <set>
#include <vector>

// Eigen
#include "Eigen/Dense"

class ccPointCloud;
class ccMesh;
namespace cloudViewer {
namespace geometry {
class LineSet;
}
}  // namespace cloudViewer

class GrainsAsEllipsoids : public QObject, public ccCustomHObject {
    Q_OBJECT

public:
    typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> Xb;

    GrainsAsEllipsoids(ecvMainAppInterface* app);

    GrainsAsEllipsoids(ccPointCloud* cloud,
                       ecvMainAppInterface* app,
                       const std::vector<std::vector<int>>& stacks,
                       const RGBAColorsTableType& colors);
    ~GrainsAsEllipsoids();

    void setLocalMaximumIndexes(const Eigen::ArrayXi& localMaximumIndexes);

    void setGrainColorsTable(const RGBAColorsTableType& colorTable);

    bool exportResultsAsCloud();

    // INIT SPHERE

    void initSphereVertices();

    void initSphereIndexes();

    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texCoords;

    std::vector<int> indices;
    std::vector<int> lineIndices;

    int sectorCount{21};
    int stackCount{21};

    // ELLIPSOID FITTING

    enum Method { DIRECT = 0 };

    void updateBBoxOnlyOne(int index);

    bool explicitToImplicit(const Eigen::Array3f& center,
                            const Eigen::Array3f& radii,
                            const Eigen::Matrix3f& rotationMatrix,
                            Eigen::ArrayXd& parameters);

    bool implicitToExplicit(const Eigen::ArrayXd& parameters,
                            Eigen::Array3f& center,
                            Eigen::Array3f& radii,
                            Eigen::Matrix3f& rotationMatrix);

    bool directFit(const Eigen::ArrayX3d& xyz, Eigen::ArrayXd& parameters);

    bool fitEllipsoidToGrain(const int grainIndex,
                             Eigen::Array3f& center,
                             Eigen::Array3f& radii,
                             Eigen::Matrix3f& rotationMatrix,
                             const Method& method = DIRECT);

    // DRAW

    //! Update mesh and lineset representations for ellipsoids
    void updateMeshAndLineSet();

    void setOnlyOne(int i);

    void showOnlyOne(bool state);

    void showAll(bool state);

    void setTransparency(double transparency) {
        m_transparency = transparency;
        m_meshNeedsUpdate = true;
        redrawDisplay();
    }
    void drawSurfaces(bool state) {
        m_drawSurfaces = state;
        m_meshNeedsUpdate = true;
        redrawDisplay();
    }
    void drawLines(bool state) {
        m_drawLines = state;
        m_meshNeedsUpdate = true;
        redrawDisplay();
    }
    void drawPoints(bool state) {
        m_drawPoints = state;
        m_meshNeedsUpdate = true;
        redrawDisplay();
    }
    void setGLPointSize(int size) {
        m_glPointSize = size;
        m_meshNeedsUpdate = true;
        redrawDisplay();
    }

    //! Clears all generated visual objects (meshes, linesets, point clouds)
    //! Call this when the underlying data (clusters) changes significantly.
    //! @param manageDBTree If true, temporarily detaches this object from the
    //! DB tree to avoid crashes
    void clearGeneratedObjects();

    // Inherited from ccHObject

    void draw(CC_DRAW_CONTEXT& context) override;

    ccBBox getOwnBB(bool withGLFeatures = false) override;

    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;

    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;

    ccPointCloud* m_cloud;
    ecvMainAppInterface* m_app;
    std::vector<std::vector<int>> m_stacks;
    std::vector<CCVector3f> m_grainColors;
    ccBBox m_ccBBoxOnlyOne;
    ccBBox m_ccBBoxAll;
    ccBBox m_ccBBox;

    Eigen::ArrayX3f ellipsoidInstance;

    std::vector<Eigen::Array3f> m_center;
    std::vector<Eigen::Array3f> m_radii;
    std::vector<Eigen::Matrix3f> m_rotationMatrix;
    std::set<int> m_fitNotOK;
    double m_transparency = 1.0;
    bool m_drawSurfaces = true;
    bool m_drawLines = true;
    bool m_drawPoints = false;

    int m_onlyOne = 0;
    bool m_showAll{true};
    int m_glPointSize = 3;

    // Mesh and LineSet representations for rendering
    std::vector<ccMesh*> m_meshes;
    std::vector<cloudViewer::geometry::LineSet*> m_lineSets;
    std::vector<ccPointCloud*>
            m_pointsClouds;  // Point clouds for drawPoints mode
    bool m_meshNeedsUpdate = true;
};
