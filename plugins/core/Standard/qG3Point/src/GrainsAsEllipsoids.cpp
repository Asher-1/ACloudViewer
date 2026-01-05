// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "GrainsAsEllipsoids.h"

/// qCC_db
#include <LineSet.h>
#include <ReferenceCloud.h>
#include <ecvGLMatrix.h>
#include <ecvMaterial.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvSerializableObject.h>

#include <QCoreApplication>
#include <QDir>
#include <fstream>
#include <iostream>

#include "G3PointAction.h"

GrainsAsEllipsoids::GrainsAsEllipsoids(ecvMainAppInterface* app)
    : m_app(app), m_cloud(nullptr) {
    assert(m_app);

    this->setMetaData("class_name", "GrainsAsEllipsoids");
    this->setMetaData("plugin_name", "G3Point");
}

GrainsAsEllipsoids::~GrainsAsEllipsoids() {}

void GrainsAsEllipsoids::clearGeneratedObjects() {
    // Clean up mesh and lineset children
    for (auto* mesh : m_meshes) {
        if (mesh) {
            removeChild(mesh);
            delete mesh;
        }
    }
    m_meshes.clear();

    for (auto* lineSet : m_lineSets) {
        if (lineSet) {
            removeChild(lineSet);
            delete lineSet;
        }
    }
    m_lineSets.clear();

    // Clean up points clouds
    for (auto* pc : m_pointsClouds) {
        if (pc) {
            removeChild(pc);
            delete pc;
        }
    }
    m_pointsClouds.clear();
}

GrainsAsEllipsoids::GrainsAsEllipsoids(
        ccPointCloud* cloud,
        ecvMainAppInterface* app,
        const std::vector<std::vector<int>>& stacks,
        const RGBAColorsTableType& colors)
    : m_cloud(cloud), m_app(app), m_stacks(stacks) {
    this->setMetaData("class_name", "GrainsAsEllipsoids");
    this->setMetaData("plugin_name", "G3Point");

    setGrainColorsTable(colors);

    m_center.resize(m_stacks.size());
    m_radii.resize(m_stacks.size());
    m_rotationMatrix.resize(m_stacks.size());

    // fit all ellipsoids
    std::cout << "[GrainsAsEllipsoids::GrainsAsEllipsoids] fit "
              << stacks.size() << " ellipsoids" << std::endl;

    lockVisibility(false);
    setVisible(true);

    m_ccBBoxAll.setValidity(false);
    m_ccBBoxAll.clear();

    for (int idx = 0; idx < m_stacks.size(); idx++) {
        if (!fitEllipsoidToGrain(idx, m_center[idx], m_radii[idx],
                                 m_rotationMatrix[idx])) {
            m_fitNotOK.insert(idx);
            CVLog::Warning(
                    "[GrainsAsEllipsoids::GrainsAsEllipsoids] fit not possible "
                    "for grain " +
                    QString::number(idx) + " of size " +
                    QString::number(m_stacks[idx].size()));
        } else {  // update the bounding box
            float maxRadius = m_radii[idx].maxCoeff();
            CCVector3 center(m_center[idx](0), m_center[idx](1),
                             m_center[idx](2));
            m_ccBBoxAll.add(CCVector3(center.x + maxRadius,
                                      center.y + maxRadius,
                                      center.z + maxRadius));
            m_ccBBoxAll.add(CCVector3(center.x - maxRadius,
                                      center.y - maxRadius,
                                      center.z - maxRadius));
        }
    }

    // remove data corresponding to stacks were the fit was not successful
    for (auto el : m_fitNotOK) {
        // if the fit is not OK, we use the centroid as a center
        int nPoints = static_cast<int>(m_stacks[el].size());
        Eigen::MatrixX3d points(nPoints, 3);
        for (int index = 0; index < nPoints; index++) {
            const CCVector3* point = m_cloud->getPoint(m_stacks[el][index]);
            points(index, 0) = point->x;
            points(index, 1) = point->y;
            points(index, 2) = point->z;
        }
        // compute the centroid of the label
        Eigen::RowVector3d centroid = points.colwise().mean();
        m_center[el] << centroid.x(), centroid.y(), centroid.z();

        m_radii[el].fill(0);

        m_rotationMatrix[el].fill(NAN);
    }

    m_ccBBoxAll.setValidity(true);
    m_meshNeedsUpdate = true;
}

void GrainsAsEllipsoids::setGrainColorsTable(
        const RGBAColorsTableType& colorTable) {
    m_grainColors.resize(colorTable.size());

    for (int k = 0; k < colorTable.size(); k++) {
        ecvColor::Rgba color = colorTable[k];
        m_grainColors[k] =
                CCVector3f(static_cast<float>(color.r) / ecvColor::MAX,
                           static_cast<float>(color.g) / ecvColor::MAX,
                           static_cast<float>(color.b) / ecvColor::MAX);
    }
}

bool GrainsAsEllipsoids::exportResultsAsCloud() {
    // create cloud
    QString cloudName = "g3point_results";
    ccPointCloud* cloud = new ccPointCloud(cloudName);

    for (int idx = 0; idx < m_center.size(); idx++) {
        // if (m_fitNotOK.count(idx))
        // {
        // 	continue;
        // }
        Eigen::Vector3f center{m_center[idx].x(), m_center[idx].y(),
                               m_center[idx].z()};
        Eigen::Vector3f point = center;
        CCVector3 ccPoint(point(0), point(1), point(2));
        cloud->addPoint(ccPoint);
    }

    // allocate colors if necessary
    if (cloud->resizeTheRGBTable()) {
        for (unsigned int index = 0; index < cloud->size(); index++) {
            ecvColor::Rgb color(m_grainColors[index].x * ecvColor::MAX * 0.8,
                                m_grainColors[index].y * ecvColor::MAX * 0.8,
                                m_grainColors[index].z * ecvColor::MAX * 0.8);
            cloud->setPointColor(index, color);
        }
    }

    int sfIdx;
    cloudViewer::ScalarField* sf;

    // EXPORT g3point_index
    sfIdx = cloud->addScalarField("g3point_index");
    if (sfIdx == -1) {
        CVLog::Error(
                "[GrainsAsEllipsoids::exportResultsAsCloud] impossible to "
                "allocate g3point_index scalar field");
        return false;
    }
    sf = cloud->getScalarField(sfIdx);
    int indexInResults = 0;
    for (int index = 0; index < m_center.size(); index++) {
        // if (m_fitNotOK.count(index)) // when the fit was not successful, the
        // point is not exported
        // {
        // 	continue;
        // }
        sf->setValue(indexInResults, index);
        indexInResults++;
    }
    sf->computeMinAndMax();

    // <EXPORT RADII>
    int sfIdxRadiusX = cloud->addScalarField("g3point_radius_x");
    int sfIdxRadiusY = cloud->addScalarField("g3point_radius_y");
    int sfIdxRadiusZ = cloud->addScalarField("g3point_radius_z");
    if (sfIdxRadiusX == -1 || sfIdxRadiusY == -1 || sfIdxRadiusZ == -1) {
        CVLog::Error(
                "[GrainsAsEllipsoids::exportResultsAsCloud] impossible to "
                "allocate scalar fields to export the radii");
        return false;
    }
    cloudViewer::ScalarField* sfRadiusX = cloud->getScalarField(sfIdxRadiusX);
    cloudViewer::ScalarField* sfRadiusY = cloud->getScalarField(sfIdxRadiusY);
    cloudViewer::ScalarField* sfRadiusZ = cloud->getScalarField(sfIdxRadiusZ);
    for (unsigned int index = 0; index < cloud->size(); index++) {
        // if (m_fitNotOK.count(index))
        // {
        // 	continue;
        // }
        sfRadiusX->setValue(index, m_radii[index].x());
        sfRadiusY->setValue(index, m_radii[index].y());
        sfRadiusZ->setValue(index, m_radii[index].z());
    }
    sfRadiusX->computeMinAndMax();
    sfRadiusY->computeMinAndMax();
    sfRadiusZ->computeMinAndMax();
    // </EXPORT RADII>

    // <EXPORT ROTATION>
    int sfIdxR00 = cloud->addScalarField("g3point_r00");
    int sfIdxR01 = cloud->addScalarField("g3point_r01");
    int sfIdxR02 = cloud->addScalarField("g3point_r02");
    int sfIdxR10 = cloud->addScalarField("g3point_r10");
    int sfIdxR11 = cloud->addScalarField("g3point_r11");
    int sfIdxR21 = cloud->addScalarField("g3point_r12");
    int sfIdxR20 = cloud->addScalarField("g3point_r20");
    int sfIdxR12 = cloud->addScalarField("g3point_r21");
    int sfIdxR22 = cloud->addScalarField("g3point_r22");
    if (sfIdxR00 == -1 || sfIdxR01 == -1 || sfIdxR02 == -1 || sfIdxR10 == -1 ||
        sfIdxR11 == -1 || sfIdxR12 == -1 || sfIdxR20 == -1 || sfIdxR21 == -1 ||
        sfIdxR22 == -1) {
        CVLog::Error(
                "[GrainsAsEllipsoids::exportResultsAsCloud] impossible to "
                "allocate scalar fields to export the rotation");
        return false;
    }
    cloudViewer::ScalarField* sfR00 = cloud->getScalarField(sfIdxR00);
    cloudViewer::ScalarField* sfR01 = cloud->getScalarField(sfIdxR01);
    cloudViewer::ScalarField* sfR02 = cloud->getScalarField(sfIdxR02);
    cloudViewer::ScalarField* sfR10 = cloud->getScalarField(sfIdxR10);
    cloudViewer::ScalarField* sfR11 = cloud->getScalarField(sfIdxR11);
    cloudViewer::ScalarField* sfR12 = cloud->getScalarField(sfIdxR12);
    cloudViewer::ScalarField* sfR20 = cloud->getScalarField(sfIdxR20);
    cloudViewer::ScalarField* sfR21 = cloud->getScalarField(sfIdxR21);
    cloudViewer::ScalarField* sfR22 = cloud->getScalarField(sfIdxR22);
    for (unsigned int index = 0; index < cloud->size(); index++) {
        // if (m_fitNotOK.count(index))
        // {
        // 	continue;
        // }
        sfR00->setValue(index, m_rotationMatrix[index](0, 0));
        sfR01->setValue(index, m_rotationMatrix[index](0, 1));
        sfR02->setValue(index, m_rotationMatrix[index](0, 2));
        sfR10->setValue(index, m_rotationMatrix[index](1, 0));
        sfR11->setValue(index, m_rotationMatrix[index](1, 1));
        sfR12->setValue(index, m_rotationMatrix[index](1, 2));
        sfR20->setValue(index, m_rotationMatrix[index](2, 0));
        sfR21->setValue(index, m_rotationMatrix[index](2, 1));
        sfR22->setValue(index, m_rotationMatrix[index](2, 2));
    }
    sfR00->computeMinAndMax();
    sfR01->computeMinAndMax();
    sfR02->computeMinAndMax();
    sfR10->computeMinAndMax();
    sfR11->computeMinAndMax();
    sfR12->computeMinAndMax();
    sfR20->computeMinAndMax();
    sfR21->computeMinAndMax();
    sfR22->computeMinAndMax();
    // </EXPORT ROTATION>

    cloud->showColors(true);
    cloud->setPointSize(9);

    m_cloud->addChild(cloud);
    m_app->addToDB(cloud);

    return true;
}

// INIT ORIGINAL SPHERE

void GrainsAsEllipsoids::initSphereVertices() {
    // clear memory of prev arrays
    std::vector<float>().swap(vertices);
    std::vector<float>().swap(normals);
    std::vector<float>().swap(texCoords);

    float x, y, z, xy;  // vertex position
    float nx, ny, nz;   // vertex normal
    float s, t;         // vertex texCoord

    float sectorStep = 2 * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;
    float sectorAngle, stackAngle;

    for (int i = 0; i <= stackCount; ++i) {
        stackAngle = M_PI / 2 - i * stackStep;  // starting from pi/2 to -pi/2
        xy = cosf(stackAngle);                  // r * cos(u)
        z = sinf(stackAngle);                   // r * sin(u)

        // add (sectorCount+1) vertices per stack
        // first and last vertices have same position and normal, but different
        // tex coords
        for (int j = 0; j <= sectorCount; ++j) {
            sectorAngle = j * sectorStep;  // starting from 0 to 2pi

            // vertex position (x, y, z)
            x = xy * cosf(sectorAngle);  // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle);  // r * cos(u) * sin(v)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // normalized vertex normal (nx, ny, nz)
            nx = x;
            ny = y;
            nz = z;
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);

            // vertex tex coord (s, t) range between [0, 1]
            s = (float)j / sectorCount;
            t = (float)i / stackCount;
            texCoords.push_back(s);
            texCoords.push_back(t);
        }
    }
}

void GrainsAsEllipsoids::initSphereIndexes() {
    // Clear previous indices
    indices.clear();
    lineIndices.clear();

    // generate CCW index list of sphere triangles
    // k1--k1+1
    // |  / |
    // | /  |
    // k2--k2+1

    int k1, k2;
    for (int i = 0; i < stackCount; ++i) {
        k1 = i * (sectorCount + 1);  // beginning of current stack
        k2 = k1 + sectorCount + 1;   // beginning of next stack

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            // 2 triangles per sector excluding first and last stacks
            // k1 => k2 => k1+1
            if (i != 0) {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }

            // k1+1 => k2 => k2+1
            if (i != (stackCount - 1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }

            // store indices for lines
            // vertical lines for all stacks, k1 => k2
            lineIndices.push_back(k1);
            lineIndices.push_back(k2);
            if (i != 0)  // horizontal lines except 1st stack, k1 => k+1
            {
                lineIndices.push_back(k1);
                lineIndices.push_back(k1 + 1);
            }
        }
    }
}

// ELLIPSOID FITTING

void GrainsAsEllipsoids::updateBBoxOnlyOne(int index) {
    m_ccBBoxOnlyOne.setValidity(false);
    m_ccBBoxOnlyOne.clear();
    if (index < m_stacks.size()) {
        if (m_fitNotOK.count(index) == 0) {
            float maxRadius = m_radii[index].maxCoeff();
            CCVector3 center(m_center[index](0), m_center[index](1),
                             m_center[index](2));
            m_ccBBoxOnlyOne.add(CCVector3(center.x + maxRadius,
                                          center.y + maxRadius,
                                          center.z + maxRadius));
            m_ccBBoxOnlyOne.add(CCVector3(center.x - maxRadius,
                                          center.y - maxRadius,
                                          center.z - maxRadius));
            m_ccBBoxOnlyOne.setValidity(true);
        }
    } else {
        CVLog::Error(
                "[GrainsAsEllipsoids::updateBBox] asking for the bounding of "
                "index " +
                QString::number(index) + " out of range");
    }
}

bool GrainsAsEllipsoids::explicitToImplicit(
        const Eigen::Array3f& center,
        const Eigen::Array3f& radii,
        const Eigen::Matrix3f& rotationMatrix,
        Eigen::ArrayXd& parameters) {
    // INSPIRED BY MATLAB CODE

    // Cast ellipsoid defined with explicit parameters to implicit vector form.
    //
    // Examples:
    //    p = ellipse_ex2im([xc,yc,zc],[xr,yr,zr],eye(3,3));

    // Matlab code => Copyright 2011 Levente Hunyadi

    float xrr = 1 / radii(0);
    float yrr = 1 / radii(1);
    float zrr = 1 / radii(2);

    float r11 = rotationMatrix.data()[0];
    float r21 = rotationMatrix.data()[1];
    float r31 = rotationMatrix.data()[2];
    float r12 = rotationMatrix.data()[3];
    float r22 = rotationMatrix.data()[4];
    float r32 = rotationMatrix.data()[5];
    float r13 = rotationMatrix.data()[6];
    float r23 = rotationMatrix.data()[7];
    float r33 = rotationMatrix.data()[8];

    float xc = center(0);
    float yc = center(1);
    float zc = center(2);

    // terms collected from symbolic expression

    parameters << pow(r11, 2) * pow(xrr, 2) + pow(r21, 2) * pow(yrr, 2) +
                          pow(r31, 2) * pow(zrr, 2),
            pow(r12, 2) * pow(xrr, 2) + pow(r22, 2) * pow(yrr, 2) +
                    pow(r32, 2) * pow(zrr, 2),
            pow(r13, 2) * pow(xrr, 2) + pow(r23, 2) * pow(yrr, 2) +
                    pow(r33, 2) * pow(zrr, 2),
            2 * r11 * r12 * pow(xrr, 2) + 2 * r21 * r22 * pow(yrr, 2) +
                    2 * r31 * r32 * pow(zrr, 2),
            2 * r11 * r13 * pow(xrr, 2) + 2 * r21 * r23 * pow(yrr, 2) +
                    2 * r31 * r33 * pow(zrr, 2),
            2 * r12 * r13 * pow(xrr, 2) + 2 * r22 * r23 * pow(yrr, 2) +
                    2 * r32 * r33 * pow(zrr, 2),
            (-2) * (pow(r11, 2) * xc * pow(xrr, 2) +
                    pow(r21, 2) * xc * pow(yrr, 2) +
                    pow(r31, 2) * xc * pow(zrr, 2) +
                    r11 * r12 * pow(xrr, 2) * yc +
                    r11 * r13 * pow(xrr, 2) * zc +
                    r21 * r22 * yc * pow(yrr, 2) +
                    r21 * r23 * pow(yrr, 2) * zc +
                    r31 * r32 * yc * pow(zrr, 2) +
                    r31 * r33 * zc * pow(zrr, 2)),
            (-2) * (pow(r12, 2) * pow(xrr, 2) * yc +
                    pow(r22, 2) * yc * pow(yrr, 2) +
                    pow(r32, 2) * yc * pow(zrr, 2) +
                    r11 * r12 * xc * pow(xrr, 2) +
                    r21 * r22 * xc * pow(yrr, 2) +
                    r12 * r13 * pow(xrr, 2) * zc +
                    r31 * r32 * xc * pow(zrr, 2) +
                    r22 * r23 * pow(yrr, 2) * zc +
                    r32 * r33 * zc * pow(zrr, 2)),
            (-2) * (pow(r13, 2) * pow(xrr, 2) * zc +
                    pow(r23, 2) * pow(yrr, 2) * zc +
                    pow(r33, 2) * zc * pow(zrr, 2) +
                    r11 * r13 * xc * pow(xrr, 2) +
                    r12 * r13 * pow(xrr, 2) * yc +
                    r21 * r23 * xc * pow(yrr, 2) +
                    r22 * r23 * yc * pow(yrr, 2) +
                    r31 * r33 * xc * pow(zrr, 2) +
                    r32 * r33 * yc * pow(zrr, 2)),
            pow(r11, 2) * pow(xc, 2) * pow(xrr, 2) +
                    2 * r11 * r12 * xc * pow(xrr, 2) * yc +
                    2 * r11 * r13 * xc * pow(xrr, 2) * zc +
                    pow(r12, 2) * pow(xrr, 2) * pow(yc, 2) +
                    2 * r12 * r13 * pow(xrr, 2) * yc * zc +
                    pow(r13, 2) * pow(xrr, 2) * pow(zc, 2) +
                    pow(r21, 2) * pow(xc, 2) * pow(yrr, 2) +
                    2 * r21 * r22 * xc * yc * pow(yrr, 2) +
                    2 * r21 * r23 * xc * pow(yrr, 2) * zc +
                    pow(r22, 2) * pow(yc, 2) * pow(yrr, 2) +
                    2 * r22 * r23 * yc * pow(yrr, 2) * zc +
                    pow(r23, 2) * pow(yrr, 2) * pow(zc, 2) +
                    pow(r31, 2) * pow(xc, 2) * pow(zrr, 2) +
                    2 * r31 * r32 * xc * yc * pow(zrr, 2) +
                    2 * r31 * r33 * xc * zc * pow(zrr, 2) +
                    pow(r32, 2) * pow(yc, 2) * pow(zrr, 2) +
                    2 * r32 * r33 * yc * zc * pow(zrr, 2) +
                    pow(r33, 2) * pow(zc, 2) * pow(zrr, 2) - 1;

    return true;
}

bool GrainsAsEllipsoids::implicitToExplicit(const Eigen::ArrayXd& parameters,
                                            Eigen::Array3f& center,
                                            Eigen::Array3f& radii,
                                            Eigen::Matrix3f& rotationMatrix) {
    // INSPIRED BY MATLAB CODE

    // Cast ellipsoid defined with implicit parameter vector to explicit form.
    // The implicit equation of a general ellipse is
    // F(x,y,z) = Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz - 1
    // = 0
    //
    // Input arguments:
    // v:
    //    the 10 parameters describing the ellipsoid algebraically
    // Output arguments:
    // center:
    //    ellispoid center coordinates [cx; cy; cz]
    // ax:
    //    ellipsoid semi-axes (radii) [a; b; c]
    // quat: NOT IN THIS CPP VERSION, ONLY MATLAB VERSION
    //    ellipsoid rotation in quaternion representation
    // R:
    //    ellipsoid rotation (radii directions as rows of the 3x3 matrix)
    //
    // See also: ellipse_im2ex

    // Matlab code => Copyright 2011 Levente Hunyadi

    Eigen::ArrayXd p = parameters;

    p(3) = 0.5 * p(3);
    p(4) = 0.5 * p(4);
    p(5) = 0.5 * p(5);
    p(6) = 0.5 * p(6);
    p(7) = 0.5 * p(7);
    p(8) = 0.5 * p(8);

    Eigen::MatrixXd q(4, 4);

    q << p(0), p(3), p(4), p(6), p(3), p(1), p(5), p(7), p(4), p(5), p(2), p(8),
            p(6), p(7), p(8), p(9);

    center = q.block(0, 0, 3, 3)
                     .colPivHouseholderQr()
                     .solve(-p(Eigen::seq(6, 8)).matrix())
                     .cast<float>();

    Eigen::MatrixXd t(4, 4);
    t = Eigen::MatrixXd::Identity(4, 4);
    t(3, 0) = center(0);
    t(3, 1) = center(1);
    t(3, 2) = center(2);

    Eigen::MatrixXd s(4, 4);
    s = t * q * t.transpose();

    // check for positive definiteness
    Eigen::LLT<Eigen::MatrixXd> lltOfA(
            (-s(3, 3) * s.block(0, 0, 3, 3).array()));
    if (lltOfA.info() != Eigen::Success) {
        return false;
    }

    Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(s.block(0, 0, 3, 3));
    if (eigensolver.info() != Eigen::Success) {
        return false;
    }

    radii = (-s(3, 3) / eigensolver.eigenvalues().array().real())
                    .sqrt()
                    .cast<float>();
    rotationMatrix =
            eigensolver.eigenvectors().transpose().real().cast<float>();

    return true;
}

bool GrainsAsEllipsoids::directFit(const Eigen::ArrayX3d& xyz,
                                   Eigen::ArrayXd& parameters) {
    // INSPIRED BY MATLAB CODE

    // Direct least squares fitting of ellipsoids under the constraint 4J - I^2
    // > 0. The constraint confines the class of ellipsoids to fit to those
    // whose smallest radius is at least half of the largest radius.
    //
    // Input arguments:
    // x,y,z;
    //    x, y and z coodinates of 3D points
    //
    // Output arguments:
    // p:
    //    a 10-parameter vector of the algebraic ellipsoid fit
    //
    // References:
    // Qingde Li and John G. Griffiths, "Least Squares Ellipsoid Specific
    // Fitting",
    //    Proceedings of the Geometric Modeling and Processing, 2004.

    // Matlab code reference => Copyright 2011 Levente Hunyadi

    Eigen::MatrixXd d(xyz.rows(), 10);

    d << xyz(Eigen::all, 0).pow(2).matrix(), xyz(Eigen::all, 1).pow(2).matrix(),
            xyz(Eigen::all, 2).pow(2).matrix(),
            (2 * xyz(Eigen::all, 1) * xyz(Eigen::all, 2)).matrix(),
            (2 * xyz(Eigen::all, 0) * xyz(Eigen::all, 2)).matrix(),
            (2 * xyz(Eigen::all, 0) * xyz(Eigen::all, 1)).matrix(),
            (2 * xyz(Eigen::all, 0)).matrix(),
            (2 * xyz(Eigen::all, 1)).matrix(),
            (2 * xyz(Eigen::all, 2)).matrix(),
            Eigen::MatrixXd::Ones(xyz.rows(), 1);

    Eigen::MatrixXd s = d.transpose() * d;

    int k = 4;
    Eigen::Matrix3d c1;
    Eigen::Matrix3d c2;
    Eigen::MatrixXd c;
    c = Eigen::MatrixXd::Zero(10, 10);
    c1 << 0, k, k, k, 0, k, k, k, 0;
    c1 = c1.array() / 2 - 1;
    c2 = -k * Eigen::Matrix3d::Identity();
    c.block(0, 0, 3, 3) = c1;
    c.block(3, 3, 3, 3) = c2;

    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> eigensolver(s, c);
    if (eigensolver.info() != Eigen::Success) {
        return false;
    }

    Eigen::ArrayXd eigenValues(10);
    Eigen::VectorXd eigenValuesAsAMatrix(10);
    eigenValues = eigensolver.eigenvalues().real();
    eigenValuesAsAMatrix = eigensolver.eigenvalues().real().matrix();
    Xb condition = (eigenValues > 0) && (!eigenValues.isInf());

    int flt = condition.count();
    //	std::cout << "flt " << flt << std::endl;
    Eigen::ArrayXd finiteValues(flt);
    finiteValues = Eigen::ArrayXd::Zero(flt);
    int finiteValuesCounter = 0;
    for (int k = 0; k < eigenValues.size(); k++) {
        if (condition(k)) {
            finiteValues(finiteValuesCounter++) = eigenValues(k);
        }
    }

    double eigenValue;
    Eigen::MatrixXd v;
    switch (flt) {
        case 1:  // regular case
            eigenValue =
                    finiteValues(0);  // there is only one positive finite value
            for (k = 0; k < 10; k++) {
                if (eigenValues(k) == eigenValue) {
                    v = eigensolver.eigenvectors()(Eigen::all, k).real();
                    break;
                }
            }
            break;
        case 0:  // degenerate case
            // # single positive eigenvalue becomes near-zero negative
            // eigenvalue due to round-off error
            eigenValue = eigenValues.abs().minCoeff();
            for (k = 0; k < 10; k++) {
                if (abs(eigenValues(k)) == eigenValue) {
                    v = eigensolver.eigenvectors()(Eigen::all, k).real();
                    break;
                }
            }
            break;
        default:  // degenerate case
            // several positive eigenvalues appear
            eigenValue = finiteValues.abs().minCoeff();
            for (k = 0; k < 10; k++) {
                if (eigenValues(k) == eigenValue) {
                    v = eigensolver.eigenvectors()(Eigen::all, k).real();
                    break;
                }
            }
            break;
    }

    parameters.resize(10);

    parameters << v(0), v(1), v(2), 2 * v(5), 2 * v(4), 2 * v(3), 2 * v(6),
            2 * v(7), 2 * v(8), v(9);

    return true;
}

bool GrainsAsEllipsoids::fitEllipsoidToGrain(const int grainIndex,
                                             Eigen::Array3f& center,
                                             Eigen::Array3f& radii,
                                             Eigen::Matrix3f& rotationMatrix,
                                             const Method& method) {
    // Shift point cloud to have only positive coordinates
    // (problem with quadfit if the point cloud is far from the coordinates of
    // the origin (0,0,0))

    bool ret = true;

    // extract the point cloud related to the current index
    cloudViewer::ReferenceCloud referenceCloud(m_cloud);
    for (int index : m_stacks[grainIndex]) {
        referenceCloud.addPointIndex(index);
    }

    ccPointCloud* grainCloud = m_cloud->partialClone(&referenceCloud);
    Eigen::Map<const Eigen::MatrixX3f, Eigen::Unaligned, Eigen::Stride<1, 3>>
            grainPoints(static_cast<const float*>(grainCloud->getPoint(0)->u),
                        grainCloud->size(), 3);

    CCVector3 bbMin;
    CCVector3 bbMax;
    grainCloud->getBoundingBox(bbMin, bbMax);
    CCVector3 bb(bbMax - bbMin);
    Eigen::Vector3d scales(bb.x, bb.y, bb.z);
    double scale = 1 / scales.maxCoeff();
    Eigen::RowVector3d means = grainPoints.cast<double>().colwise().mean();

    Eigen::ArrayXd p(10);

    switch (method) {
        case DIRECT:
            // Direct least squares fitting of ellipsoids under the constraint
            // 4J - I**2 > 0. The constraint confines the class of ellipsoids to
            // fit to those whose smallest radius is at least half of the
            // largest radius.

            if (!directFit(
                        scale * (grainPoints.cast<double>().rowwise() - means),
                        p))  // Ellipsoid fit
            {
                return false;
            }

            if (!implicitToExplicit(
                        p, center, radii,
                        rotationMatrix))  // Get the explicit parameters
            {
                return false;
            }

            break;
        default:
            break;
    }

    // Rescale the explicit parameters (the rotation matrix is unchanged by the
    // scaling)
    center = center / scale + Eigen::Array3f(means.cast<float>());
    radii = radii / scale;

    // re-order the radii
    std::vector<float> sortedRadii{radii(0), radii(1), radii(2)};
    std::sort(sortedRadii.begin(), sortedRadii.end());
    Eigen::Array3f updatedRadii = {
            sortedRadii[0], sortedRadii[1],
            sortedRadii[2]};  // from the smallest to the largest
    Eigen::Matrix3f updatedRotationMatrix;
    for (int k = 0; k < 3; k++) {
        float radius = updatedRadii(k);
        int col = 0;
        for (int idx = 0; idx < 3; idx++) {
            if (radii[idx] == radius) {
                break;
            }
            col++;
        }
        updatedRotationMatrix(k, 0) = rotationMatrix(col, 0);
        updatedRotationMatrix(k, 1) = rotationMatrix(col, 1);
        updatedRotationMatrix(k, 2) = rotationMatrix(col, 2);
    }

    radii = updatedRadii;
    rotationMatrix = updatedRotationMatrix;

    ret = explicitToImplicit(center, radii, rotationMatrix, p);

    return ret;
}

// DRAW

void GrainsAsEllipsoids::updateMeshAndLineSet() {
    // Validate basic state
    if (m_center.empty() || m_radii.empty() || m_rotationMatrix.empty() ||
        m_grainColors.empty()) {
        CVLog::Warning(
                "[GrainsAsEllipsoids::updateMeshAndLineSet] Empty data arrays, "
                "cannot update mesh");
        return;
    }

    // Ensure sphere template is initialized
    if (vertices.empty() || indices.empty() || lineIndices.empty()) {
        initSphereVertices();
        initSphereIndexes();
        // Verify initialization succeeded
        if (vertices.empty() || indices.empty() || lineIndices.empty()) {
            CVLog::Error(
                    "[GrainsAsEllipsoids::updateMeshAndLineSet] Failed to "
                    "initialize sphere template");
            return;
        }
    }

    ecvMainAppInterface::ccHObjectContext objContext;
    bool hasContext = false;
    // Temporarily remove from DB Tree if requested and possible
    if (m_app && getParent()) {
        objContext = m_app->removeObjectTemporarilyFromDBTree(this);
        hasContext = true;
    }

    // -------------------------------------------------------------------------
    // 1. Create objects if they don't exist
    // -------------------------------------------------------------------------

    // Check if resize is needed (e.g. if data changed significantly)
    if (m_meshes.size() != m_center.size()) {
        clearGeneratedObjects();
        m_meshes.resize(m_center.size(), nullptr);
        m_lineSets.resize(m_center.size(), nullptr);
        m_pointsClouds.resize(m_center.size(), nullptr);
    }

    // Iterate through all grains to create or update
    for (int idx = 0; idx < static_cast<int>(m_center.size()); idx++) {
        // Only create if missing (nullptr)
        if (!m_meshes[idx]) {
            // Validate index bounds and data validity
            if (m_fitNotOK.count(idx)) {
                continue;  // Skip ellipsoids that failed to fit
            }

            // Check if rotation matrix contains NaN
            bool hasNaN = false;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (std::isnan(m_rotationMatrix[idx](i, j))) {
                        hasNaN = true;
                        break;
                    }
                }
                if (hasNaN) break;
            }
            if (hasNaN) {
                CVLog::Warning(
                        "[GrainsAsEllipsoids::updateMeshAndLineSet] Rotation "
                        "matrix "
                        "contains NaN at index: " +
                        QString::number(idx));
                continue;
            }

            // Validate radii are positive
            if (m_radii[idx](0) <= 0 || m_radii[idx](1) <= 0 ||
                m_radii[idx](2) <= 0) {
                CVLog::Warning(
                        "[GrainsAsEllipsoids::updateMeshAndLineSet] Invalid "
                        "radii "
                        "at index: " +
                        QString::number(idx));
                continue;
            }

            // --- Prepare Ellipsoid Vertices ---
            Eigen::Matrix3f rotation(m_rotationMatrix[idx].transpose());
            Eigen::Array3f center = m_center[idx];
            Eigen::Array3f radii = m_radii[idx];

            std::vector<Eigen::Vector3d> ellipsoidVertices;
            ellipsoidVertices.reserve(vertices.size() / 3);

            for (size_t i = 0; i < vertices.size(); i += 3) {
                Eigen::Vector3f sphereVertex(vertices[i], vertices[i + 1],
                                             vertices[i + 2]);
                Eigen::Vector3f scaledVertex(sphereVertex.x() * radii(0),
                                             sphereVertex.y() * radii(1),
                                             sphereVertex.z() * radii(2));
                Eigen::Vector3f rotatedVertex = rotation * scaledVertex;
                Eigen::Vector3d finalVertex(rotatedVertex.x() + center(0),
                                            rotatedVertex.y() + center(1),
                                            rotatedVertex.z() + center(2));
                ellipsoidVertices.push_back(finalVertex);
            }

            // --- Create Mesh ---
            if (!indices.empty()) {
                ccPointCloud* vertexCloud = new ccPointCloud("vertices");
                if (vertexCloud->resize(
                            static_cast<unsigned>(ellipsoidVertices.size()))) {
                    vertexCloud->setEigenPoints(ellipsoidVertices);

                    ccMesh* mesh = new ccMesh(vertexCloud);
                    mesh->setName(QString("Ellipsoid_%1_Mesh").arg(idx));

                    for (size_t i = 0; i < indices.size(); i += 3) {
                        mesh->addTriangle(
                                static_cast<unsigned>(indices[i]),
                                static_cast<unsigned>(indices[i + 1]),
                                static_cast<unsigned>(indices[i + 2]));
                    }

                    CCVector3f colorVec = m_grainColors[idx];
                    ecvColor::Rgbaf materialColor(
                            colorVec.x, colorVec.y, colorVec.z,
                            static_cast<float>(m_transparency));

                    ccMaterial::Shared material(new ccMaterial(
                            QString("Ellipsoid_%1_Material").arg(idx)));
                    material->setDiffuse(materialColor);
                    material->setIllum(0);

                    ccMaterialSet* materialSet = new ccMaterialSet(
                            QString("Ellipsoid_%1_MaterialSet").arg(idx));
                    materialSet->addMaterial(material);
                    mesh->setMaterialSet(materialSet);
                    mesh->showMaterials(true);

                    vertexCloud->setEnabled(false);
                    vertexCloud->setLocked(false);
                    mesh->addChild(vertexCloud);

                    m_meshes[idx] = mesh;
                    addChild(mesh);
                    // Visibility will be set in the update loop below
                } else {
                    delete vertexCloud;
                }
            }

            // --- Create LineSet ---
            if (!lineIndices.empty() && !ellipsoidVertices.empty()) {
                // Check line indices validity (simplified check)
                std::vector<Eigen::Vector2i> lines;
                lines.reserve(lineIndices.size() / 2);
                for (size_t i = 0; i < lineIndices.size(); i += 2) {
                    if (i + 1 < lineIndices.size()) {
                        lines.push_back(Eigen::Vector2i(lineIndices[i],
                                                        lineIndices[i + 1]));
                    }
                }

                cloudViewer::geometry::LineSet* lineSet =
                        new cloudViewer::geometry::LineSet(
                                ellipsoidVertices, lines,
                                QString("Ellipsoid_%1_Lines")
                                        .arg(idx)
                                        .toLatin1()
                                        .data());

                CCVector3f color = m_grainColors[idx];
                Eigen::Vector3d lineColor(color.x * 0.8, color.y * 0.8,
                                          color.z * 0.8);
                lineSet->PaintUniformColor(lineColor);

                m_lineSets[idx] = lineSet;
                addChild(lineSet);
            }

            // --- Create PointCloud ---
            if (m_cloud && idx < static_cast<int>(m_stacks.size())) {
                const std::vector<int>& stack = m_stacks[idx];
                if (!stack.empty()) {
                    cloudViewer::ReferenceCloud referenceCloud(m_cloud);
                    for (int index : stack) {
                        referenceCloud.addPointIndex(index);
                    }

                    ccPointCloud* pc = m_cloud->partialClone(&referenceCloud);
                    if (pc) {
                        pc->setName(QString("Ellipsoid_%1_Points").arg(idx));

                        if (pc->resizeTheRGBTable()) {
                            CCVector3f color = m_grainColors[idx];
                            ecvColor::Rgb rgbColor(
                                    static_cast<unsigned char>(
                                            color.x * ecvColor::MAX * 0.8f),
                                    static_cast<unsigned char>(
                                            color.y * ecvColor::MAX * 0.8f),
                                    static_cast<unsigned char>(
                                            color.z * ecvColor::MAX * 0.8f));
                            for (unsigned int i = 0; i < pc->size(); i++) {
                                pc->setPointColor(i, rgbColor);
                            }
                            pc->showColors(true);
                        }

                        m_pointsClouds[idx] = pc;
                        addChild(pc);
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Update Visibility and Properties
    // -------------------------------------------------------------------------
    for (int idx = 0; idx < static_cast<int>(m_center.size()); idx++) {
        bool isVisible = m_showAll || (idx == m_onlyOne);

        // Meshes
        if (m_meshes.size() > idx && m_meshes[idx]) {
            bool showMesh = isVisible && m_drawSurfaces;
            m_meshes[idx]->setVisible(showMesh);
            m_meshes[idx]->setEnabled(showMesh);

            // Update Transparency if needed
            if (showMesh) {
                ccMaterialSet* matSet = const_cast<ccMaterialSet*>(
                        m_meshes[idx]->getMaterialSet());
                if (matSet && !matSet->empty()) {
                    // CShared is QSharedPointer<const ccMaterial>
                    // We need a non-const Shared pointer to modify it
                    ccMaterial::CShared cMat = matSet->front();

                    // If we need to modify it, we should check if we can cast
                    // away constness (dangerous if truly const) or if we should
                    // clone/replace. Since we created it, we know it's mutable
                    // underlying object.
                    ccMaterial::Shared mat =
                            qSharedPointerConstCast<ccMaterial>(cMat);

                    if (mat) {
                        const ecvColor::Rgbaf& diffuse = mat->getDiffuseFront();
                        if (std::abs(diffuse.a -
                                     static_cast<float>(m_transparency)) >
                            1e-4) {
                            ecvColor::Rgbaf newColor = diffuse;
                            newColor.a = static_cast<float>(m_transparency);
                            mat->setDiffuse(newColor);
                        }
                    }
                }
            }
        }

        // LineSets
        if (m_lineSets.size() > idx && m_lineSets[idx]) {
            bool showLine = isVisible && m_drawLines;
            m_lineSets[idx]->setVisible(showLine);
            m_lineSets[idx]->setEnabled(showLine);
        }

        // PointClouds
        if (m_pointsClouds.size() > idx && m_pointsClouds[idx]) {
            bool showPoints = isVisible && m_drawPoints;
            m_pointsClouds[idx]->setVisible(showPoints);
            m_pointsClouds[idx]->setEnabled(showPoints);

            if (showPoints && m_glPointSize > 0) {
                m_pointsClouds[idx]->setPointSize(
                        static_cast<float>(m_glPointSize));
            }
        }
    }

    // Put back into DB Tree
    if (hasContext && m_app) {
        m_app->putObjectBackIntoDBTree(this, objContext);
    }
}

void GrainsAsEllipsoids::setOnlyOne(int i) {
    m_onlyOne = i;
    updateBBoxOnlyOne(i);
    m_meshNeedsUpdate = true;
    redrawDisplay();
}

void GrainsAsEllipsoids::showOnlyOne(bool state) {
    m_showAll = !state;
    m_ccBBox = m_ccBBoxOnlyOne;
    m_meshNeedsUpdate = true;
    redrawDisplay();
}

void GrainsAsEllipsoids::showAll(bool state) {
    m_showAll = state;
    m_ccBBox = m_ccBBoxAll;
    m_meshNeedsUpdate = true;
    redrawDisplay();
}

void GrainsAsEllipsoids::draw(CC_DRAW_CONTEXT& context) {
    if (m_radii.empty())  // nothing to draw, probably due to a bad
                          // initialization
        return;

    // Update mesh and lineset representations if needed
    if (m_meshNeedsUpdate) {
        updateMeshAndLineSet();
        m_meshNeedsUpdate = false;
    }

    // Call parent draw method to handle children drawing and other standard
    // behavior.
    //
    // ccHObject::draw() will:
    // 1. Call drawMeOnly(context) for this object (empty by default for
    //    ccCustomHObject)
    // 2. Iterate through all children in m_children and call
    // child->draw(context)
    //    for each child (see ecvHObject.cpp:1500-1502)
    //
    // Our child objects (meshes, linesets, point clouds) added via addChild()
    // will have their draw() method called, which in turn calls their
    // drawMeOnly() method, which calls ecvDisplayTools::Draw(context, this)
    // to perform the actual rendering.
    //
    // This ensures all child objects are rendered automatically.
    ccHObject::draw(context);
}

ccBBox GrainsAsEllipsoids::getOwnBB(bool withGLFeatures) { return m_ccBBox; }

/// template <class Type, int N, class ComponentType> static
///          <Eigen::Array3f, 1, float>
bool genericArrayToFile(const std::vector<Eigen::Array3f>& data, QFile& out) {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));

    // removed to allow saving empty clouds
    // if (data.empty())
    //{
    //	return ccSerializableObject::MemoryError();
    // }

    int N = 1;

    // component count (dataVersion>=20)
    ::uint8_t componentCount = static_cast<::uint8_t>(N);
    if (out.write((const char*)&componentCount, 1) < 0)
        return ccSerializableObject::WriteError();

    // element count = array size (dataVersion>=20)
    ::uint32_t elementCount = static_cast<::uint32_t>(data.size());
    if (out.write((const char*)&elementCount, 4) < 0)
        return ccSerializableObject::WriteError();

    // array data (dataVersion>=20)
    {
        // DGM: do it by chunks, in case it's too big to be processed by the
        // system
        const char* _data = (const char*)data.data();
        qint64 byteCount = static_cast<qint64>(elementCount);
        byteCount *= sizeof(Eigen::Array3f);
        while (byteCount != 0) {
            static const qint64 s_maxByteSaveCount =
                    (1 << 26);  // 64 Mb each time
            qint64 saveCount = std::min(byteCount, s_maxByteSaveCount);
            if (out.write(_data, saveCount) < 0)
                return ccSerializableObject::WriteError();
            _data += saveCount;
            byteCount -= saveCount;
        }
    }
    return true;
}

bool readArrayHeader(QFile& in,
                     short dataVersion,
                     ::uint8_t& componentCount,
                     ::uint32_t& elementCount) {
    assert(in.isOpen() && (in.openMode() & QIODevice::ReadOnly));

    if (dataVersion < 20) return ccSerializableObject::CorruptError();

    // component count (dataVersion>=20)
    if (in.read((char*)&componentCount, 1) < 0)
        return ccSerializableObject::ReadError();

    // element count = array size (dataVersion>=20)
    if (in.read((char*)&elementCount, 4) < 0)
        return ccSerializableObject::ReadError();

    return true;
}

template <typename T>
bool stdVectorToFile(QString name, std::vector<T> vector) {
    std::ofstream file(name.toLatin1());
    int elementSize = vector[0].size();
    for (int i = 0; i < vector.size(); i++) {
        for (int j = 0; j < elementSize; j++) {
            file << vector[i][j] << ", ";
        }
        file << std::endl;
    }
    return true;
}

bool rotationMatrixToFile(QString name,
                          std::vector<Eigen::Matrix3f> rotationMatrix) {
    std::ofstream file(name.toLatin1());
    int elementSize = rotationMatrix[0].size();
    for (int i = 0; i < rotationMatrix.size(); i++) {
        for (int j = 0; j < elementSize; j++) {
            file << rotationMatrix[i](j) << ", ";
        }
        file << std::endl;
    }
    return true;
}

bool GrainsAsEllipsoids::toFile_MeOnly(QFile& out, short dataVersion) const {
    CVLog::Print("[G3Point] write GrainsAsEllipsoids object in .bin");

    if (!ccHObject::toFile_MeOnly(out, dataVersion)) {
        return false;
    }

    if (!ccSerializationHelper::GenericArrayToFile<Eigen::Array3f, 1,
                                                   Eigen::Array3f>(m_center,
                                                                   out))
        return WriteError();

    if (!ccSerializationHelper::GenericArrayToFile<Eigen::Array3f, 1,
                                                   Eigen::Array3f>(m_radii,
                                                                   out))
        return WriteError();

    if (!ccSerializationHelper::GenericArrayToFile<Eigen::Matrix3f, 1,
                                                   Eigen::Matrix3f>(
                m_rotationMatrix, out))
        return WriteError();

    if (!ccSerializationHelper::GenericArrayToFile<CCVector3f, 1, CCVector3f>(
                m_grainColors, out))
        return WriteError();

    return true;
}

short GrainsAsEllipsoids::minimumFileVersion_MeOnly() const {
    return ccHObject::minimumFileVersion_MeOnly();
}

bool GrainsAsEllipsoids::fromFile_MeOnly(QFile& in,
                                         short dataVersion,
                                         int flags,
                                         LoadedIDMap& oldToNewIDMap) {
    CVLog::Print("[G3Point] read GrainsAsEllipsoids object from .bin");

    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    if (!ccSerializationHelper::GenericArrayFromFile<Eigen::Array3f, 1,
                                                     Eigen::Array3f>(
                m_center, in, dataVersion)) {
        CVLog::Warning("[G3Point] error reading m_center");
        return ReadError();
    }

    if (!ccSerializationHelper::GenericArrayFromFile<Eigen::Array3f, 1,
                                                     Eigen::Array3f>(
                m_radii, in, dataVersion)) {
        CVLog::Warning("[G3Point] error reading m_radii");
        return ReadError();
    }

    if (!ccSerializationHelper::GenericArrayFromFile<Eigen::Matrix3f, 1,
                                                     Eigen::Matrix3f>(
                m_rotationMatrix, in, dataVersion)) {
        CVLog::Warning("[G3Point] error reading m_rorationMatrix");
        return ReadError();
    }

    if (!ccSerializationHelper::GenericArrayFromFile<CCVector3f, 1, CCVector3f>(
                m_grainColors, in, dataVersion)) {
        CVLog::Warning("[G3Point] error reading m_rorationMatrix");
        return ReadError();
    }

    lockVisibility(false);
    setVisible(true);

    m_ccBBoxAll.setValidity(false);
    m_ccBBoxAll.clear();

    for (int idx = 0; idx < m_center.size(); idx++) {
        float maxRadius = m_radii[idx].maxCoeff();
        CCVector3 center(m_center[idx](0), m_center[idx](1), m_center[idx](2));
        if (m_radii[idx].x() !=
            -1)  // all radii are equal to zero when the fit was not successful
        {
            m_fitNotOK.insert(idx);
            continue;
        }
        m_ccBBoxAll.add(CCVector3(center.x + maxRadius, center.y + maxRadius,
                                  center.z + maxRadius));
        m_ccBBoxAll.add(CCVector3(center.x - maxRadius, center.y - maxRadius,
                                  center.z - maxRadius));
    }

    m_ccBBoxAll.setValidity(true);

    m_ccBBox = m_ccBBoxAll;
    m_meshNeedsUpdate = true;
    redrawDisplay();

    return true;
}
