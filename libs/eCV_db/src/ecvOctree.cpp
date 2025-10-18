// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvOctree.h"

#include "ecvBox.h"
#include "ecvCameraSensor.h"
#include "ecvDisplayTools.h"
#include "ecvNormalVectors.h"
#include "ecvPointCloud.h"
#include "ecvProgressDialog.h"
#include "ecvScalarField.h"

// CV_CORE_LIB
#include <Neighbourhood.h>
#include <RayAndBox.h>
#include <ScalarFieldTools.h>

ccOctree::ccOctree(ccGenericPointCloud* aCloud)
    : cloudViewer::DgmOctree(aCloud),
      m_theAssociatedCloudAsGPC(aCloud),
      m_displayedLevel(1),
      m_displayMode(WIRE),
      m_visible(true),
      m_frustumIntersector(nullptr) {}

ccOctree::~ccOctree() {
    if (m_frustumIntersector) {
        delete m_frustumIntersector;
        m_frustumIntersector = nullptr;
    }
}

bool ccOctree::intersectWithFrustum(ccCameraSensor* sensor,
                                    std::vector<unsigned>& inCameraFrustum) {
    if (!sensor) return false;

    // initialization
    float globalPlaneCoefficients[6][4];
    CCVector3 globalCorners[8];
    CCVector3 globalEdges[6];
    CCVector3 globalCenter;
    sensor->computeGlobalPlaneCoefficients(
            globalPlaneCoefficients, globalCorners, globalEdges, globalCenter);

    if (!m_frustumIntersector) {
        m_frustumIntersector = new ccOctreeFrustumIntersector();
        if (!m_frustumIntersector->build(this)) {
            CVLog::Warning(
                    "[ccOctree::intersectWithFrustum] Not enough memory!");
            return false;
        }
    }

    // get points of cells in frustum
    std::vector<std::pair<unsigned, CCVector3>> pointsToTest;
    m_frustumIntersector->computeFrustumIntersectionWithOctree(
            pointsToTest, inCameraFrustum, globalPlaneCoefficients,
            globalCorners, globalEdges, globalCenter);

    // project points
    for (size_t i = 0; i < pointsToTest.size(); i++) {
        if (sensor->isGlobalCoordInFrustum(pointsToTest[i].second /*, false*/))
            inCameraFrustum.push_back(pointsToTest[i].first);
    }

    return true;
}

void ccOctree::setDisplayedLevel(int level) {
    if (level != m_displayedLevel) {
        m_displayedLevel = level;
        // m_glListIsDeprecated = true;
    }
}

void ccOctree::setDisplayMode(DisplayMode mode) {
    if (m_displayMode != mode) {
        m_displayMode = mode;
        // m_glListIsDeprecated = true;
    }
}

void ccOctree::clear() {
    // warn the others that the octree organization is going to change
    emit updated();

    DgmOctree::clear();
}

ccBBox ccOctree::getSquareBB() const { return ccBBox(m_dimMin, m_dimMax); }

ccBBox ccOctree::getPointsBB() const {
    return ccBBox(m_pointsMin, m_pointsMax);
}

void ccOctree::multiplyBoundingBox(const PointCoordinateType multFactor) {
    m_dimMin *= multFactor;
    m_dimMax *= multFactor;
    m_pointsMin *= multFactor;
    m_pointsMax *= multFactor;

    for (int i = 0; i <= MAX_OCTREE_LEVEL; ++i) m_cellSize[i] *= multFactor;
}

void ccOctree::translateBoundingBox(const CCVector3& T) {
    m_dimMin += T;
    m_dimMax += T;
    m_pointsMin += T;
    m_pointsMax += T;
}

/*** RENDERING METHODS ***/

void ccOctree::draw(CC_DRAW_CONTEXT& context) {
    if (!m_theAssociatedCloudAsGPC || m_thePointsAndTheirCellCodes.empty()) {
        return;
    }

    if (!ecvDisplayTools::GetCurrentScreen()) return;

    if (m_displayMode == WIRE) {
        // this display mode is too heavy to be stored as a GL list
        //(therefore we always render it dynamically)

        void* additionalParameters[] = {
                reinterpret_cast<void*>(m_frustumIntersector),
                reinterpret_cast<void*>(&m_visible)};
        executeFunctionForAllCellsAtLevel(m_displayedLevel, &DrawCellAsABox,
                                          additionalParameters);
    } else {
        glDrawParams glParams;
        m_theAssociatedCloudAsGPC->getDrawingParameters(glParams);
    }
}

bool ccOctree::DrawCellAsABox(
        const cloudViewer::DgmOctree::octreeCell& cell,
        void** additionalParameters,
        cloudViewer::NormalizedProgress* nProgress /*=0*/) {
    ccOctreeFrustumIntersector* ofi =
            static_cast<ccOctreeFrustumIntersector*>(additionalParameters[0]);
    bool visible = *(static_cast<bool*>(additionalParameters[1]));
    CCVector3 bbMin, bbMax;
    cell.parentOctree->computeCellLimits(cell.truncatedCode, cell.level, bbMin,
                                         bbMax, true);

    ccOctreeFrustumIntersector::OctreeCellVisibility vis =
            ccOctreeFrustumIntersector::CELL_OUTSIDE_FRUSTUM;
    if (ofi) vis = ofi->positionFromFrustum(cell.truncatedCode, cell.level);

    CC_DRAW_CONTEXT context;
    context.viewID = QString("Octree-") + QString::number(cell.truncatedCode);

    if (visible) {
        // outside
        if (vis == ccOctreeFrustumIntersector::CELL_OUTSIDE_FRUSTUM) {
            context.bbDefaultCol = ecvColor::green;
        } else {
            context.defaultLineWidth = 2;
            // inside
            if (vis == ccOctreeFrustumIntersector::CELL_INSIDE_FRUSTUM)
                context.bbDefaultCol = ecvColor::magenta;
            // intersecting
            else
                context.bbDefaultCol = ecvColor::blue;
        }

        context.meshRenderingMode = MESH_RENDERING_MODE::ECV_WIREFRAME_MODE;
        ccBBox cellBox(bbMin, bbMax);
        ecvDisplayTools::DrawBBox(context, &cellBox);
    } else {
        context.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
        context.removeViewID = context.viewID;
        ecvDisplayTools::RemoveEntities(context);
    }

    return true;
}

bool ccOctree::DrawCellAsAPoint(
        const cloudViewer::DgmOctree::octreeCell& cell,
        void** additionalParameters,
        cloudViewer::NormalizedProgress* nProgress /*=0*/) {
    // variables additionnelles
    glDrawParams* glParams =
            reinterpret_cast<glDrawParams*>(additionalParameters[0]);
    ccGenericPointCloud* cloud =
            reinterpret_cast<ccGenericPointCloud*>(additionalParameters[1]);
    // QOpenGLFunctions_2_1* glFunc	=
    // static_cast<QOpenGLFunctions_2_1*>(additionalParameters[2]);
    // assert(glFunc != nullptr);

    if (glParams->showSF) {
        ScalarType dist = cloudViewer::ScalarFieldTools::computeMeanScalarValue(
                cell.points);
        const ecvColor::Rgb* col = cloud->getScalarValueColor(dist);
        // glFunc->glColor3ubv(col ? col->rgb : ecvColor::lightGrey.rgb);
    } else if (glParams->showColors) {
        ColorCompType col[3];
        ComputeAverageColor(cell.points, cloud, col);
        // glFunc->glColor3ubv(col);
    }

    if (glParams->showNorms) {
        CCVector3 N = ComputeAverageNorm(cell.points, cloud);
        // ccGL::Normal3v(glFunc, N.u);
    }

    const CCVector3* gravityCenter =
            cloudViewer::Neighbourhood(cell.points).getGravityCenter();
    // ccGL::Vertex3v(glFunc, gravityCenter->u);

    return true;
}

bool ccOctree::DrawCellAsAPrimitive(
        const cloudViewer::DgmOctree::octreeCell& cell,
        void** additionalParameters,
        cloudViewer::NormalizedProgress* nProgress /*=0*/) {
    // variables additionnelles
    glDrawParams* glParams =
            reinterpret_cast<glDrawParams*>(additionalParameters[0]);
    ccGenericPointCloud* cloud =
            reinterpret_cast<ccGenericPointCloud*>(additionalParameters[1]);
    ccGenericPrimitive* primitive =
            reinterpret_cast<ccGenericPrimitive*>(additionalParameters[2]);
    CC_DRAW_CONTEXT* context =
            reinterpret_cast<CC_DRAW_CONTEXT*>(additionalParameters[3]);

    // get the set of OpenGL functions (version 2.1)
    // QOpenGLFunctions_2_1* glFunc =
    // context->glFunctions<QOpenGLFunctions_2_1>(); assert(glFunc != nullptr);

    // if (glFunc == nullptr)
    //	return false;

    CCVector3 cellCenter;
    cell.parentOctree->computeCellCenter(cell.truncatedCode, cell.level,
                                         cellCenter, true);

    if (glParams->showSF) {
        ScalarType dist = cloudViewer::ScalarFieldTools::computeMeanScalarValue(
                cell.points);
        const ecvColor::Rgb* rgb = cloud->getScalarValueColor(dist);
        if (rgb) primitive->setColor(*rgb);
    } else if (glParams->showColors) {
        ecvColor::Rgb col;
        ComputeAverageColor(cell.points, cloud, col.rgb);
        primitive->setColor(col);
    }

    if (glParams->showNorms) {
        CCVector3 N = ComputeAverageNorm(cell.points, cloud);
        if (primitive->getTriNormsTable()) {
            // only one normal!
            primitive->getTriNormsTable()->setValue(
                    0, ccNormalVectors::GetNormIndex(N.u));
        }
    }

    // glFunc->glPushMatrix();
    // ccGL::Translate(glFunc, cellCenter.x, cellCenter.y, cellCenter.z);
    primitive->draw(*context);
    // glFunc->glPopMatrix();

    return true;
}

void ccOctree::ComputeAverageColor(cloudViewer::ReferenceCloud* subset,
                                   ccGenericPointCloud* sourceCloud,
                                   ColorCompType meanCol[]) {
    if (!subset || subset->size() == 0 || !sourceCloud) return;

    assert(sourceCloud->hasColors());
    assert(subset->getAssociatedCloud() ==
           static_cast<cloudViewer::GenericIndexedCloud*>(sourceCloud));

    Tuple3Tpl<double> sum(0, 0, 0);

    unsigned n = subset->size();
    for (unsigned i = 0; i < n; ++i) {
        const ecvColor::Rgb& _theColor =
                sourceCloud->getPointColor(subset->getPointGlobalIndex(i));
        sum.x += _theColor.r;
        sum.y += _theColor.g;
        sum.z += _theColor.b;
    }

    meanCol[0] = static_cast<ColorCompType>(sum.x / n);
    meanCol[1] = static_cast<ColorCompType>(sum.y / n);
    meanCol[2] = static_cast<ColorCompType>(sum.z / n);
}

CCVector3 ccOctree::ComputeAverageNorm(cloudViewer::ReferenceCloud* subset,
                                       ccGenericPointCloud* sourceCloud) {
    CCVector3 N(0, 0, 0);

    if (!subset || subset->size() == 0 || !sourceCloud) return N;

    assert(sourceCloud->hasNormals());
    assert(subset->getAssociatedCloud() ==
           static_cast<cloudViewer::GenericIndexedCloud*>(sourceCloud));

    unsigned n = subset->size();
    for (unsigned i = 0; i < n; ++i) {
        const CCVector3& Ni =
                sourceCloud->getPointNormal(subset->getPointGlobalIndex(i));
        N += Ni;
    }

    N.normalize();
    return N;
}

bool ccOctree::pointPicking(const CCVector2d& clickPos,
                            const ccGLCameraParameters& camera,
                            PointDescriptor& output,
                            double pickWidth_pix /*=3.0*/) const {
    output.point = 0;
    output.squareDistd = -1.0;

    if (!m_theAssociatedCloudAsGPC) {
        assert(false);
        return false;
    }

    if (m_thePointsAndTheirCellCodes.empty()) {
        // nothing to do
        return false;
    }

    CCVector3d clickPosd(clickPos.x, clickPos.y, 0.0);
    CCVector3d X(0, 0, 0);
    if (!camera.unproject(clickPosd, X)) {
        return false;
    }

    ccGLMatrix trans;
    bool hasGLTrans =
            m_theAssociatedCloudAsGPC->getAbsoluteGLTransformation(trans);

    // compute 3D picking 'ray'
    CCVector3 rayAxis, rayOrigin;
    {
        CCVector3d clickPosd2(clickPos.x, clickPos.y, 1.0);
        CCVector3d Y(0, 0, 0);
        if (!camera.unproject(clickPosd2, Y)) {
            return false;
        }

        rayAxis = CCVector3::fromArray((Y - X).u);
        rayOrigin = CCVector3::fromArray(X.u);

        if (hasGLTrans) {
            ccGLMatrix iTrans = trans.inverse();
            iTrans.applyRotation(rayAxis);
            iTrans.apply(rayOrigin);
        }

        rayAxis.normalize();  // normalize afterwards as the local
                              // transformation may have a scale != 1
    }

    CCVector3 margin(0, 0, 0);
    double maxFOV_rad = 0;
    if (camera.perspective) {
        maxFOV_rad = 0.002 * pickWidth_pix;  // empirical conversion from pixels
                                             // to FOV angle (in radians)
    } else {
        double maxRadius = pickWidth_pix * camera.pixelSize / 2;
        margin = CCVector3(1, 1, 1) *
                 static_cast<PointCoordinateType>(maxRadius);
    }

    // first test with the total bounding box
    Ray<PointCoordinateType> ray(rayAxis, rayOrigin);
    if (!AABB<PointCoordinateType>(m_dimMin - margin, m_dimMax + margin)
                 .intersects(ray)) {
        // no intersection
        return true;  // DGM: false would mean that an error occurred!
                      // (output.point == 0 means that nothing has been found)
    }

    // no need to go too deep
    const unsigned char maxLevel = findBestLevelForAGivenPopulationPerCell(10);

    // starting level of subdivision
    unsigned char level = 1;
    // binary shift for cell code truncation at current level
    unsigned char currentBitDec = GET_BIT_SHIFT(level);
    // current cell code
    CellCode currentCellCode = INVALID_CELL_CODE;
    CellCode currentCellTruncatedCode = INVALID_CELL_CODE;
    // whether the current cell should be skipped or not
    bool skipThisCell = false;

#ifdef DEBUG_PICKING_MECHANISM
    m_theAssociatedCloud->enableScalarField();
#endif

    // ray with origin expressed in the local coordinate system!
    Ray<PointCoordinateType> rayLocal(rayAxis, rayOrigin - m_dimMin);

    // visibility table (if any)
    const ccGenericPointCloud::VisibilityTableType* visTable =
            m_theAssociatedCloudAsGPC->isVisibilityTableInstantiated()
                    ? &m_theAssociatedCloudAsGPC->getTheVisibilityArray()
                    : 0;

    // scalar field with hidden values (if any)
    ccScalarField* activeSF = 0;
    if (m_theAssociatedCloudAsGPC->sfShown() &&
        m_theAssociatedCloudAsGPC->isA(CV_TYPES::POINT_CLOUD) &&
        !visTable  // if the visibility table is instantiated, we always display
                   // ALL points
    ) {
        ccPointCloud* pc =
                static_cast<ccPointCloud*>(m_theAssociatedCloudAsGPC);
        ccScalarField* sf = pc->getCurrentDisplayedScalarField();
        if (sf && sf->mayHaveHiddenValues() && sf->getColorScale()) {
            // we must take this SF display parameters into account as some
            // points may be hidden!
            activeSF = sf;
        }
    }

    // let's sweep through the octree
    for (cellsContainer::const_iterator it =
                 m_thePointsAndTheirCellCodes.begin();
         it != m_thePointsAndTheirCellCodes.end(); ++it) {
        CellCode truncatedCode = (it->theCode >> currentBitDec);

        // new cell?
        if (truncatedCode != currentCellTruncatedCode) {
            // look for the biggest 'parent' cell that englobes this cell and
            // the previous one (if any)
            while (level > 1) {
                unsigned char bitDec = GET_BIT_SHIFT(level - 1);
                if ((it->theCode >> bitDec) == (currentCellCode >> bitDec)) {
                    // same parent cell, we can stop here
                    break;
                }
                --level;
            }

            currentCellCode = it->theCode;

            // now try to go deeper with the new cell
            while (level < maxLevel) {
                Tuple3i cellPos;
                getCellPos(it->theCode, level, cellPos, false);

                // first test with the total bounding box
                PointCoordinateType halfCellSize = getCellSize(level) / 2;
                CCVector3 cellCenter((2 * cellPos.x + 1) * halfCellSize,
                                     (2 * cellPos.y + 1) * halfCellSize,
                                     (2 * cellPos.z + 1) * halfCellSize);

                CCVector3 halfCell =
                        CCVector3(halfCellSize, halfCellSize, halfCellSize);

                if (camera.perspective) {
                    double radialSqDist, sqDistToOrigin;
                    rayLocal.squareDistances(cellCenter, radialSqDist,
                                             sqDistToOrigin);

                    double dx = sqrt(sqDistToOrigin);
                    double dy = std::max<double>(
                            0, sqrt(radialSqDist) - SQRT_3 * halfCellSize);
                    double fov_rad = atan2(dy, dx);

                    skipThisCell = (fov_rad > maxFOV_rad);
                } else {
                    skipThisCell = !AABB<PointCoordinateType>(
                                            cellCenter - halfCell - margin,
                                            cellCenter + halfCell + margin)
                                            .intersects(rayLocal);
                }

                if (skipThisCell)
                    break;
                else
                    ++level;
            }

            currentBitDec = GET_BIT_SHIFT(level);
            currentCellTruncatedCode = (currentCellCode >> currentBitDec);
        }

#ifdef DEBUG_PICKING_MECHANISM
        m_theAssociatedCloud->setPointScalarValue(it->theIndex, level);
#endif

        if (!skipThisCell) {
            // we shouldn't test points that are actually hidden!
            if ((!visTable || visTable->at(it->theIndex) == POINT_VISIBLE) &&
                (!activeSF ||
                 activeSF->getColor(activeSF->getValue(it->theIndex)))) {
                // test the point
                const CCVector3* P =
                        m_theAssociatedCloud->getPoint(it->theIndex);
                CCVector3 Q = *P;
                if (hasGLTrans) {
                    trans.apply(Q);
                }

                CCVector3d Q2D;
                camera.project(Q, Q2D);

                if (fabs(Q2D.x - clickPos.x) <= pickWidth_pix &&
                    fabs(Q2D.y - clickPos.y) <= pickWidth_pix) {
                    double squareDist =
                            CCVector3d(X.x - Q.x, X.y - Q.y, X.z - Q.z)
                                    .norm2d();
                    if (!output.point || squareDist < output.squareDistd) {
                        output.point = P;
                        output.pointIndex = it->theIndex;
                        output.squareDistd = squareDist;
                    }
                }
            }
        }
    }

    return true;
}

PointCoordinateType ccOctree::GuessNaiveRadius(ccGenericPointCloud* cloud) {
    if (!cloud) {
        assert(false);
        return 0;
    }

    PointCoordinateType largestDim = cloud->getOwnBB().getMaxBoxDim();

    return largestDim /
           std::min<unsigned>(100, std::max<unsigned>(1, cloud->size() / 100));
}

PointCoordinateType ccOctree::GuessBestRadiusAutoComputeOctree(
        ccGenericPointCloud* cloud,
        const BestRadiusParams& params,
        QWidget* parentWidget /*=nullptr*/) {
    if (!cloud) {
        assert(false);
        return 0;
    }

    if (!cloud->getOctree()) {
        ecvProgressDialog pDlg(true, parentWidget);
        if (!cloud->computeOctree(&pDlg)) {
            CVLog::Error(tr("Could not compute octree for cloud '%1'")
                                 .arg(cloud->getName()));
            return 0;
        }
    }

    return ccOctree::GuessBestRadius(cloud, params, cloud->getOctree().data());
}

PointCoordinateType ccOctree::GuessBestRadius(
        ccGenericPointCloud* cloud,
        const BestRadiusParams& params,
        cloudViewer::DgmOctree* inputOctree /*=nullptr*/,
        cloudViewer::GenericProgressCallback* progressCb /*=nullptr*/) {
    if (!cloud) {
        assert(false);
        return 0;
    }

    cloudViewer::DgmOctree* octree = inputOctree;
    if (!octree) {
        octree = new cloudViewer::DgmOctree(cloud);
        if (octree->build(progressCb) <= 0) {
            delete octree;
            CVLog::Warning(
                    "[GuessBestRadius] Failed to compute the cloud octree");
            return 0;
        }
    }

    PointCoordinateType bestRadius = GuessNaiveRadius(cloud);
    if (bestRadius == 0) {
        CVLog::Warning("[GuessBestRadius] The cloud has invalid dimensions");
        return 0;
    }

    if (cloud->size() < 100) {
        // no need to do anything else for very small clouds!
        return bestRadius;
    }

    // we are now going to sample the cloud so as to compute statistics on the
    // density
    {
        const unsigned sampleCount =
                std::min<unsigned>(200, cloud->size() / 10);

        double aimedPop = params.aimedPopulationPerCell;
        PointCoordinateType radius = bestRadius;
        PointCoordinateType lastRadius = radius;
        double lastMeanPop = 0;

        std::random_device rd;   // non-deterministic generator
        std::mt19937 gen(rd());  // to seed mersenne twister.
        std::uniform_int_distribution<unsigned> dist(0, cloud->size() - 1);

        // we may have to do this several times
        for (size_t attempt = 0; attempt < 10; ++attempt) {
            int totalCount = 0;
            int totalSquareCount = 0;
            int minPop = 0;
            int maxPop = 0;
            int aboveMinPopCount = 0;

            unsigned char octreeLevel =
                    octree->findBestLevelForAGivenNeighbourhoodSizeExtraction(
                            radius);

            for (size_t i = 0; i < sampleCount; ++i) {
                unsigned randomIndex = dist(gen);
                assert(randomIndex < cloud->size());

                const CCVector3* P = cloud->getPoint(randomIndex);
                cloudViewer::DgmOctree::NeighboursSet Yk;
                int n = octree->getPointsInSphericalNeighbourhood(
                        *P, radius, Yk, octreeLevel);
                assert(n >= 1);

                totalCount += n;
                totalSquareCount += n * n;
                if (i == 0) {
                    minPop = maxPop = n;
                } else {
                    if (n < minPop)
                        minPop = n;
                    else if (n > maxPop)
                        maxPop = n;
                }

                if (n >= params.minCellPopulation) {
                    ++aboveMinPopCount;
                }
            }

            double meanPop = static_cast<double>(totalCount) / sampleCount;
            double stdDevPop = sqrt(std::abs(
                    static_cast<double>(totalSquareCount) / sampleCount -
                    meanPop * meanPop));
            double aboveMinPopRatio =
                    static_cast<double>(aboveMinPopCount) / sampleCount;

            CVLog::Print(QString("[GuessBestRadius] Radius = %1 -> samples "
                                 "population in [%2 ; %3] (mean %4 / std. dev. "
                                 "%5 / %6% above minimum)")
                                 .arg(radius)
                                 .arg(minPop)
                                 .arg(maxPop)
                                 .arg(meanPop)
                                 .arg(stdDevPop)
                                 .arg(aboveMinPopRatio * 100));

            if (std::abs(meanPop - aimedPop) < params.aimedPopulationRange) {
                // we have found a correct radius
                bestRadius = radius;

                if (aboveMinPopRatio < params.minAboveMinRatio) {
                    // CVLog::Warning("[GuessBestRadius] The cloud density is
                    // very inhomogeneous! You may have to increase the radius
                    // to get valid normals everywhere... but the result will be
                    // smoother");
                    aimedPop = params.aimedPopulationPerCell +
                               (2.0 * stdDevPop) /* * (1.0-aboveMinPopRatio)*/;
                    assert(aimedPop >= params.aimedPopulationPerCell);
                } else {
                    break;
                }
            }

            // otherwise we have to find a better estimate for the radius
            PointCoordinateType newRadius = radius;
            //(warning: we consider below that the number of points is
            // proportional to the SURFACE of the neighborhood)
            assert(meanPop >= 1.0);
            if (attempt == 0) {
                // this is our best (only) guess for the moment
                bestRadius = radius;

                newRadius = radius * sqrt(aimedPop / meanPop);
            } else {
                // keep track of our best guess nevertheless
                if (std::abs(meanPop - aimedPop) <
                    std::abs(bestRadius - aimedPop)) {
                    bestRadius = radius;
                }

                double slope = (radius * radius - lastRadius * lastRadius) /
                               (meanPop - lastMeanPop);
                PointCoordinateType newSquareRadius =
                        lastRadius * lastRadius +
                        (aimedPop - lastMeanPop) * slope;
                if (newSquareRadius > 0) {
                    newRadius = sqrt(newSquareRadius);
                } else {
                    // can't do any better!
                    break;
                }
            }

            lastRadius = radius;
            lastMeanPop = meanPop;

            radius = newRadius;
        }
    }

    if (octree && !inputOctree) {
        delete octree;
        octree = nullptr;
    }

    return bestRadius;
}
