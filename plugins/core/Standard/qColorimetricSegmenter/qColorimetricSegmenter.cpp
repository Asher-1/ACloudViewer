// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qColorimetricSegmenter.h"

#include "HSV.h"
#include "HSVDialog.h"
#include "KmeansDlg.h"
#include "QuantiDialog.h"
#include "RgbDialog.h"
#include "ScalarDialog.h"

// ACloudViewer
#include <CVLog.h>
#include <ecvPointCloud.h>

// cloudViewer
#include <DistanceComputationTools.h>

// System
#include <algorithm>
#include <map>

// Qt
#include <QMainWindow>

static void ShowDurationNow(
        const std::chrono::high_resolution_clock::time_point& startTime) {
    auto stopTime = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               stopTime - startTime)
                               .count();

    // Print duration of execution
    CVLog::Print("Time to execute: " + QString::number(duration_ms) +
                 " milliseconds");
}

static inline bool Inside(ColorCompType lower,
                          ColorCompType value,
                          ColorCompType upper) {
    Q_ASSERT(lower <= upper);
    return (value >= lower && value <= upper);
}

ColorimetricSegmenter::ColorimetricSegmenter(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/ColorimetricSegmenter/info.json"),
      m_action_filterRgb(nullptr)
      /*, m_action_filterRgbWithSegmentation(nullptr)*/
      ,
      m_action_filterHSV(nullptr),
      m_action_filterScalar(nullptr),
      m_action_histogramClustering(nullptr),
      m_action_kMeansClustering(nullptr),
      m_addPointError(false) {}

void ColorimetricSegmenter::onNewSelection(
        const ccHObject::Container& selectedEntities) {
    // Only enable our action if something is selected.
    bool activateColorFilters = false;
    bool activateScalarFilter = false;
    for (ccHObject* entity : selectedEntities) {
        if (entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            if (entity->hasColors()) {
                activateColorFilters = true;
            } else if (entity->hasDisplayedScalarField()) {
                activateScalarFilter = true;
            }
        }
    }

    // Activate only if only one of them is activated
    if (activateColorFilters && activateScalarFilter) {
        activateColorFilters = activateScalarFilter = false;
    }

    if (m_action_filterRgb)
        m_action_filterRgb->setEnabled(activateColorFilters);
    if (m_action_filterHSV)
        m_action_filterHSV->setEnabled(activateColorFilters);
    // if (m_action_filterRgbWithSegmentation)
    //	m_action_filterRgbWithSegmentation->setEnabled(activateColorFilters);
    if (m_action_filterScalar)
        m_action_filterScalar->setEnabled(activateScalarFilter);
    if (m_action_histogramClustering)
        m_action_histogramClustering->setEnabled(activateColorFilters);
    if (m_action_kMeansClustering)
        m_action_kMeansClustering->setEnabled(activateColorFilters);
}

QList<QAction*> ColorimetricSegmenter::getActions() {
    // RGB Filter
    if (!m_action_filterRgb) {
        m_action_filterRgb = new QAction("Filter RGB", this);
        m_action_filterRgb->setToolTip(
                "Filter the points of the selected cloud by RGB color");
        m_action_filterRgb->setIcon(
                QIcon(":/CC/plugin/ColorimetricSegmenter/images/icon_rgb.png"));

        // Connect appropriate signal
        connect(m_action_filterRgb, &QAction::triggered, this,
                &ColorimetricSegmenter::filterRgb);
    }

    /*if (!m_action_filterRgbWithSegmentation)
    {
            // Here we use the default plugin name, description, and icon,
            // but each action should have its own.
            m_action_filterRgbWithSegmentation = new QAction("Filter RGB using
    segmentation", this); m_action_filterRgbWithSegmentation->setToolTip("Filter
    the points on the selected cloud by RGB color using segmentation");
            m_action_filterRgbWithSegmentation->setIcon(getIcon());

            // Connect appropriate signal
            connect(m_action_filterRgbWithSegmentation, &QAction::triggered,
    this, &ColorimetricSegmenter::filterRgbWithSegmentation);
    }*/

    // HSV Filter
    if (!m_action_filterHSV) {
        m_action_filterHSV = new QAction("Filter HSV", this);
        m_action_filterHSV->setToolTip(
                "Filter the points of the selected cloud by HSV color");
        m_action_filterHSV->setIcon(
                QIcon(":/CC/plugin/ColorimetricSegmenter/images/icon_hsv.png"));

        // Connect appropriate signal
        connect(m_action_filterHSV, &QAction::triggered, this,
                &ColorimetricSegmenter::filterHSV);
    }

    // Scalar filter
    if (!m_action_filterScalar) {
        m_action_filterScalar = new QAction("Filter scalar", this);
        m_action_filterScalar->setToolTip(
                "Filter the points of the selected cloud using scalar field");
        m_action_filterScalar->setIcon(QIcon(
                ":/CC/plugin/ColorimetricSegmenter/images/icon_scalar.png"));

        // Connect appropriate signal
        connect(m_action_filterScalar, &QAction::triggered, this,
                &ColorimetricSegmenter::filterScalar);
    }

    if (!m_action_histogramClustering) {
        m_action_histogramClustering =
                new QAction("Histogram Clustering", this);
        m_action_histogramClustering->setToolTip(
                "Quantify the number of colors using Histogram Clustering");
        m_action_histogramClustering->setIcon(QIcon(
                ":/CC/plugin/ColorimetricSegmenter/images/icon_quantif_h.png"));

        // Connect appropriate signal
        connect(m_action_histogramClustering, &QAction::triggered, this,
                &ColorimetricSegmenter::HistogramClustering);
    }

    if (!m_action_kMeansClustering) {
        // Here we use the default plugin name, description, and icon,
        // but each action should have its own.
        m_action_kMeansClustering = new QAction("Kmeans Clustering", this);
        m_action_kMeansClustering->setToolTip(
                "Quantify the number of colors using Kmeans Clustering");
        m_action_kMeansClustering->setIcon(QIcon(
                ":/CC/plugin/ColorimetricSegmenter/images/icon_quantif_k.png"));

        // Connect appropriate signal
        connect(m_action_kMeansClustering, &QAction::triggered, this,
                &ColorimetricSegmenter::KmeansClustering);
    }

    return {m_action_filterRgb, m_action_filterHSV,
            // m_action_filterRgbWithSegmentation,
            m_action_filterScalar, m_action_histogramClustering,
            m_action_kMeansClustering};
}

// Get all point clouds that are selected in CC
// return a vector with ccPointCloud objects
std::vector<ccPointCloud*> ColorimetricSegmenter::getSelectedPointClouds() {
    if (m_app == nullptr) {
        // m_app should have already been initialized by CC when plugin is
        // loaded
        Q_ASSERT(false);
        return std::vector<ccPointCloud*>{};
    }

    ccHObject::Container selectedEntities = m_app->getSelectedEntities();
    std::vector<ccPointCloud*> clouds;
    for (size_t i = 0; i < selectedEntities.size(); ++i) {
        if (selectedEntities[i]->isKindOf(CV_TYPES::POINT_CLOUD)) {
            clouds.push_back(static_cast<ccPointCloud*>(selectedEntities[i]));
        }
    }

    return clouds;
}

// Algorithm for the RGB filter
// It uses a color range with RGB values, and keeps the points with a color
// within that range.
void ColorimetricSegmenter::filterRgb() {
    if (m_app == nullptr) {
        // m_app should have already been initialized by CC when plugin is
        // loaded
        Q_ASSERT(false);

        return;
    }

    // check valid window
    if (!m_app->getActiveWindow()) {
        m_app->dispToConsole("[ColorimetricSegmenter] No active 3D view",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    std::vector<ccPointCloud*> clouds = getSelectedPointClouds();
    if (clouds.empty()) {
        Q_ASSERT(false);
        return;
    }

    // Retrieve parameters from dialog
    RgbDialog rgbDlg(m_app->pickingHub(), m_app->getMainWindow());

    rgbDlg.show();  // necessary for setModal to be retained

    if (!rgbDlg.exec()) return;

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Get all values to make the color range with RGB values
    int redInf = static_cast<int>(
            std::min(rgbDlg.red_first->value(), rgbDlg.red_second->value()));
    int redSup = static_cast<int>(
            std::max(rgbDlg.red_first->value(), rgbDlg.red_second->value()));
    int greenInf = static_cast<int>(std::min(rgbDlg.green_first->value(),
                                             rgbDlg.green_second->value()));
    int greenSup = static_cast<int>(std::max(rgbDlg.green_first->value(),
                                             rgbDlg.green_second->value()));
    int blueInf = static_cast<int>(
            std::min(rgbDlg.blue_first->value(), rgbDlg.blue_second->value()));
    int blueSup = static_cast<int>(
            std::max(rgbDlg.blue_first->value(), rgbDlg.blue_second->value()));

    if (rgbDlg.margin->value() > 0) {
        // error margin
        int marginError =
                static_cast<int>(rgbDlg.margin->value() * 2.56);  // 256 / 100%

        redInf -= marginError;
        redSup += marginError;
        greenInf -= marginError;
        greenSup += marginError;
        blueInf -= marginError;
        blueSup += marginError;
    }

    // Set to min or max value (0-255)
    {
        redInf = std::max(redInf, 0);
        greenInf = std::max(greenInf, 0);
        blueInf = std::max(blueInf, 0);

        redSup = std::min(redSup, 255);
        greenSup = std::min(greenSup, 255);
        blueSup = std::min(blueSup, 255);
    }

    for (ccPointCloud* cloud : clouds) {
        if (cloud && cloud->hasColors()) {
            // Use only references for speed reasons
            cloudViewer::ReferenceCloud filteredCloudInside(cloud);
            cloudViewer::ReferenceCloud filteredCloudOutside(cloud);

            for (unsigned j = 0; j < cloud->size(); ++j) {
                const ecvColor::Rgb& rgb = cloud->getPointColor(j);
                if (Inside(static_cast<ColorCompType>(redInf), rgb.r,
                           static_cast<ColorCompType>(redSup)) &&
                    Inside(static_cast<ColorCompType>(greenInf), rgb.g,
                           static_cast<ColorCompType>(greenSup)) &&
                    Inside(static_cast<ColorCompType>(blueInf), rgb.b,
                           static_cast<ColorCompType>(blueSup))) {
                    addPoint(filteredCloudInside, j);
                } else {
                    addPoint(filteredCloudOutside, j);
                }

                if (m_addPointError) {
                    return;
                }
            }
            QString name = "Rmin:" + QString::number(redInf) +
                           "/Gmin:" + QString::number(greenInf) +
                           "/Bmin:" + QString::number(blueInf) +
                           "/Rmax:" + QString::number(redSup) +
                           "/Gmax:" + QString::number(greenSup) +
                           "/Bmax:" + QString::number(blueSup);

            createClouds<const RgbDialog&>(rgbDlg, cloud, filteredCloudInside,
                                           filteredCloudOutside, name);

            m_app->dispToConsole(
                    "[ColorimetricSegmenter] Cloud successfully filtered ! ",
                    ecvMainAppInterface::STD_CONSOLE_MESSAGE);
        }
    }

    ShowDurationNow(startTime);
}

void ColorimetricSegmenter::filterScalar() {
    if (m_app == nullptr) {
        // m_app should have already been initialized by CC when plugin is
        // loaded
        Q_ASSERT(false);

        return;
    }

    // check valid window
    if (!m_app->getActiveWindow()) {
        m_app->dispToConsole("[ColorimetricSegmenter] No active 3D view",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // Retrieve parameters from dialog
    ScalarDialog scalarDlg(m_app->pickingHub(), m_app->getMainWindow());

    scalarDlg.show();  // necessary for setModal to be retained

    if (!scalarDlg.exec()) return;

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    ScalarType marginError = scalarDlg.margin->value() / 100.0f;
    ScalarType minVal = static_cast<ScalarType>(
            std::min(scalarDlg.first->value(), scalarDlg.second->value()));
    ScalarType maxVal = static_cast<ScalarType>(
            std::max(scalarDlg.first->value(), scalarDlg.second->value()));
    // DGM: this way of applying the error margin is a bit strange
    minVal -= (marginError * minVal);
    maxVal += (marginError * maxVal);

    std::vector<ccPointCloud*> clouds = getSelectedPointClouds();
    for (ccPointCloud* cloud : clouds) {
        // Use only references for speed reasons
        cloudViewer::ReferenceCloud filteredCloudInside(cloud);
        cloudViewer::ReferenceCloud filteredCloudOutside(cloud);

        for (unsigned j = 0; j < cloud->size(); ++j) {
            const ScalarType val = cloud->getPointScalarValue(j);
            addPoint(val >= minVal && val <= maxVal ? filteredCloudInside
                                                    : filteredCloudOutside,
                     j);

            if (m_addPointError) {
                return;
            }
        }
        QString name = "min:" + QString::number(minVal) +
                       "/max:" + QString::number(maxVal);

        createClouds<const ScalarDialog&>(scalarDlg, cloud, filteredCloudInside,
                                          filteredCloudOutside, name);

        m_app->dispToConsole(
                "[ColorimetricSegmenter] Cloud successfully filtered ! ",
                ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }

    ShowDurationNow(startTime);
}

typedef QSharedPointer<cloudViewer::ReferenceCloud> _Region;
typedef std::vector<_Region> _RegionSet;
typedef std::vector<_RegionSet> SetOfRegionSet;

/**
 * @brief KNNRegions Determines the neighboring regions of a region.
 * @param basePointCloud The base cloud containing the points.
 * @param regions The list containing all the regions in the base cloud point.
 * @param region The region to compare with others.
 * @param k Max number of neighbours to find.
 * @param neighbours The resulting nearest regions.
 * @param thresholdDistance The maximum distance to search for neighbors.
 */
static bool KNNRegions(ccPointCloud* basePointCloud,
                       const _RegionSet& regions,
                       const _Region& region,
                       unsigned k,
                       _RegionSet& neighbourRegions,
                       unsigned thresholdDistance) {
    QScopedPointer<ccPointCloud> regionCloud(
            basePointCloud->partialClone(region.data()));
    if (!regionCloud) {
        // not enough memory
        return false;
    }

    // compute distances
    cloudViewer::DistanceComputationTools::Cloud2CloudDistancesComputationParams
            params = cloudViewer::DistanceComputationTools::
                    Cloud2CloudDistancesComputationParams();
    {
        params.kNNForLocalModel = k;
        params.maxSearchDist = thresholdDistance;
    }

    std::vector<double> distancesToCentralRegion;
    distancesToCentralRegion.reserve(regions.size());

    for (const _Region& r : regions) {
        QScopedPointer<ccPointCloud> neighbourCloud(
                basePointCloud->partialClone(r.data()));
        if (!neighbourCloud) {
            // not enough memory
            return false;
        }
        // DGM: warning, the computeCloud2CloudDistances method doesn't return a
        // distance value (but a status / error) distances are stored in the
        // active scalar field (one per point!)
        int result = cloudViewer::DistanceComputationTools::
                computeCloud2CloudDistances(neighbourCloud.data(),
                                            regionCloud.data(), params);
        if (result >= 0) {
            double meanDistance = 0.0;
            for (unsigned i = 0; i < neighbourCloud->size(); ++i) {
                meanDistance += static_cast<double>(
                        neighbourCloud->getPointScalarValue(i));
            }
            meanDistance /= neighbourCloud->size();

            distancesToCentralRegion.push_back(meanDistance);
        } else {
            // failed to compute the distances
            return false;
        }
    }
    regionCloud.reset(nullptr);

    // sort the regions by their distance
    std::vector<size_t> regionIndices(regions.size());
    for (size_t i = 0; i < regions.size(); ++i) regionIndices[i] = i;

    std::sort(regionIndices.begin(), regionIndices.end(),
              [&](size_t a, size_t b) {
                  return distancesToCentralRegion[a] <
                         distancesToCentralRegion[b];
              });

    // then extract the 'k' nearest regions.
    for (size_t regionIndex : regionIndices) {
        if (neighbourRegions.size() < k) {
            neighbourRegions.push_back(regions[regionIndex]);
        } else {
            break;
        }
    }

    return true;
}

/**
 * @brief colorimetricalDifference Compute colorimetrical difference between two
 * RGB color values.
 * @param c1 First color value.
 * @param c2 Second color value.
 * @return Colorimetrical difference.
 */
static double ColorimetricalDifference(ecvColor::Rgb c1, ecvColor::Rgb c2) {
    int dr = static_cast<int>(c1.r) - c2.r;
    int dg = static_cast<int>(c1.g) - c2.g;
    int db = static_cast<int>(c1.b) - c2.b;
    return sqrt(static_cast<double>(dr * dr + dg * dg + db * db));
}

/**
Compute the average color (RGB)
@param cloud : cloud which contains the points
@param subset : subset of points
Returns average color (RGB)
*/
static ecvColor::Rgb ComputeAverageColor(const ccPointCloud& cloud,
                                         cloudViewer::ReferenceCloud* subset) {
    if (!subset || subset->size() == 0) {
        Q_ASSERT(false);
        return ecvColor::white;
    }

    size_t count = subset->size();
    if (count == 0) {
        return ecvColor::white;
    } else if (count == 1) {
        return cloud.getPointColor(subset->getPointGlobalIndex(0));
    }

    // other formula to compute the average can be used
    size_t redSum = 0, greenSum = 0, blueSum = 0;
    for (unsigned j = 0; j < subset->size(); ++j) {
        const ecvColor::Rgb& rgb =
                cloud.getPointColor(subset->getPointGlobalIndex(j));
        redSum += rgb.r;
        greenSum += rgb.g;
        blueSum += rgb.b;
    }

    ecvColor::Rgb res(
            static_cast<ColorCompType>(std::min(
                    redSum / count, static_cast<size_t>(ecvColor::MAX))),
            static_cast<ColorCompType>(std::min(
                    greenSum / count, static_cast<size_t>(ecvColor::MAX))),
            static_cast<ColorCompType>(std::min(
                    blueSum / count, static_cast<size_t>(ecvColor::MAX))));

    return res;
}

/**
 * @brief colorimetricalDifference compute mean colorimetrical difference
 * between two reference clouds. The points in both clouds must be represented
 * in RGB value.
 * @param basePointCloud The base cloud on which the reference clouds are based.
 * @param c1 The first reference cloud.
 * @param c2 The second reference cloud.
 * @return Colorimetrical difference.
 */
double ColorimetricalDifference(const ccPointCloud& basePointCloud,
                                cloudViewer::ReferenceCloud* c1,
                                cloudViewer::ReferenceCloud* c2) {
    ecvColor::Rgb rgb1 = ComputeAverageColor(basePointCloud, c1);
    ecvColor::Rgb rgb2 = ComputeAverageColor(basePointCloud, c2);

    return ColorimetricalDifference(rgb1, rgb2);
}

bool ColorimetricSegmenter::RegionGrowing(RegionSet& regions,
                                          ccPointCloud* pointCloud,
                                          const unsigned TNN,
                                          const double TPP,
                                          const double TD) {
    if (!pointCloud || pointCloud->size() == 0) {
        Q_ASSERT(false);
        return false;
    }
    size_t pointCount = pointCloud->size();

    try {
        std::vector<unsigned> unlabeledPoints;
        unlabeledPoints.resize(pointCount);
        for (unsigned j = 0; j < pointCount; ++j) {
            unlabeledPoints.push_back(j);
        }

        std::vector<unsigned> pointIndices;

        cloudViewer::DgmOctree* octree = new cloudViewer::DgmOctree(
                pointCloud);  // used to search nearest neighbors
        octree->build();

        // while there is points in {P} that haven’t been labeled
        while (!unlabeledPoints.empty()) {
            // push an unlabeled point into stack Points
            pointIndices.push_back(unlabeledPoints.back());
            unlabeledPoints.pop_back();

            // initialize a new region Rc and add current point to R
            Region rc(new cloudViewer::ReferenceCloud(pointCloud));
            rc->addPointIndex(unlabeledPoints.back());

            // while stack Points is not empty
            while (!pointIndices.empty()) {
                // pop Points’ top element Tpoint
                unsigned tPointIndex = pointIndices.back();
                pointIndices.pop_back();

                // for each point p in {KNNTNN(Tpoint)}
                cloudViewer::DgmOctree::NearestNeighboursSearchStruct nNSS =
                        cloudViewer::DgmOctree::NearestNeighboursSearchStruct();
                {
                    nNSS.level = 1;
                    nNSS.queryPoint = *(pointCloud->getPoint(tPointIndex));
                    octree->getCellPos(octree->getCellCode(tPointIndex), 1,
                                       nNSS.cellPos, false);
                    octree->computeCellCenter(octree->getCellCode(tPointIndex),
                                              1, nNSS.cellCenter);
                    nNSS.maxSearchSquareDistd = TD;
                    nNSS.minNumberOfNeighbors = TNN;
                }
                octree->findNearestNeighborsStartingFromCell(nNSS);

                for (std::size_t i = 0; i < nNSS.pointsInNeighbourhood.size();
                     i++) {
                    unsigned p = nNSS.pointsInNeighbourhood[i].pointIndex;
                    // if p is labelled
                    if (std::find(unlabeledPoints.begin(),
                                  unlabeledPoints.end(),
                                  p) != unlabeledPoints.end()) {
                        continue;
                    }

                    if (ColorimetricalDifference(
                                pointCloud->getPointColor(p),
                                pointCloud->getPointColor(tPointIndex)) < TPP) {
                        pointIndices.push_back(p);
                        rc->addPointIndex(p);
                    }
                }
            }
            regions.push_back(rc);
        }
    } catch (const std::bad_alloc&) {
        // not enough memory
        return false;
    }

    return true;
}

/**
 * @brief findRegion Find a given region in a vector of reference clouds.
 * @param container Container containing all the regions.
 * @param region Region to search for in the vector.
 * @return The index of the region if found, -1 in the other case.
 */
static int FindRegion(const std::vector<_RegionSet>& container,
                      cloudViewer::ReferenceCloud* region) {
    for (size_t i = 0; i < container.size(); ++i) {
        const _RegionSet& l = container[i];
        if (std::find(l.begin(), l.end(), region) != l.end()) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool ColorimetricSegmenter::RegionMergingAndRefinement(
        RegionSet& mergedRegions,
        ccPointCloud* basePointCloud,
        const RegionSet& regions,
        const unsigned TNN,
        const double TRR,
        const double TD,
        const unsigned Min) {
    std::vector<_RegionSet> homogeneous;

    // for each region Ri in {R}
    for (const Region& ri : regions) {
        // if Ri is not in {H}
        _RegionSet* riSet = nullptr;
        int riSetIndex = FindRegion(homogeneous, ri.data());
        if (riSetIndex == -1) {
            // create a new list to record Ri
            homogeneous.resize(homogeneous.size() + 1);
            riSet = &homogeneous.back();
            riSet->push_back(ri);
        } else {
            riSet = &(homogeneous[riSetIndex]);
        }

        // for each region Rj in {KNNTNN2,TD2(Ri)}
        RegionSet knnResult;
        if (!KNNRegions(basePointCloud, regions, ri, TNN, knnResult, TD)) {
            // process failed
            return false;
        }

        for (const Region& rj : knnResult) {
            // if CD(Ri,Rj)<TRR
            if (ColorimetricalDifference(*basePointCloud, ri.data(),
                                         rj.data()) < TNN) {
                // if Rj is in {H}
                int regionIndex = FindRegion(homogeneous, rj.data());
                if (regionIndex < 0) {
                    // add Rj to the list which contains Ri
                    riSet->push_back(rj);
                }
            }
        }
    }

    // merge all the regions in the same list in {H} and get {R’}
    for (const _RegionSet& l : homogeneous) {
        Region merged(
                new cloudViewer::ReferenceCloud(l[0]->getAssociatedCloud()));
        for (const Region& li : l) {
            if (li && !merged->add(*li)) {
                // not enough memory
                return false;
            }
        }
        mergedRegions.push_back(merged);
    }

    // std::vector<cloudViewer::ReferenceCloud*>* knnResult;
    //  for each region Ri in {R’}
    /*for (cloudViewer::ReferenceCloud* r : *mergedRegionsRef)
    {
            // if sizeof(Ri)<Min
            if(r->size() < Min)
            {
                    // merge Ri to its nearest neighbors
                    KNNRegions(basePointCloud, mergedRegionsRef, r, 1,
    knnResult, 0); mergedRegionsRef.
            }
    }*/

    // Return the merged and refined {R’}
    return true;
}

// filterRgbWithSegmentation parameters
static const unsigned TNN = 1;
static const double TPP = 2.0;
static const double TD = 2.0;
static const double TRR = 2.0;
static const unsigned Min = 2;

void ColorimetricSegmenter::filterRgbWithSegmentation() {
    if (m_app == nullptr) {
        // m_app should have already been initialized by CC when plugin is
        // loaded
        Q_ASSERT(false);

        return;
    }

    std::vector<ccPointCloud*> clouds = getSelectedPointClouds();
    if (clouds.empty()) {
        Q_ASSERT(false);
        return;
    }

    // Retrieve parameters from dialog
    RgbDialog rgbDlg(m_app->pickingHub(), m_app->getMainWindow());

    rgbDlg.show();  // necessary for setModal to be retained

    if (!rgbDlg.exec()) return;

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Get margin value (percent)
    double marginError = rgbDlg.margin->value() / 100.0;

    // Get all values to make the color range with RGB values
    int redInf = static_cast<int>(
            rgbDlg.red_first->value() -
            static_cast<int>(marginError * rgbDlg.red_first->value()));
    int redSup = static_cast<int>(
            rgbDlg.red_second->value() +
            static_cast<int>(marginError * rgbDlg.red_second->value()));
    int greenInf = static_cast<int>(
            rgbDlg.green_first->value() -
            static_cast<int>(marginError * rgbDlg.green_first->value()));
    int greenSup = static_cast<int>(
            rgbDlg.green_second->value() +
            static_cast<int>(marginError * rgbDlg.green_second->value()));
    int blueInf = static_cast<int>(
            rgbDlg.blue_first->value() -
            static_cast<int>(marginError * rgbDlg.blue_first->value()));
    int blueSup = static_cast<int>(
            rgbDlg.blue_second->value() +
            static_cast<int>(marginError * rgbDlg.blue_second->value()));

    redInf = std::max(0, redInf);
    greenInf = std::max(0, greenInf);
    blueInf = std::max(0, blueInf);
    redSup = std::min(255, redSup);
    greenSup = std::min(255, greenSup);
    blueSup = std::min(255, blueSup);

    for (ccPointCloud* cloud : clouds) {
        if (cloud->hasColors()) {
            RegionSet regions;
            if (!RegionGrowing(regions, cloud, TNN, TPP, TD)) {
                CVLog::Error("Process failed (not enough memory?)");
                return;
            }

            RegionSet mergedRegions;
            RegionMergingAndRefinement(mergedRegions, cloud, regions, TNN, TRR,
                                       TD, Min);
            // m_app->dispToConsole(QString("[ColorimetricSegmenter] regions
            // %1").arg(regions->size()),
            // ecvMainAppInterface::STD_CONSOLE_MESSAGE);

            // retrieve the nearest region (in color range)
            for (Region& r : mergedRegions) {
                ecvColor::Rgb mean = ComputeAverageColor(*cloud, r.data());
                if (Inside(static_cast<ColorCompType>(redInf), mean.r,
                           static_cast<ColorCompType>(redSup)) &&
                    Inside(static_cast<ColorCompType>(greenInf), mean.g,
                           static_cast<ColorCompType>(greenSup)) &&
                    Inside(static_cast<ColorCompType>(blueInf), mean.b,
                           static_cast<ColorCompType>(blueSup))) {
                    ccPointCloud* newCloud = cloud->partialClone(r.data());
                    if (newCloud) {
                        cloud->setEnabled(false);
                        if (cloud->getParent()) {
                            cloud->getParent()->addChild(newCloud);
                        }

                        m_app->addToDB(newCloud, false, true, false, true);

                        m_app->dispToConsole(
                                "[ColorimetricSegmenter] Cloud successfully "
                                "filtered with segmentation!",
                                ecvMainAppInterface::STD_CONSOLE_MESSAGE);
                    } else {
                        m_app->dispToConsole(
                                "Not enough memory",
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                        return;
                    }
                }
            }
        }
    }

    ShowDurationNow(startTime);
}

// Algorithm for the HSV filter
// It uses the Hue-Saturation-Value (HSV) color space to filter the point cloud
void ColorimetricSegmenter::filterHSV() {
    if (m_app == nullptr) {
        // m_app should have already been initialized by CC when plugin is
        // loaded
        Q_ASSERT(false);
        return;
    }

    // Check valid window
    if (!m_app->getActiveWindow()) {
        m_app->dispToConsole("[ColorimetricSegmenter] No active 3D view",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    std::vector<ccPointCloud*> clouds = getSelectedPointClouds();
    if (clouds.empty()) {
        Q_ASSERT(false);
        return;
    }

    // Retrieve parameters from dialog
    HSVDialog hsvDlg(m_app->pickingHub(), m_app->getMainWindow());

    hsvDlg.show();  // necessary for setModal to be retained

    if (!hsvDlg.exec()) return;

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Get HSV values
    Hsv hsv_first;
    hsv_first.h = static_cast<uint16_t>(hsvDlg.hue_first->value());
    hsv_first.s = static_cast<uint16_t>(hsvDlg.sat_first->value());
    hsv_first.v = static_cast<uint16_t>(hsvDlg.val_first->value());

    // We use look-up tables for faster comparisons
    static const uint8_t LOW = 0;
    static const uint8_t MIDDLE = 1;
    static const uint8_t HIGH = 2;
    uint8_t level[101];
    {
        for (unsigned i = 0; i <= 25; ++i) level[i] = LOW;
        for (unsigned i = 26; i <= 60; ++i) level[i] = MIDDLE;
        for (unsigned i = 61; i <= 100; ++i) level[i] = HIGH;
    }

    static const uint8_t RED = 0;
    static const uint8_t YELLOW = 1;
    static const uint8_t GREEN = 2;
    static const uint8_t CYAN = 3;
    static const uint8_t BLUE = 4;
    static const uint8_t MAGENTA = 5;
    uint8_t section[360];
    {
        for (unsigned i = 0; i <= 30; ++i) section[i] = RED;
        for (unsigned i = 31; i <= 90; ++i) section[i] = YELLOW;
        for (unsigned i = 91; i <= 150; ++i) section[i] = GREEN;
        for (unsigned i = 151; i <= 210; ++i) section[i] = CYAN;
        for (unsigned i = 211; i <= 270; ++i) section[i] = BLUE;
        for (unsigned i = 271; i <= 330; ++i) section[i] = MAGENTA;
        for (unsigned i = 331; i <= 359; ++i) section[i] = RED;
    }

    for (ccPointCloud* cloud : clouds) {
        if (cloud->hasColors()) {
            // Use only references for speed reasons
            cloudViewer::ReferenceCloud filteredCloudInside(cloud);
            cloudViewer::ReferenceCloud filteredCloudOutside(cloud);

            // We manually add color ranges with HSV values
            for (unsigned j = 0; j < cloud->size(); ++j) {
                const ecvColor::Rgb& rgb = cloud->getPointColor(j);
                Hsv hsv_current(rgb);
                assert(hsv_current.h <= 359);
                assert(hsv_current.s <= 100);
                assert(hsv_current.v <= 100);

                if (level[hsv_first.s] == LOW &&
                    level[hsv_current.s] == LOW)  // low saturation
                {
                    // If Saturation is too small, considering Hue is useless
                    // We only check that Value is equivalent
                    addPoint(level[hsv_first.v] == level[hsv_current.v]
                                     ? filteredCloudInside
                                     : filteredCloudOutside,
                             j);
                } else if (level[hsv_first.s] != LOW &&
                           level[hsv_current.s] !=
                                   LOW)  // middle to high saturation
                {
                    if (level[hsv_first.v] == LOW &&
                        level[hsv_current.v] == LOW)  // dark
                    {
                        addPoint(filteredCloudInside, j);
                    } else if (level[hsv_first.v] != LOW &&
                               level[hsv_current.v] != LOW)  // non-dark
                    {
                        addPoint(section[hsv_first.h] == section[hsv_current.h]
                                         ? filteredCloudInside
                                         : filteredCloudOutside,
                                 j);
                    } else {
                        addPoint(filteredCloudOutside, j);
                    }
                } else {
                    addPoint(filteredCloudOutside, j);
                }

                if (m_addPointError) {
                    return;
                }
            }

            QString name = "h:" + QString::number(hsv_first.h, 'f', 0) +
                           "/s:" + QString::number(hsv_first.s, 'f', 0) +
                           "/v:" + QString::number(hsv_first.v, 'f', 0);
            createClouds<const HSVDialog&>(hsvDlg, cloud, filteredCloudInside,
                                           filteredCloudOutside, name);

            m_app->dispToConsole(
                    "[ColorimetricSegmenter] Cloud successfully filtered ! ",
                    ecvMainAppInterface::STD_CONSOLE_MESSAGE);
        }
    }

    ShowDurationNow(startTime);
}

// Method to add point to a ReferenceCloud*
bool ColorimetricSegmenter::addPoint(cloudViewer::ReferenceCloud& filteredCloud,
                                     unsigned int j) {
    m_addPointError = !filteredCloud.addPointIndex(j);

    if (m_addPointError) {
        // not enough memory
        m_app->dispToConsole("[ColorimetricSegmenter] Error, filter canceled.");
    }

    return m_addPointError;
}

// Method to interact with the component "Which points to keep"
template <typename T>
void ColorimetricSegmenter::createClouds(
        T& dlg,
        ccPointCloud* cloud,
        const cloudViewer::ReferenceCloud& filteredCloudInside,
        const cloudViewer::ReferenceCloud& filteredCloudOutside,
        QString name) {
    if (dlg.retain->isChecked()) {
        createCloud(cloud, filteredCloudInside, name + ".inside");
    } else if (dlg.exclude->isChecked()) {
        createCloud(cloud, filteredCloudOutside, name + ".outside");
    } else if (dlg.both->isChecked()) {
        createCloud(cloud, filteredCloudInside, name + ".inside");
        createCloud(cloud, filteredCloudOutside, name + ".outside");
    }
}

// Method to create a new cloud
void ColorimetricSegmenter::createCloud(
        ccPointCloud* cloud,
        const cloudViewer::ReferenceCloud& referenceCloud,
        QString name) {
    if (!cloud) {
        Q_ASSERT(false);
        return;
    }

    ccPointCloud* newCloud = cloud->partialClone(&referenceCloud);
    if (!newCloud) {
        m_app->dispToConsole("Not enough memory");
        return;
    }

    newCloud->setName(name);
    cloud->setEnabled(false);
    if (cloud->getParent()) {
        cloud->getParent()->addChild(newCloud);
    }

    m_app->addToDB(newCloud, false, true, false, true);
}

/**
Generate nxnxn clusters of points according to their color value (RGB)
@param cloud : the cloud which we work with
@param clusterPerDim : coefficient uses to split each RGB component
Returns a map of nxnxn keys, for each key a vector of the points index in the
partition
*/
typedef std::map<size_t, std::vector<unsigned>> ClusterMap;
static bool GetKeyCluster(const ccPointCloud& cloud,
                          size_t clusterPerDim,
                          ClusterMap& clusterMap) {
    Q_ASSERT(ecvColor::MAX == 255);

    try {
        for (unsigned i = 0; i < cloud.size(); i++) {
            const ecvColor::Rgb& rgb = cloud.getPointColor(i);

            size_t redCluster = (static_cast<size_t>(rgb.r) * clusterPerDim) >>
                                8;  // shift 8 bits (= division by 256)
            size_t greenCluster =
                    (static_cast<size_t>(rgb.g) * clusterPerDim) >>
                    8;  // shift 8 bits (= division by 256)
            size_t blueCluster = (static_cast<size_t>(rgb.b) * clusterPerDim) >>
                                 8;  // shift 8 bits (= division by 256)

            size_t index =
                    redCluster + (greenCluster + blueCluster * clusterPerDim) *
                                         clusterPerDim;

            // we add the point to the right container
            clusterMap[index].push_back(i);
        }
    } catch (const std::bad_alloc&) {
        return false;
    }

    return true;
}
/**
Compute the average color (RGB)
@param cloud : cloud which contains the points
@param bucket : vector of indexes of points
Returns average color (RGB)
*/
static ecvColor::Rgb ComputeAverageColor(const ccPointCloud& cloud,
                                         const std::vector<unsigned>& bucket) {
    size_t count = bucket.size();
    if (count == 0) {
        return ecvColor::white;
    } else if (count == 1) {
        return cloud.getPointColor(bucket.front());
    }

    // other formula to compute the average can be used
    size_t redSum = 0, greenSum = 0, blueSum = 0;
    for (unsigned pointIndex : bucket) {
        const ecvColor::Rgb& rgba = cloud.getPointColor(pointIndex);
        redSum += rgba.r;
        greenSum += rgba.g;
        blueSum += rgba.b;
    }

    ecvColor::Rgb res(
            static_cast<ColorCompType>(std::min(
                    redSum / count, static_cast<size_t>(ecvColor::MAX))),
            static_cast<ColorCompType>(std::min(
                    greenSum / count, static_cast<size_t>(ecvColor::MAX))),
            static_cast<ColorCompType>(std::min(
                    blueSum / count, static_cast<size_t>(ecvColor::MAX))));

    return res;
}

/**
Compute the distance between two colors
/!\ the formula can be modified, here it is simple to be as quick as possible
*/
static int ColorDistance(const ecvColor::Rgb& c1, const ecvColor::Rgb& c2) {
    return (static_cast<int>(c1.r) - c2.r) + (static_cast<int>(c1.b) - c2.b) +
           (static_cast<int>(c1.g) - c2.g);
}

/**
Generate a pointcloud quantified using an histogram clustering
The purpose is to counter luminance variation due to the merge of different
scans
*/
void ColorimetricSegmenter::HistogramClustering() {
    if (m_app == nullptr) {
        // m_app should have already been initialized by CC when plugin is
        // loaded
        Q_ASSERT(false);

        return;
    }

    std::vector<ccPointCloud*> clouds =
            ColorimetricSegmenter::getSelectedPointClouds();
    if (clouds.empty()) {
        Q_ASSERT(false);
        return;
    }

    // creation of the window
    QuantiDialog quantiDlg(m_app->getMainWindow());
    if (!quantiDlg.exec()) return;

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    int nbClusterByComponent = quantiDlg.area_quanti->value();
    if (nbClusterByComponent < 0) {
        Q_ASSERT(false);
        return;
    }

    for (ccPointCloud* cloud : clouds) {
        if (cloud->hasColors()) {
            ClusterMap clusterMap;
            if (!GetKeyCluster(*cloud,
                               static_cast<size_t>(nbClusterByComponent),
                               clusterMap)) {
                m_app->dispToConsole("Not enough memory",
                                     ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                break;
            }

            ccPointCloud* histCloud = cloud->cloneThis();
            if (!histCloud) {
                m_app->dispToConsole("Not enough memory",
                                     ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                break;
            }

            histCloud->setName(
                    QString("HistogramClustering: Indice Q = %1 // colors = %2")
                            .arg(nbClusterByComponent)
                            .arg(nbClusterByComponent * nbClusterByComponent *
                                 nbClusterByComponent));

            for (auto it = clusterMap.begin(); it != clusterMap.end(); it++) {
                ecvColor::Rgb averageColor =
                        ComputeAverageColor(*histCloud, it->second);

                for (unsigned pointIndex : it->second) {
                    (*histCloud).setPointColor(pointIndex, averageColor);
                }
            }

            cloud->setEnabled(false);
            if (cloud->getParent()) {
                cloud->getParent()->addChild(histCloud);
            }

            m_app->addToDB(histCloud, false, true, false, true);

            m_app->dispToConsole(
                    "[ColorimetricSegmenter] Cloud successfully clustered!",
                    ecvMainAppInterface::STD_CONSOLE_MESSAGE);
        }
    }

    ShowDurationNow(startTime);
}

/**
K-means algorithm
@param k : k clusters
@param it : limit of iterations before returns a result
Returns a cloud quantified
*/
static ccPointCloud* ComputeKmeansClustering(ccPointCloud* theCloud,
                                             unsigned K,
                                             int maxIterationCount) {
    // valid parameters?
    if (!theCloud || K == 0) {
        Q_ASSERT(false);
        return nullptr;
    }

    unsigned pointCount = theCloud->size();
    if (pointCount == 0) return nullptr;

    if (K >= pointCount) {
        CVLog::Warning(
                "Cloud %1 has less point than the expected number of classes.");
        return nullptr;
    }

    ccPointCloud* KCloud = nullptr;

    try {
        std::vector<ecvColor::Rgb> clusterCenters;  // K clusters centers
        std::vector<std::size_t>
                clusterIndex;  // index of the cluster the point belongs to

        clusterIndex.resize(pointCount);
        clusterCenters.resize(K);

        // init (regularly sampled) classes centers
        double step = static_cast<double>(pointCount) / K;
        for (unsigned j = 0; j < K; ++j) {
            // TODO: this initialization is pretty biased... To be improved?
            clusterCenters[j] = theCloud->getPointColor(
                    static_cast<unsigned>(std::ceil(step * j)));
        }

        // let's start
        int iteration = 0;
        for (; iteration < maxIterationCount; ++iteration) {
            bool meansHaveMoved = false;

            // assign each point (color) to the nearest cluster
            for (unsigned i = 0; i < pointCount; ++i) {
                const ecvColor::Rgb& color = theCloud->getPointColor(i);

                std::size_t minK = 0;
                int minDistsToMean =
                        std::abs(ColorDistance(color, clusterCenters[minK]));

                // we look for the nearest cluster center
                for (unsigned j = 1; j < K; ++j) {
                    int distToMean =
                            std::abs(ColorDistance(color, clusterCenters[j]));
                    if (distToMean < minDistsToMean) {
                        minDistsToMean = distToMean;
                        minK = j;
                    }
                }

                clusterIndex[i] = minK;
            }

            // update the clusters centers
            std::vector<std::vector<unsigned>> clusters;
            clusters.resize(K);
            for (unsigned i = 0; i < pointCount; ++i) {
                unsigned index = static_cast<unsigned>(
                        clusterIndex[static_cast<std::size_t>(i)]);
                clusters[index].push_back(i);
            }

            CVLog::Print("Iteration " + QString::number(iteration));
            for (unsigned j = 0; j < K; ++j) {
                const std::vector<unsigned>& cluster = clusters[j];
                if (cluster.empty()) {
                    continue;
                }

                ecvColor::Rgb newMean = ComputeAverageColor(*theCloud, cluster);

                if (!meansHaveMoved &&
                    ColorDistance(clusterCenters[j], newMean) != 0) {
                    meansHaveMoved = true;
                }

                clusterCenters[j] = newMean;
            }

            if (!meansHaveMoved) {
                break;
            }
        }

        KCloud = theCloud->cloneThis();
        if (!KCloud) {
            // not enough memory
            return nullptr;
        }
        KCloud->setName("Kmeans clustering: K = " + QString::number(K) +
                        " / it = " + QString::number(iteration));

        // set color for each cluster
        for (unsigned i = 0; i < pointCount; i++) {
            KCloud->setPointColor(i, clusterCenters[clusterIndex[i]]);
        }

    } catch (const std::bad_alloc&) {
        // not enough memory
        return nullptr;
    }

    return KCloud;
}

/**
Algorithm based on k-means for clustering points cloud by its colors
*/
void ColorimetricSegmenter::KmeansClustering() {
    std::vector<ccPointCloud*> clouds =
            ColorimetricSegmenter::getSelectedPointClouds();
    if (clouds.empty()) {
        Q_ASSERT(false);
        return;
    }

    KmeansDlg kmeansDlg(m_app->getMainWindow());
    if (!kmeansDlg.exec()) return;

    assert(kmeansDlg.spinBox_k->value() >= 0);
    unsigned K = static_cast<unsigned>(kmeansDlg.spinBox_k->value());
    int iterationCount = kmeansDlg.spinBox_it->value();

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    for (ccPointCloud* cloud : clouds) {
        ccPointCloud* kcloud =
                ComputeKmeansClustering(cloud, K, iterationCount);
        if (!kcloud) {
            m_app->dispToConsole(QString("[ColorimetricSegmenter] Failed to "
                                         "cluster cloud %1")
                                         .arg(cloud->getName()),
                                 ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
            continue;
        }

        cloud->setEnabled(false);
        if (cloud->getParent()) {
            cloud->getParent()->addChild(kcloud);
        }

        m_app->addToDB(kcloud, false, true, false, true);
        m_app->dispToConsole(
                "[ColorimetricSegmenter] Cloud successfully clustered!",
                ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }

    ShowDurationNow(startTime);
}
