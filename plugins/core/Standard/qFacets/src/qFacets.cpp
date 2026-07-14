// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qFacets.h"

#include <algorithm>
#include <cassert>
#include <exception>

// Local
#include "classificationParamsDlg.h"
#include "disclaimerDialog.h"
#include "facetsClassifier.h"
#include "facetsExportDlg.h"
#include "fastMarchingForFacetExtraction.h"
#include "kdTreeForFacetExtraction.h"
#include "qFacetsCommands.h"
#include "stereogramDlg.h"

// Qt
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QInputDialog>
#include <QLocale>
#include <QMessageBox>
#include <QSettings>
#include <QTextStream>
#include <QWidget>
#include <QtGui>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

// CV_CORE_LIB
#include <CVLog.h>
#include <Neighbourhood.h>

// CV_DB_LIB
#include <ShpDBFFields.h>
#include <ecvFileUtils.h>
#include <ecvGenericPointCloud.h>
#include <ecvHObjectCaster.h>
#include <ecvKdTree.h>
#include <ecvMesh.h>
#include <ecvOctree.h>  //for ComputeAverageNorm
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>
#include <ecvScalarField.h>

// CV_IO_LIB
#include <ShpFilter.h>

// semi-persistent dialog values
static unsigned s_octreeLevel = 8;
static bool s_fmUseRetroProjectionError = false;

static unsigned s_minPointsPerFacet = 10;
static double s_errorMaxPerFacet = 0.2;
static int s_errorMeasureType = 3;  // max dist @ 99 %
static double s_maxEdgeLength = 1.0;

static double s_kdTreeFusionMaxAngle_deg = 20.0;
static double s_kdTreeFusionMaxRelativeDistance = 1.0;

static double s_classifAngleStep = 30.0;
static double s_classifMaxDist = 1.0;

static double s_stereogramAngleStep = 30.0;
static double s_stereogramResolution_deg = 5.0;
static ccPointCloud* s_lastCloud = nullptr;

qFacets::qFacets(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qFacets/info.json"),
      m_doFuseKdTreeCells(nullptr),
      m_fastMarchingExtraction(nullptr),
      m_doExportFacets(nullptr),
      m_doExportFacetsInfo(nullptr),
      m_doClassifyFacetsByAngle(nullptr),
      m_doShowStereogram(nullptr),
      m_stereogramDialog(nullptr) {}

qFacets::~qFacets() {
    if (m_stereogramDialog) {
        try {
            m_stereogramDialog->close();
            delete m_stereogramDialog;
            m_stereogramDialog = nullptr;
        } catch (std::exception& e) {
            CVLog::Warning(
                    tr("closing facets dialog failed! [%1]").arg(e.what()));
        }
    }
}

QList<QAction*> qFacets::getActions() {
    // actions
    if (!m_doFuseKdTreeCells) {
        m_doFuseKdTreeCells = new QAction(tr("Extract facets (Kd-tree)"), this);
        m_doFuseKdTreeCells->setToolTip(
                tr("Detect planar facets by fusing Kd-tree cells"));
        m_doFuseKdTreeCells->setIcon(QIcon(
                QString::fromUtf8(":/CC/plugin/qFacets/images/extractKD.png")));
        // connect signal
        connect(m_doFuseKdTreeCells, &QAction::triggered, this,
                &qFacets::fuseKdTreeCells);
    }

    if (!m_fastMarchingExtraction) {
        m_fastMarchingExtraction =
                new QAction(tr("Extract facets (Fast Marching)"), this);
        m_fastMarchingExtraction->setToolTip(
                tr("Detect planar facets with Fast Marching"));
        m_fastMarchingExtraction->setIcon(QIcon(
                QString::fromUtf8(":/CC/plugin/qFacets/images/extractFM.png")));
        // connect signal
        connect(m_fastMarchingExtraction, &QAction::triggered, this,
                &qFacets::extractFacetsWithFM);
    }

    if (!m_doExportFacets) {
        m_doExportFacets = new QAction(tr("Export facets (SHP)"), this);
        m_doExportFacets->setToolTip(
                tr("Exports one or several facets to a shapefile"));
        m_doExportFacets->setIcon(QIcon(
                QString::fromUtf8(":/CC/plugin/qFacets/images/shpFile.png")));
        // connect signal
        connect(m_doExportFacets, &QAction::triggered, this,
                &qFacets::exportFacets);
    }

    if (!m_doExportFacetsInfo) {
        m_doExportFacetsInfo =
                new QAction(tr("Export facets info (CSV)"), this);
        m_doExportFacetsInfo->setToolTip(
                tr("Exports various information on a set of facets (ASCII CSV "
                   "file)"));
        m_doExportFacetsInfo->setIcon(QIcon(
                QString::fromUtf8(":/CC/plugin/qFacets/images/csvFile.png")));
        // connect signal
        connect(m_doExportFacetsInfo, &QAction::triggered, this,
                &qFacets::exportFacetsInfo);
    }

    if (!m_doClassifyFacetsByAngle) {
        m_doClassifyFacetsByAngle =
                new QAction(tr("Classify facets by orientation"), this);
        m_doClassifyFacetsByAngle->setToolTip(
                tr("Classifies facets based on their orienation (dip & dip "
                   "direction)"));
        m_doClassifyFacetsByAngle->setIcon(QIcon(QString::fromUtf8(
                ":/CC/plugin/qFacets/images/classifIcon.png")));
        // connect signal
        connect(m_doClassifyFacetsByAngle, &QAction::triggered, this,
                [=]() { classifyFacetsByAngle(); });
    }

    if (!m_doShowStereogram) {
        m_doShowStereogram = new QAction(tr("Show stereogram"), this);
        m_doShowStereogram->setToolTip(
                tr("Computes and displays a stereogram (+ interactive "
                   "filtering)"));
        m_doShowStereogram->setIcon(QIcon(QString::fromUtf8(
                ":/CC/plugin/qFacets/images/stereogram.png")));
        // connect signal
        connect(m_doShowStereogram, &QAction::triggered, this,
                &qFacets::showStereogram);
    }

    return QList<QAction*>{
            m_doFuseKdTreeCells,  m_fastMarchingExtraction,  m_doExportFacets,
            m_doExportFacetsInfo, m_doClassifyFacetsByAngle, m_doShowStereogram,
    };
}

void qFacets::onNewSelection(const ccHObject::Container& selectedEntities) {
    if (m_doFuseKdTreeCells)
        m_doFuseKdTreeCells->setEnabled(
                selectedEntities.size() == 1 &&
                selectedEntities.back()->isA(CV_TYPES::POINT_CLOUD));
    if (m_fastMarchingExtraction)
        m_fastMarchingExtraction->setEnabled(
                selectedEntities.size() == 1 &&
                selectedEntities.back()->isA(CV_TYPES::POINT_CLOUD));
    if (m_doExportFacets)
        m_doExportFacets->setEnabled(selectedEntities.size() != 0);
    if (m_doExportFacetsInfo)
        m_doExportFacetsInfo->setEnabled(selectedEntities.size() != 0);
    if (m_doClassifyFacetsByAngle)
        m_doClassifyFacetsByAngle->setEnabled(
                selectedEntities.size() == 1 &&
                selectedEntities.back()->isA(CV_TYPES::HIERARCHY_OBJECT));
    if (m_doShowStereogram)
        m_doShowStereogram->setEnabled(
                selectedEntities.size() == 1 &&
                (selectedEntities.back()->isA(CV_TYPES::HIERARCHY_OBJECT) ||
                 selectedEntities.back()->isA(CV_TYPES::POINT_CLOUD)));
}

void qFacets::extractFacetsWithFM() {
    extractFacets(CellsFusionDlg::ALGO_FAST_MARCHING);
}

void qFacets::fuseKdTreeCells() { extractFacets(CellsFusionDlg::ALGO_KD_TREE); }

void qFacets::extractFacets(CellsFusionDlg::Algorithm algo) {
    // disclaimer accepted?
    if (!ShowDisclaimer(m_app)) return;

    assert(m_app);
    if (!m_app) return;

    // we expect a unique cloud as input
    const ccHObject::Container& selectedEntities = m_app->getSelectedEntities();
    ccPointCloud* pc =
            (m_app->haveOneSelection()
                     ? ccHObjectCaster::ToPointCloud(selectedEntities.back())
                     : nullptr);
    if (!pc) {
        m_app->dispToConsole(tr("Select one and only one point cloud!"),
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    if (algo != CellsFusionDlg::ALGO_FAST_MARCHING &&
        algo != CellsFusionDlg::ALGO_KD_TREE) {
        m_app->dispToConsole(tr("Internal error: invalid algorithm type!"),
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    // first time: we compute the max edge length automatically
    if (s_lastCloud != pc) {
        s_maxEdgeLength =
                static_cast<double>(pc->getOwnBB().getMinBoxDim()) / 50;
        s_minPointsPerFacet = std::max<unsigned>(pc->size() / 100000, 10);
        s_lastCloud = pc;
    }

    CellsFusionDlg fusionDlg(algo, m_app->getMainWindow());
    if (algo == CellsFusionDlg::ALGO_FAST_MARCHING)
        fusionDlg.octreeLevelSpinBox->setCloud(pc);

    fusionDlg.octreeLevelSpinBox->setValue(s_octreeLevel);
    fusionDlg.useRetroProjectionCheckBox->setChecked(
            s_fmUseRetroProjectionError);
    fusionDlg.minPointsPerFacetSpinBox->setValue(s_minPointsPerFacet);
    fusionDlg.errorMeasureComboBox->setCurrentIndex(s_errorMeasureType);
    fusionDlg.maxRMSDoubleSpinBox->setValue(s_errorMaxPerFacet);
    fusionDlg.maxAngleDoubleSpinBox->setValue(s_kdTreeFusionMaxAngle_deg);
    fusionDlg.maxRelativeDistDoubleSpinBox->setValue(
            s_kdTreeFusionMaxRelativeDistance);
    fusionDlg.maxEdgeLengthDoubleSpinBox->setValue(s_maxEdgeLength);
    //"no normal" warning
    fusionDlg.noNormalWarningLabel->setVisible(!pc->hasNormals());

    if (!fusionDlg.exec()) return;

    s_octreeLevel = fusionDlg.octreeLevelSpinBox->value();
    s_fmUseRetroProjectionError =
            fusionDlg.useRetroProjectionCheckBox->isChecked();
    s_minPointsPerFacet = fusionDlg.minPointsPerFacetSpinBox->value();
    s_errorMeasureType = fusionDlg.errorMeasureComboBox->currentIndex();
    s_errorMaxPerFacet = fusionDlg.maxRMSDoubleSpinBox->value();
    s_kdTreeFusionMaxAngle_deg = fusionDlg.maxAngleDoubleSpinBox->value();
    s_kdTreeFusionMaxRelativeDistance =
            fusionDlg.maxRelativeDistDoubleSpinBox->value();
    s_maxEdgeLength = fusionDlg.maxEdgeLengthDoubleSpinBox->value();

    // convert 'errorMeasureComboBox' index to enum
    cloudViewer::DistanceComputationTools::ERROR_MEASURES errorMeasure =
            cloudViewer::DistanceComputationTools::RMS;
    switch (s_errorMeasureType) {
        case 0:
            errorMeasure = cloudViewer::DistanceComputationTools::RMS;
            break;
        case 1:
            errorMeasure =
                    cloudViewer::DistanceComputationTools::MAX_DIST_68_PERCENT;
            break;
        case 2:
            errorMeasure =
                    cloudViewer::DistanceComputationTools::MAX_DIST_95_PERCENT;
            break;
        case 3:
            errorMeasure =
                    cloudViewer::DistanceComputationTools::MAX_DIST_99_PERCENT;
            break;
        case 4:
            errorMeasure = cloudViewer::DistanceComputationTools::MAX_DIST;
            break;
        default:
            assert(false);
            break;
    }

    // create scalar field to host the fusion result
    const char c_defaultSFName[] = "facet indexes";
    int sfIdx = pc->getScalarFieldIndexByName(c_defaultSFName);
    if (sfIdx < 0) sfIdx = pc->addScalarField(c_defaultSFName);
    if (sfIdx < 0) {
        m_app->dispToConsole(
                tr("Couldn't allocate a new scalar field for computing fusion "
                   "labels! Try to free some memory ..."),
                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }
    pc->setCurrentScalarField(sfIdx);

    // computation
    QElapsedTimer eTimer;
    eTimer.start();
    ecvProgressDialog pDlg(true, m_app->getActiveWindow());

    bool success = true;
    if (algo == CellsFusionDlg::ALGO_KD_TREE) {
        // we need a kd-tree
        QElapsedTimer eTimer;
        eTimer.start();
        ccKdTree kdtree(pc);

        if (kdtree.build(s_errorMaxPerFacet / 2, errorMeasure,
                         s_minPointsPerFacet, 1000, &pDlg)) {
            qint64 elapsedTime_ms = eTimer.elapsed();
            m_app->dispToConsole(
                    tr("[qFacets] Kd-tree construction timing: %1 s")
                            .arg(static_cast<double>(elapsedTime_ms) / 1.0e3, 0,
                                 'f', 3),
                    ecvMainAppInterface::STD_CONSOLE_MESSAGE);

            success = ccKdTreeForFacetExtraction::FuseCells(
                    &kdtree, s_errorMaxPerFacet, errorMeasure,
                    s_kdTreeFusionMaxAngle_deg,
                    static_cast<PointCoordinateType>(
                            s_kdTreeFusionMaxRelativeDistance),
                    true, &pDlg);
        } else {
            m_app->dispToConsole(
                    tr("Failed to build Kd-tree! (not enough memory?)"),
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            success = false;
        }
    } else if (algo == CellsFusionDlg::ALGO_FAST_MARCHING) {
        int result = FastMarchingForFacetExtraction::ExtractPlanarFacets(
                pc, static_cast<unsigned char>(s_octreeLevel),
                static_cast<ScalarType>(s_errorMaxPerFacet), errorMeasure,
                s_fmUseRetroProjectionError, &pDlg, pc->getOctree().data());

        success = (result >= 0);
    }

    if (success) {
        pc->setCurrentScalarField(
                sfIdx);  // for
                         // AutoSegmentationTools::extractConnectedComponents

        cloudViewer::ReferenceCloudContainer components;
        if (!cloudViewer::AutoSegmentationTools::extractConnectedComponents(
                    pc, components)) {
            m_app->dispToConsole(tr("Failed to extract fused components! (not "
                                    "enough memory?)"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        } else {
            // we remove the temporary scalar field (otherwise it will be copied
            // to the sub-clouds!)
            ccScalarField* indexSF =
                    static_cast<ccScalarField*>(pc->getScalarField(sfIdx));
            indexSF->link();  // to prevent deletion below
            pc->deleteScalarField(sfIdx);
            sfIdx = -1;

            bool error = false;
            ccHObject* group =
                    CreateFacets(pc, components, s_minPointsPerFacet,
                                 s_maxEdgeLength, false, error, &pDlg);

            if (group) {
                switch (algo) {
                    case CellsFusionDlg::ALGO_KD_TREE:
                        group->setName(
                                group->getName() +
                                tr(" [Kd-tree][error < %1][angle < %2 deg.]")
                                        .arg(s_errorMaxPerFacet)
                                        .arg(s_kdTreeFusionMaxAngle_deg));
                        break;
                    case CellsFusionDlg::ALGO_FAST_MARCHING:
                        group->setName(group->getName() +
                                       tr(" [FM][level %2][error < %1]")
                                               .arg(s_octreeLevel)
                                               .arg(s_errorMaxPerFacet));
                        break;
                    default:
                        break;
                }

                unsigned count = group->getChildrenNumber();
                m_app->dispToConsole(tr("[qFacets] %1 facet(s) where created "
                                        "from cloud '%2'")
                                             .arg(count)
                                             .arg(pc->getName()));

                if (error) {
                    m_app->dispToConsole(
                            tr("Error(s) occurred during the generation of "
                               "facets! Result may be incomplete"),
                            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                } else {
                    // we but back the scalar field
                    if (indexSF) sfIdx = pc->addScalarField(indexSF);
                }

                m_app->addToDB(group);
                group->setRedrawFlagRecursive(true);
            } else if (error) {
                m_app->dispToConsole(tr("An error occurred during the "
                                        "generation of facets!"),
                                     ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            } else {
                m_app->dispToConsole(tr("No facet remains! Check the "
                                        "parameters (min size, etc.)"),
                                     ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
            }
        }
    } else {
        m_app->dispToConsole(tr("An error occurred during the fusion process!"),
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }

    if (sfIdx >= 0) {
        pc->getScalarField(sfIdx)->computeMinAndMax();
#ifdef _DEBUG
        pc->setCurrentDisplayedScalarField(sfIdx);
        pc->showSF(true);
#endif
    }

    m_app->redrawAll();
}

ccHObject* qFacets::CreateFacets(
        ccPointCloud* cloud,
        cloudViewer::ReferenceCloudContainer& components,
        unsigned minPointsPerComponent,
        double maxEdgeLength,
        bool randomColors,
        bool& error,
        cloudViewer::GenericProgressCallback* progress) {
    if (!cloud) {
        error = true;
        return nullptr;
    }

    ccHObject* ccGroup =
            new ccHObject(cloud->getName() + QObject::tr(" [facets]"));
    ccGroup->setDisplay(cloud->getDisplay());
    ccGroup->setVisible(true);

    bool cloudHasNormal = cloud->hasNormals();
    size_t componentCount = components.size();

    cloudViewer::NormalizedProgress nProgress(
            progress, static_cast<unsigned>(componentCount));
    if (progress) {
        progress->setMethodTitle(
                QObject::tr("Facets creation").toUtf8().constData());
        progress->setInfo(QObject::tr("Components: %1")
                                  .arg(componentCount)
                                  .toUtf8()
                                  .constData());
    }

    error = false;
    while (!components.empty()) {
        cloudViewer::ReferenceCloud* compIndexes = components.back();
        components.pop_back();

        if (compIndexes && compIndexes->size() >= minPointsPerComponent) {
            ccPointCloud* facetCloud = cloud->partialClone(compIndexes);
            if (!facetCloud) {
                error = true;
                delete facetCloud;
                facetCloud = nullptr;
            } else {
                ccFacet* facet = ccFacet::Create(
                        facetCloud,
                        static_cast<PointCoordinateType>(maxEdgeLength), true);
                if (facet) {
                    QString facetName =
                            QObject::tr("facet %1 (rms=%2)")
                                    .arg(ccGroup->getChildrenNumber())
                                    .arg(facet->getRMS());
                    facet->setName(facetName);
                    if (facet->getPolygon()) {
                        facet->getPolygon()->enableStippling(false);
                        facet->getPolygon()->showNormals(false);
                    }
                    if (facet->getContour()) {
                        facet->getContour()->copyGlobalShiftAndScale(
                                *facetCloud);
                    }

                    if (cloudHasNormal) {
                        CCVector3 N = ccOctree::ComputeAverageNorm(compIndexes,
                                                                   cloud);

                        if (N.dot(facet->getNormal()) < 0)
                            facet->invertNormal();
                    }

#ifdef _DEBUG
                    facet->showNormalVector(true);
#endif

                    ecvColor::Rgb col, darkCol;
                    if (randomColors) {
                        col = ecvColor::Generator::Random();
                        assert(c_darkColorRatio <= 1.0);
                        darkCol.r = static_cast<ColorCompType>(
                                static_cast<double>(col.r) * c_darkColorRatio);
                        darkCol.g = static_cast<ColorCompType>(
                                static_cast<double>(col.g) * c_darkColorRatio);
                        darkCol.b = static_cast<ColorCompType>(
                                static_cast<double>(col.b) * c_darkColorRatio);
                    } else {
                        CCVector3 N = facet->getNormal();
                        PointCoordinateType dip, dipDir;
                        ccNormalVectors::ConvertNormalToDipAndDipDir(N, dip,
                                                                     dipDir);
                        FacetsClassifier::GenerateSubfamilyColor(
                                col, dip, dipDir, 0, 1, &darkCol);
                    }
                    facet->setColor(col);
                    if (facet->getContour()) {
                        facet->getContour()->setColor(darkCol);
                        facet->getContour()->setWidth(2);
                    }
                    ccGroup->addChild(facet);
                }
            }

            delete compIndexes;
            compIndexes = nullptr;
        }

        if (progress) {
            nProgress.oneStep();
            if (progress->isCancelRequested()) {
                error = true;
                break;
            }
        }
    }

    if (ccGroup->getChildrenNumber() == 0) {
        delete ccGroup;
        ccGroup = nullptr;
    }

    return ccGroup;
}

void qFacets::getFacetsInCurrentSelection(FacetSet& facets) const {
    facets.clear();

    // look for potential facets
    for (ccHObject* entity : m_app->getSelectedEntities()) {
        if (entity->isA(CV_TYPES::FACET)) {
            ccFacet* facet = static_cast<ccFacet*>(entity);
            if (facet->getContour())  // if no contour, we won't be able to save
                                      // it?!
                facets.insert(facet);
        } else  // if (entity->isA(CV_TYPES::HIERARCHY_OBJECT)) //recursively
                // tests group's children
        {
            ccHObject::Container childFacets;
            entity->filterChildren(childFacets, true, CV_TYPES::FACET);

            for (ccHObject* childFacet : childFacets) {
                ccFacet* facet = static_cast<ccFacet*>(childFacet);
                if (facet->getContour())  // if no contour, we won't be able to
                                          // save it?!
                {
                    facets.insert(facet);
                }
            }
        }
    }
}

// standard meta-data for the qFacets plugin
struct FacetMetaData {
    int facetIndex;
    CCVector3 center;
    CCVector3d globalCenter;
    CCVector3 normal;
    double surface;
    int dip_deg;
    int dipDir_deg;
    double rms;
    int familyIndex;
    int subfamilyIndex;

    FacetMetaData()
        : facetIndex(-1),
          center(0, 0, 0),
          globalCenter(0, 0, 0),
          normal(0, 0, 1),
          surface(0.0),
          dip_deg(0),
          dipDir_deg(0),
          rms(0.0),
          familyIndex(0),
          subfamilyIndex(0) {}
};

void GetFacetMetaData(const ccFacet* facet, FacetMetaData& data) {
    // try to get the facet index from the facet name!
    {
        QStringList tokens =
                facet->getName().split(" ", QtCompat::SkipEmptyParts);
        if (tokens.size() > 1 && tokens[0] == QString("facet")) {
            bool ok = true;
            data.facetIndex = tokens[1].toInt(&ok);
            if (!ok) data.facetIndex = -1;
        }
    }

    data.center = facet->getCenter();

    ccHObject::Container clouds;
    if (facet->filterChildren(clouds, false, CV_TYPES::POINT_CLOUD, true) !=
        0) {
        data.globalCenter =
                static_cast<const ccGenericPointCloud*>(clouds.front())
                        ->toGlobal3d(data.center);
    } else {
        assert(false);
        data.globalCenter = data.center.toDouble();
    }

    data.normal = facet->getNormal();
    data.surface = facet->getSurface();
    data.rms = facet->getRMS();

    // family and subfamily indexes
    QVariant fi = facet->getMetaData(s_OriFamilyKey);
    if (fi.isValid()) data.familyIndex = fi.toInt();
    QVariant sfi = facet->getMetaData(s_OriSubFamilyKey);
    if (sfi.isValid()) data.subfamilyIndex = sfi.toInt();

    // compute dip direction & dip
    {
        PointCoordinateType dipDir = 0, dip = 0;
        ccNormalVectors::ConvertNormalToDipAndDipDir(data.normal, dip, dipDir);
        data.dipDir_deg = static_cast<int>(dipDir);
        data.dip_deg = static_cast<int>(dip);
    }
}

// helper: computes a facet horizontal and vertical extensions
void ComputeFacetExtensions(CCVector3& N,
                            ccPolyline* facetContour,
                            double& horizExt,
                            double& vertExt) {
    // horizontal and vertical extensions
    horizExt = vertExt = 0;

    cloudViewer::GenericIndexedCloudPersist* vertCloud =
            facetContour->getAssociatedCloud();
    if (vertCloud) {
        // oriRotMat.applyRotation(N); //DGM: oriRotMat is only for display!
        // we assume that at this point the "up" direction is always (0,0,1)
        CCVector3 Xf(1, 0, 0), Yf(0, 1, 0);
        // we get the horizontal vector on the plane
        CCVector3 D = CCVector3(0, 0, 1).cross(N);
        if (cloudViewer::GreaterThanEpsilon(
                    D.norm2()))  // otherwise the facet is horizontal!
        {
            Yf = D;
            Yf.normalize();
            Xf = N.cross(Yf);
        }

        const CCVector3* G =
                cloudViewer::Neighbourhood(vertCloud).getGravityCenter();

        ccBBox box;
        for (unsigned i = 0; i < vertCloud->size(); ++i) {
            const CCVector3 P = *(vertCloud->getPoint(i)) - *G;
            CCVector3 p(P.dot(Xf), P.dot(Yf), 0);
            box.add(p);
        }

        vertExt = box.getDiagVec().x;
        horizExt = box.getDiagVec().y;
    }
}

void qFacets::exportFacets() {
    assert(m_app);
    if (!m_app) return;

    if (!ShowDisclaimer(m_app)) return;

    FacetSet facets;
    getFacetsInCurrentSelection(facets);

    if (facets.empty()) {
        m_app->dispToConsole(
                tr("Couldn't find any facet in the current selection!"),
                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    FacetsExportDlg fDlg(FacetsExportDlg::SHAPE_FILE_IO,
                         m_app->getMainWindow());

    fDlg.coordsInCSVBox->setVisible(false);

    QSettings settings;
    settings.beginGroup("qFacets");
    QString facetsSavePath =
            settings.value("exportPath", ecvFileUtils::defaultDocPath())
                    .toString();
    fDlg.destinationPathLineEdit->setText(facetsSavePath +
                                          QString("/facets.shp"));

    if (!fDlg.exec()) return;

    QString filename = fDlg.destinationPathLineEdit->text();
    settings.setValue("exportPath", QFileInfo(filename).absolutePath());

    if (QFile(filename).exists()) {
        if (QMessageBox::warning(
                    m_app->getMainWindow(), tr("File already exists!"),
                    tr("File already exists! Are you sure you want to "
                       "overwrite it?"),
                    QMessageBox::Yes, QMessageBox::No) == QMessageBox::No)
            return;
    }

    bool useNativeOrientation = fDlg.nativeOriRadioButton->isChecked();
    bool useGlobalOrientation = fDlg.verticalOriRadioButton->isChecked();
    bool useCustomOrientation = fDlg.customOriRadioButton->isChecked();

    double nX = 0.0;
    double nY = 0.0;
    double nZ = 1.0;

    if (useCustomOrientation) {
        nX = fDlg.nXLineEdit->text().toDouble();
        nY = fDlg.nYLineEdit->text().toDouble();
        nZ = fDlg.nZLineEdit->text().toDouble();
    }

    if (!ExecuteExportFacets(facets, filename, useNativeOrientation,
                             useGlobalOrientation, useCustomOrientation, nX, nY,
                             nZ, false)) {
        m_app->dispToConsole(tr("ExportFacets failed for some reason"),
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }
}

void qFacets::showStereogram() {
    assert(m_app);
    if (!m_app) return;

    // disclaimer accepted?
    if (!ShowDisclaimer(m_app)) return;

    // we expect a facet group or a cloud
    const ccHObject::Container& selectedEntities = m_app->getSelectedEntities();
    if (!m_app->haveOneSelection() ||
        (!selectedEntities.back()->isA(CV_TYPES::HIERARCHY_OBJECT) &&
         !selectedEntities.back()->isA(CV_TYPES::POINT_CLOUD))) {
        m_app->dispToConsole(tr("Select a group of facets or a point cloud!"));
        return;
    }

    StereogramParamsDlg stereogramParamsDlg(m_app->getMainWindow());
    stereogramParamsDlg.angleStepDoubleSpinBox->setValue(s_stereogramAngleStep);
    stereogramParamsDlg.resolutionDoubleSpinBox->setValue(
            s_stereogramResolution_deg);
    if (!stereogramParamsDlg.exec()) return;

    s_stereogramAngleStep = stereogramParamsDlg.angleStepDoubleSpinBox->value();
    s_stereogramResolution_deg =
            stereogramParamsDlg.resolutionDoubleSpinBox->value();

    if (!m_stereogramDialog) {
        m_stereogramDialog = new StereogramDialog(m_app);
    }
    if (m_stereogramDialog->init(s_stereogramAngleStep, selectedEntities.back(),
                                 s_stereogramResolution_deg)) {
        m_stereogramDialog->show();
        m_stereogramDialog->raise();
    }
}

void qFacets::classifyFacetsByAngle() {
    assert(m_app);
    if (!m_app) return;

    // disclaimer accepted?
    if (!ShowDisclaimer(m_app)) return;

    // we expect a facet group
    const ccHObject::Container& selectedEntities = m_app->getSelectedEntities();
    if (!m_app->haveOneSelection() ||
        !selectedEntities.back()->isA(CV_TYPES::HIERARCHY_OBJECT)) {
        m_app->dispToConsole(tr("Select a group of facets!"));
        return;
    }

    ClassificationParamsDlg classifParamsDlg(m_app->getMainWindow());
    classifParamsDlg.angleStepDoubleSpinBox->setValue(s_classifAngleStep);
    classifParamsDlg.maxDistDoubleSpinBox->setValue(s_classifMaxDist);
    if (!classifParamsDlg.exec()) return;

    s_classifAngleStep = classifParamsDlg.angleStepDoubleSpinBox->value();
    s_stereogramAngleStep =
            s_classifAngleStep;  // we automatically copy it to the stereogram's
                                 // equivalent parameter
    s_classifMaxDist = classifParamsDlg.maxDistDoubleSpinBox->value();

    ccHObject* group = selectedEntities.back();
    classifyFacetsByAngle(group, s_classifAngleStep, s_classifMaxDist);
}

void qFacets::classifyFacetsByAngle(ccHObject* group,
                                    double angleStep_deg,
                                    double maxDist) {
    assert(m_app);
    if (!m_app) return;

    assert(group);

    if (group->isA(CV_TYPES::HIERARCHY_OBJECT)) {
        if (group->getParent()) {
            m_app->removeFromDB(group, false);
        }

        bool success =
                FacetsClassifier::ByOrientation(group, angleStep_deg, maxDist);
        m_app->addToDB(group);

        if (!success) {
            m_app->dispToConsole(tr("An error occurred while classifying the "
                                    "facets! (not enough memory?)"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
    }
}

void qFacets::exportFacetsInfo() {
    assert(m_app);
    if (!m_app) return;

    if (!ShowDisclaimer(m_app)) return;

    FacetSet facets;
    getFacetsInCurrentSelection(facets);

    if (facets.empty()) {
        m_app->dispToConsole(
                tr("Couldn't find any facet in the current selection!"),
                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    FacetsExportDlg fDlg(FacetsExportDlg::ASCII_FILE_IO,
                         m_app->getMainWindow());
    fDlg.orientationGroupBox->setEnabled(false);

    QSettings settings;
    settings.beginGroup("qFacets");
    QString facetsSavePath =
            settings.value("exportPath", ecvFileUtils::defaultDocPath())
                    .toString();
    fDlg.destinationPathLineEdit->setText(facetsSavePath +
                                          QString("/facets.csv"));

    if (!fDlg.exec()) return;

    QString filename = fDlg.destinationPathLineEdit->text();
    settings.setValue("exportPath", QFileInfo(filename).absolutePath());

    bool coordsInCsv = fDlg.coordsInCSVCheckBox->isChecked();

    bool useNativeOrientation = fDlg.nativeOriRadioButton->isChecked();
    bool useGlobalOrientation = fDlg.verticalOriRadioButton->isChecked();
    bool useCustomOrientation = fDlg.customOriRadioButton->isChecked();

    double nX = 0.0;
    double nY = 0.0;
    double nZ = 1.0;

    if (useCustomOrientation) {
        nX = fDlg.nXLineEdit->text().toDouble();
        nY = fDlg.nYLineEdit->text().toDouble();
        nZ = fDlg.nZLineEdit->text().toDouble();
    }

    if (QFile(filename).exists()) {
        if (QMessageBox::warning(m_app->getMainWindow(), tr("Overwrite"),
                                 tr("File already exists! Are you sure you "
                                    "want to overwrite it?"),
                                 QMessageBox::Yes,
                                 QMessageBox::No) == QMessageBox::No)
            return;
    }

    if (!ExecuteExportFacetsInfo(facets, filename, coordsInCsv,
                                 useNativeOrientation, useGlobalOrientation,
                                 useCustomOrientation, nX, nY, nZ, false)) {
        m_app->dispToConsole(tr("ExportFacetsInfo failed for some reason"),
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }
}

ccHObject* qFacets::ExecuteFacetExtraction(ccPointCloud* pc,
                                           const FacetsParams& params,
                                           bool& error,
                                           ecvProgressDialog* progressDlg) {
    error = false;
    if (!pc) {
        error = true;
        return nullptr;
    }

    if (params.algo != CellsFusionDlg::ALGO_FAST_MARCHING &&
        params.algo != CellsFusionDlg::ALGO_KD_TREE) {
        error = true;
        return nullptr;
    }

    cloudViewer::GenericProgressCallback* progress = progressDlg;

    const char c_defaultSFName[] = "facet indexes";
    int sfIdx = pc->getScalarFieldIndexByName(c_defaultSFName);
    if (sfIdx < 0) sfIdx = pc->addScalarField(c_defaultSFName);
    if (sfIdx < 0) {
        CVLog::Error(QObject::tr(
                "Couldn't allocate a new scalar field for computing "
                "fusion labels!"));
        error = true;
        return nullptr;
    }
    pc->setCurrentScalarField(sfIdx);

    const cloudViewer::DistanceComputationTools::ERROR_MEASURES errorMeasure =
            params.errorMeasure;

    bool success = true;
    if (params.algo == CellsFusionDlg::ALGO_KD_TREE) {
        if (progress) {
            progress->setMethodTitle("Building Kd-tree...");
        }
        ccKdTree kdtree(pc);

        if (kdtree.build(params.errorMaxPerFacet / 2, errorMeasure,
                         params.minPointsPerFacet, 1000, progress)) {
            success = ccKdTreeForFacetExtraction::FuseCells(
                    &kdtree, params.errorMaxPerFacet, errorMeasure,
                    params.kdTreeFusionMaxAngleDeg,
                    static_cast<PointCoordinateType>(
                            params.kdTreeFusionMaxRelativeDistance),
                    true, progress);
        } else {
            CVLog::Error(QObject::tr(
                    "Failed to build Kd-tree! (not enough memory?)"));
            success = false;
        }
    } else if (params.algo == CellsFusionDlg::ALGO_FAST_MARCHING) {
        int result = FastMarchingForFacetExtraction::ExtractPlanarFacets(
                pc, static_cast<unsigned char>(params.octreeLevel),
                static_cast<ScalarType>(params.errorMaxPerFacet), errorMeasure,
                params.useRetroProjectionError, progress,
                pc->getOctree().data());

        success = (result >= 0);
    }

    ccHObject* group = nullptr;
    if (success) {
        pc->setCurrentScalarField(sfIdx);

        cloudViewer::ReferenceCloudContainer components;
        if (!cloudViewer::AutoSegmentationTools::extractConnectedComponents(
                    pc, components)) {
            CVLog::Error(QObject::tr(
                    "Failed to extract fused components! (not enough "
                    "memory?)"));
            error = true;
        } else {
            ccScalarField* indexSF =
                    static_cast<ccScalarField*>(pc->getScalarField(sfIdx));
            if (!indexSF) {
                assert(false);
                error = true;
                return nullptr;
            }

            indexSF->link();
            pc->deleteScalarField(sfIdx);
            sfIdx = -1;

            group = CreateFacets(pc, components, params.minPointsPerFacet,
                                 params.maxEdgeLength, false, error, progress);

            if (!error) {
                if (indexSF) sfIdx = pc->addScalarField(indexSF);
            }
            indexSF->release();
        }
    } else {
        CVLog::Error(
                QObject::tr("An error occurred during the fusion process!"));
        error = true;
    }

    if (sfIdx >= 0) {
        pc->getScalarField(sfIdx)->computeMinAndMax();
#ifdef _DEBUG
        pc->setCurrentDisplayedScalarField(sfIdx);
        pc->showSF(true);
#endif
    }

    return group;
}

bool qFacets::ExecuteExportFacets(const FacetSet& facets,
                                  const QString& filename,
                                  bool useNativeOrientation,
                                  bool useGlobalOrientation,
                                  bool useCustomOrientation,
                                  double nX,
                                  double nY,
                                  double nZ,
                                  bool silentMode) {
    if (facets.empty()) {
        if (!silentMode) {
            CVLog::Error(QObject::tr("No facets to export info to shape file"));
        }
        return false;
    }

    IntegerDBFField facetIndex(QObject::tr("index"));
    DoubleDBFField facetSurface(QObject::tr("surface"));
    DoubleDBFField facetRMS(QObject::tr("rms"));
    IntegerDBFField facetDipDir(QObject::tr("dip_dir"));
    IntegerDBFField facetDip(QObject::tr("dip"));
    IntegerDBFField familyIndex(QObject::tr("family_ind"));
    IntegerDBFField subfamilyIndex(QObject::tr("subfam_ind"));
    DoubleDBFField3D facetNormal(QObject::tr("normal"));
    DoubleDBFField3D facetBarycenter(QObject::tr("center"));
    DoubleDBFField3D facetGlobalBarycenter(QObject::tr("GlbCtr"));
    DoubleDBFField horizExtension(QObject::tr("horiz_ext"));
    DoubleDBFField vertExtension(QObject::tr("vert_ext"));
    DoubleDBFField surfaceExtension(QObject::tr("surf_ext"));

    size_t facetCount = facets.size();
    try {
        facetIndex.values.reserve(facetCount);
        facetSurface.values.reserve(facetCount);
        facetRMS.values.reserve(facetCount);
        facetDipDir.values.reserve(facetCount);
        facetDip.values.reserve(facetCount);
        familyIndex.values.reserve(facetCount);
        subfamilyIndex.values.reserve(facetCount);
        facetNormal.values.reserve(facetCount);
        facetBarycenter.values.reserve(facetCount);
        facetGlobalBarycenter.values.reserve(facetCount);
        horizExtension.values.reserve(facetCount);
        vertExtension.values.reserve(facetCount);
        surfaceExtension.values.reserve(facetCount);
    } catch (const std::bad_alloc&) {
        if (!silentMode) {
            CVLog::Error(QObject::tr("Not enough memory!"));
        }
        return false;
    }

    ccHObject toSave(QObject::tr("facets"));

    ccGLMatrix oriRotMat =
            CalcOriRotMat(facets, useNativeOrientation, useGlobalOrientation,
                          useCustomOrientation, nX, nY, nZ);

    for (ccFacet* facet : facets) {
        ccPolyline* poly = facet->getContour();
        if (!poly) {
            continue;
        }

        if (!useNativeOrientation) {
            cloudViewer::GenericIndexedCloudPersist* vertices =
                    poly->getAssociatedCloud();
            if (!vertices || vertices->size() < 3) continue;

            ccPolyline* newPoly = new ccPolyline(*poly);
            ccPointCloud* pc = (newPoly ? dynamic_cast<ccPointCloud*>(
                                                  newPoly->getAssociatedCloud())
                                        : nullptr);
            if (pc) {
                pc->applyGLTransformation_recursive(&oriRotMat);
            } else {
                if (!silentMode) {
                    CVLog::Warning(
                            QObject::tr("Failed to change the orientation of "
                                        "polyline '%1'! (not enough memory)")
                                    .arg(poly->getName()));
                }
                delete newPoly;
                continue;
            }

            newPoly->set2DMode(true);
            poly = newPoly;
        }

        toSave.addChild(poly, useNativeOrientation
                                      ? ccHObject::DP_NONE
                                      : ccHObject::DP_PARENT_OF_OTHER);

        FacetMetaData data;
        GetFacetMetaData(facet, data);

        double horizExt = 0, vertExt = 0;
        ComputeFacetExtensions(data.normal, poly, horizExt, vertExt);

        facetIndex.values.push_back(data.facetIndex);
        facetSurface.values.push_back(data.surface);
        facetRMS.values.push_back(data.rms);
        facetDipDir.values.push_back(data.dipDir_deg);
        facetDip.values.push_back(data.dip_deg);
        familyIndex.values.push_back(data.familyIndex);
        subfamilyIndex.values.push_back(data.subfamilyIndex);
        facetNormal.values.push_back(data.normal.toDouble());
        facetBarycenter.values.push_back(data.center.toDouble());
        facetGlobalBarycenter.values.push_back(data.globalCenter);
        vertExtension.values.push_back(vertExt);
        horizExtension.values.push_back(horizExt);
        surfaceExtension.values.push_back(horizExt * vertExt);
    }

    if (toSave.getChildrenNumber() == 0) {
        return false;
    }

    std::vector<GenericDBFField*> fields;
    fields.push_back(&facetIndex);
    fields.push_back(&facetBarycenter);
    fields.push_back(&facetGlobalBarycenter);
    fields.push_back(&facetNormal);
    fields.push_back(&facetRMS);
    fields.push_back(&horizExtension);
    fields.push_back(&vertExtension);
    fields.push_back(&surfaceExtension);
    fields.push_back(&facetSurface);
    fields.push_back(&facetDipDir);
    fields.push_back(&facetDip);
    fields.push_back(&familyIndex);
    fields.push_back(&subfamilyIndex);
    ShpFilter filter;
    filter.treatClosedPolylinesAsPolygons(true);
    ShpFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;
    if (filter.saveToFile(&toSave, fields, filename, saveParams) ==
        CC_FERR_NO_ERROR) {
        if (!silentMode) {
            CVLog::Print(QObject::tr("[qFacets] File '%1' successfully saved")
                                 .arg(filename));
        }
        return true;
    }

    if (!silentMode) {
        CVLog::Warning(QObject::tr("[qFacets] Failed to save file '%1'!")
                               .arg(filename));
    }
    return false;
}

ccGLMatrix qFacets::CalcOriRotMat(const FacetSet& facets,
                                  bool useNativeOrientation,
                                  bool useGlobalOrientation,
                                  bool useCustomOrientation,
                                  double nX,
                                  double nY,
                                  double nZ) {
    CCVector3 X(1, 0, 0);
    CCVector3 Y(0, 1, 0);
    CCVector3 Z(0, 0, 1);

    if (!useNativeOrientation) {
        if (useCustomOrientation) {
            Z = CCVector3(static_cast<PointCoordinateType>(nX),
                          static_cast<PointCoordinateType>(nY),
                          static_cast<PointCoordinateType>(nZ));
            Z.normalize();
        } else if (useGlobalOrientation) {
            CCVector3d Nsum(0, 0, 0);
            for (const ccFacet* facet : facets) {
                double surf = facet->getSurface();
                CCVector3 N = facet->getNormal();
                Nsum.x += static_cast<double>(N.x) * surf;
                Nsum.y += static_cast<double>(N.y) * surf;
                Nsum.z += static_cast<double>(N.z) * surf;
            }
            Nsum.normalize();

            Z = CCVector3(static_cast<PointCoordinateType>(Nsum.x),
                          static_cast<PointCoordinateType>(Nsum.y),
                          static_cast<PointCoordinateType>(Nsum.z));
        }

        CCVector3 D = Z.cross(CCVector3(0, 0, 1));
        if (cloudViewer::GreaterThanEpsilon(D.norm2())) {
            X = -D;
            X.normalize();
            Y = Z.cross(X);
        }
    }

    CCVector3 C(0, 0, 0);
    {
        double weightSum = 0;
        for (const ccFacet* facet : facets) {
            double surf = facet->getSurface();
            CCVector3 Ci = facet->getCenter();
            C += Ci * static_cast<PointCoordinateType>(surf);
            weightSum += surf;
        }
        if (weightSum) C /= static_cast<PointCoordinateType>(weightSum);
    }

    ccGLMatrix oriRotMat;
    oriRotMat.toIdentity();
    if (!useNativeOrientation) {
        oriRotMat.getColumn(0)[0] = static_cast<float>(X.x);
        oriRotMat.getColumn(0)[1] = static_cast<float>(X.y);
        oriRotMat.getColumn(0)[2] = static_cast<float>(X.z);
        oriRotMat.getColumn(1)[0] = static_cast<float>(Y.x);
        oriRotMat.getColumn(1)[1] = static_cast<float>(Y.y);
        oriRotMat.getColumn(1)[2] = static_cast<float>(Y.z);
        oriRotMat.getColumn(2)[0] = static_cast<float>(Z.x);
        oriRotMat.getColumn(2)[1] = static_cast<float>(Z.y);
        oriRotMat.getColumn(2)[2] = static_cast<float>(Z.z);
        oriRotMat.invert();

        ccGLMatrix transMat;
        transMat.setTranslation(-C);
        oriRotMat = oriRotMat * transMat;
        oriRotMat.setTranslation(oriRotMat.getTranslationAsVec3D() + C);
    }

    return oriRotMat;
}

bool qFacets::ExecuteExportFacetsInfo(const FacetSet& facets,
                                      const QString& filename,
                                      bool coordsInCsv,
                                      bool useNativeOrientation,
                                      bool useGlobalOrientation,
                                      bool useCustomOrientation,
                                      double nX,
                                      double nY,
                                      double nZ,
                                      bool silentMode) {
    if (facets.empty()) {
        if (!silentMode) {
            CVLog::Error(QObject::tr("No facets to export info to csv"));
        }
        return false;
    }

    QFile outFile(filename);
    if (!outFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (!silentMode) {
            CVLog::Error(QObject::tr("Failed to open file for writing: %1")
                                 .arg(filename));
        }
        return false;
    }

    QTextStream outStream(&outFile);
    outStream << "Index;";
    outStream << " CenterX;";
    outStream << " CenterY;";
    outStream << " CenterZ;";
    outStream << " GlobalCenterX;";
    outStream << " GlobalCenterY;";
    outStream << " GlobalCenterZ;";
    outStream << " NormalX;";
    outStream << " NormalY;";
    outStream << " NormalZ;";
    outStream << " RMS;";
    outStream << " Horiz_ext;";
    outStream << " Vert_ext;";
    outStream << " Surf_ext;";
    outStream << " Surface;";
    outStream << " Dip dir.;";
    outStream << " Dip;";
    outStream << " Family ind.;";
    outStream << " Subfamily ind.;";
    if (coordsInCsv) {
        outStream << " Geom;";
    }
    outStream << " \n";

    ccGLMatrix oriRotMat =
            CalcOriRotMat(facets, useNativeOrientation, useGlobalOrientation,
                          useCustomOrientation, nX, nY, nZ);

    for (ccFacet* facet : facets) {
        FacetMetaData data;
        GetFacetMetaData(facet, data);

        double horizExt = 0, vertExt = 0;
        ComputeFacetExtensions(data.normal, facet->getContour(), horizExt,
                               vertExt);

        outStream << data.facetIndex << ";";
        outStream << data.center.x << ";" << data.center.y << ";"
                  << data.center.z << ";";
        outStream << data.globalCenter.x << ";" << data.globalCenter.y << ";"
                  << data.globalCenter.z << ";";
        outStream << data.normal.x << ";" << data.normal.y << ";"
                  << data.normal.z << ";";
        outStream << data.rms << ";";
        outStream << horizExt << ";";
        outStream << vertExt << ";";
        outStream << horizExt * vertExt << ";";
        outStream << data.surface << ";";
        outStream << data.dipDir_deg << ";";
        outStream << data.dip_deg << ";";
        outStream << data.familyIndex << ";";
        outStream << data.subfamilyIndex << ";";
        if (coordsInCsv) {
            ccPolyline* poly = facet->getContour();
            if (!useNativeOrientation) {
                cloudViewer::GenericIndexedCloudPersist* vertices =
                        poly->getAssociatedCloud();
                if (!vertices || vertices->size() < 3) {
                    continue;
                }

                ccPolyline* newPoly = new ccPolyline(*poly);
                ccPointCloud* pc =
                        (newPoly ? dynamic_cast<ccPointCloud*>(
                                           newPoly->getAssociatedCloud())
                                 : nullptr);
                if (pc) {
                    pc->applyGLTransformation_recursive(&oriRotMat);
                } else {
                    if (!silentMode) {
                        CVLog::Warning(
                                QObject::tr(
                                        "Failed to change the orientation "
                                        "of polyline '%1'! (not enough memory)")
                                        .arg(poly->getName()));
                    }
                    delete newPoly;
                    continue;
                }

                newPoly->set2DMode(true);
                poly = newPoly;
            }
            QString polygonz = PolylineCoordsToWKT_POLYGONZ(poly, 3);
            outStream << "\"" << polygonz << "\"" << ";";
        }
        outStream << "\n";
    }

    outFile.close();

    if (!silentMode) {
        CVLog::Print(QObject::tr("[qFacets] File '%1' successfully saved")
                             .arg(filename));
    }
    return true;
}

QString qFacets::PolylineCoordsToWKT_POLYGONZ(const ccPolyline* polyline,
                                              unsigned int precision) {
    if (!polyline) {
        return QString("Error: Polyline object is null.");
    }

    const unsigned pointCount = polyline->size();
    if (pointCount < 3) {
        return QString("Invalid WKT input: POLYGON Z requires min 4 points. "
                       "Found %1.")
                .arg(pointCount);
    }

    QString wkt = "POLYGON Z ((";
    QLocale locale(QLocale::C);

    for (unsigned int i = 0; i < pointCount; ++i) {
        CCVector3d p = polyline->getPoint(i)->toDouble();

        if (i > 0) {
            wkt.append(", ");
        }

        wkt.append(locale.toString(p.x, 'f', static_cast<int>(precision)));
        wkt.append(" ");
        wkt.append(locale.toString(p.y, 'f', static_cast<int>(precision)));
        wkt.append(" ");
        wkt.append(locale.toString(p.z, 'f', static_cast<int>(precision)));
    }

    CCVector3d p = polyline->getPoint(0)->toDouble();
    wkt.append(", ");
    wkt.append(locale.toString(p.x, 'f', static_cast<int>(precision)));
    wkt.append(" ");
    wkt.append(locale.toString(p.y, 'f', static_cast<int>(precision)));
    wkt.append(" ");
    wkt.append(locale.toString(p.z, 'f', static_cast<int>(precision)));
    wkt.append("))");

    return wkt;
}

void qFacets::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) {
        assert(false);
        return;
    }
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandFacets));
}
