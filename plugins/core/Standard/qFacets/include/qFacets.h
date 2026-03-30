// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "cellsFusionDlg.h"

// Qt
#include <QObject>

// qCC
#include "ecvStdPluginInterface.h"

// cloudViewer
#include <AutoSegmentationTools.h>
#include <DistanceComputationTools.h>
#include <ReferenceCloud.h>

// System
#include <unordered_set>

class QAction;
class ccCommandLineInterface;
class ccHObject;
class ccPointCloud;
class ccPolyline;
class ccFacet;
class ecvProgressDialog;

//! Facet detection plugin (BRGM)
/** BRGM: BUREAU DE RECHERCHES GEOLOGIQUES ET MINIERES - http://www.brgm.fr/
 **/
class qFacets : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qFacets" FILE
                          "../info.json")

public:
    //! Default constructor
    qFacets(QObject* parent = nullptr);

    //! Destructor
    virtual ~qFacets();

    // inherited from ccStdPluginInterface
    virtual void onNewSelection(
            const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;
    virtual void registerCommands(ccCommandLineInterface* cmd) override;

    //! Set of facets (pointers)
    typedef std::unordered_set<ccFacet*> FacetSet;

    //! Parameters for headless (CLI) facet extraction
    struct FacetsParams {
        bool extractFacets = false;
        CellsFusionDlg::Algorithm algo = CellsFusionDlg::ALGO_KD_TREE;
        double kdTreeFusionMaxAngleDeg = 20.0;
        double kdTreeFusionMaxRelativeDistance = 1.0;
        unsigned octreeLevel = 8;
        bool useRetroProjectionError = false;
        double errorMaxPerFacet = 0.2;
        unsigned minPointsPerFacet = 10;
        double maxEdgeLength = 1.0;
        cloudViewer::DistanceComputationTools::ERROR_MEASURES errorMeasure =
                cloudViewer::DistanceComputationTools::MAX_DIST_99_PERCENT;
        bool classifyFacetsByAngle = false;
        double classifAngleStep = 30.0;
        double classifMaxDist = 1.0;
        bool exportFacets = false;
        QString shapeFilename = "facets.shp";
        bool useNativeOrientation = true;
        bool useGlobalOrientation = false;
        bool useCustomOrientation = false;
        double nX = 0, nY = 0, nZ = 1;
        bool exportFacetsInfo = false;
        QString csvFilename = "facets.csv";
        bool coordsInCsv = false;
    };

    //! Headless facet extraction (used by CLI command)
    static ccHObject* ExecuteFacetExtraction(
            ccPointCloud* cloud,
            const FacetsParams& params,
            bool& error,
            ecvProgressDialog* progressDlg = nullptr);

    //! Headless facet export to shapefile
    static bool ExecuteExportFacets(const FacetSet& facets,
                                    const QString& filename,
                                    bool useNativeOrientation,
                                    bool useGlobalOrientation,
                                    bool useCustomOrientation,
                                    double nX,
                                    double nY,
                                    double nZ,
                                    bool silentMode);

    //! Headless facet info export to CSV
    static bool ExecuteExportFacetsInfo(const FacetSet& facets,
                                        const QString& filename,
                                        bool coordsInCsv,
                                        bool useNativeOrientation,
                                        bool useGlobalOrientation,
                                        bool useCustomOrientation,
                                        double nX,
                                        double nY,
                                        double nZ,
                                        bool silentMode);

protected slots:

    //! Fuses the cells of a kd-tree to produces planar facets
    void fuseKdTreeCells();

    //! Uses Fast Marching to detect planar facets
    void extractFacetsWithFM();

    //! Exports facets (as shapefiles)
    void exportFacets();

    //! Exports statistics on a set of facets
    void exportFacetsInfo();

    //! Classifies facets by orientation
    void classifyFacetsByAngle();

    //! Displays the selected entity stereogram
    void showStereogram();

protected:
    //! Uses the given algorithm to detect planar facets
    void extractFacets(CellsFusionDlg::Algorithm algo);

    //! Creates facets from components
    ccHObject* createFacets(ccPointCloud* cloud,
                            cloudViewer::ReferenceCloudContainer& components,
                            unsigned minPointsPerComponent,
                            double maxEdgeLength,
                            bool randomColors,
                            bool& error);

    //! Returns all the facets in the current selection
    void getFacetsInCurrentSelection(FacetSet& facets) const;

    //! Classifies facets by orientation
    void classifyFacetsByAngle(ccHObject* group,
                               double angleStep_deg,
                               double maxDist);

    //! Associated action
    QAction* m_doFuseKdTreeCells;
    //! Associated action
    QAction* m_fastMarchingExtraction;
    //! Associated action
    QAction* m_doExportFacets;
    //! Associated action
    QAction* m_doExportFacetsInfo;
    //! Associated action
    QAction* m_doClassifyFacetsByAngle;
    //! Associated action
    QAction* m_doShowStereogram;
};
