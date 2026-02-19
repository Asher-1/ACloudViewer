// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_volumeCalcDlg.h>

// Local
#include "ecv2.5DimEditor.h"

// Qt
#include <QDialog>

class ccGenericPointCloud;
class ccPointCloud;
class ccPolyline;
class QComboBox;

/**
 * @class ccVolumeCalcTool
 * @brief Volume calculation tool dialog
 * 
 * Interactive tool for computing volumes between two point clouds or
 * between a point cloud and a reference plane. Uses 2.5D rasterization
 * to create a grid-based height map for volume calculations.
 * 
 * Features:
 * - Volume computation between two clouds (ground/ceiling model)
 * - Added/removed volume analysis
 * - Surface area calculation
 * - Multiple projection types (min, max, average, median)
 * - Empty cell filling strategies
 * - Export results as point cloud or report
 * - Real-time 2D visualization of height grid
 * 
 * Applications:
 * - Earthwork volume calculations
 * - Stockpile volume measurements
 * - Cut/fill analysis
 * - Terrain change detection
 * 
 * @see cc2Point5DimEditor
 * @see ccRasterGrid
 */
class ccVolumeCalcTool : public QDialog,
                         public cc2Point5DimEditor,
                         public Ui::VolumeCalcDialog {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param cloud1 First point cloud (ground)
     * @param cloud2 Second point cloud (ceiling)
     * @param parent Parent widget
     */
    ccVolumeCalcTool(ccGenericPointCloud* cloud1,
                     ccGenericPointCloud* cloud2,
                     QWidget* parent = 0);

    /**
     * @brief Destructor
     */
    ~ccVolumeCalcTool();

    /**
     * @brief Get grid step (inherited from cc2Point5DimEditor)
     * @return Grid cell size
     */
    virtual double getGridStep() const override;
    
    /**
     * @brief Get projection dimension (inherited from cc2Point5DimEditor)
     * @return Projection axis (0=X, 1=Y, 2=Z)
     */
    virtual unsigned char getProjectionDimension() const override;
    
    /**
     * @brief Get projection type (inherited from cc2Point5DimEditor)
     * @return Projection type (min, max, average, etc.)
     */
    virtual ccRasterGrid::ProjectionType getTypeOfProjection() const override;

    /**
     * @struct ReportInfo
     * @brief Volume calculation report information
     * 
     * Contains all computed volume and surface metrics.
     */
    struct ReportInfo {
        /**
         * @brief Default constructor
         */
        ReportInfo()
            : volume(0),
              addedVolume(0),
              removedVolume(0),
              surface(0),
              matchingPrecent(0),
              ceilNonMatchingPercent(0),
              groundNonMatchingPercent(0),
              averageNeighborsPerCell(0) {}

        /**
         * @brief Convert report to formatted text
         * @param precision Number of decimal places (default: 6)
         * @return Formatted report string
         */
        QString toText(int precision = 6) const;

        double volume;                      ///< Net volume
        double addedVolume;                 ///< Added material volume
        double removedVolume;               ///< Removed material volume
        double surface;                     ///< Surface area
        float matchingPrecent;              ///< Percentage of matching cells
        float ceilNonMatchingPercent;       ///< Percentage of ceiling-only cells
        float groundNonMatchingPercent;     ///< Percentage of ground-only cells
        double averageNeighborsPerCell;     ///< Average neighbors per grid cell
    };

    //! Static accessor
    static bool ComputeVolume(
            ccRasterGrid& grid,
            ccGenericPointCloud* ground,
            ccGenericPointCloud* ceil,
            const ccBBox& gridBox,
            unsigned char vertDim,
            double gridStep,
            unsigned gridWidth,
            unsigned gridHeight,
            ccRasterGrid::ProjectionType projectionType,
            ccRasterGrid::EmptyCellFillOption groundEmptyCellFillStrategy,
            ccRasterGrid::EmptyCellFillOption ceilEmptyCellFillStrategy,
            ccVolumeCalcTool::ReportInfo& reportInfo,
            double groundHeight,
            double ceilHeight,
            QWidget* parentWidget = 0);

    //! Converts a (volume) grid to a point cloud
    static ccPointCloud* ConvertGridToCloud(ccRasterGrid& grid,
                                            const ccBBox& gridBox,
                                            unsigned char vertDim,
                                            bool exportToOriginalCS);

protected slots:

    //! Accepts the dialog and save settings
    void saveSettingsAndAccept();

    //! Save persistent settings and 'accept' dialog
    void saveSettings();

    //! Called when the projection direction changes
    void projectionDirChanged(int);

    //! Called when the SF projection type changes
    void sfProjectionTypeChanged(int);

    // Inherited from cc2Point5DimEditor
    virtual bool showGridBoxEditor() override;

    //! Called when the (ground) empty cell filling strategy changes
    void groundFillEmptyCellStrategyChanged(int);
    //! Called when the (ceil) empty cell filling strategy changes
    void ceilFillEmptyCellStrategyChanged(int);

    //! Called when the an option of the grid generation has changed
    void gridOptionChanged();

    //! Updates the gid info
    void updateGridInfo();

    //! Update the grid and the 2D display
    void updateGridAndDisplay();

    //! Swap roles
    void swapRoles();

    //! Ground source changed
    void groundSourceChanged(int);
    //! Ceil source changed
    void ceilSourceChanged(int);

    //! Exports info to clipboard
    void exportToClipboard() const;

    //! Exports the grid as a point cloud
    void exportGridAsCloud() const;

    //! Sets the displayed number precision
    void setDisplayedNumberPrecision(int);

protected:  // standard methods
    // Inherited from cc2Point5DimEditor
    virtual void gridIsUpToDate(bool state) override;

    //! Load persistent settings
    void loadSettings();

    //! Updates the grid
    bool updateGrid();

    //! Converts the grid to a point cloud
    ccPointCloud* convertGridToCloud(bool exportToOriginalCS) const;

    //! Outputs the report
    void outputReport(const ReportInfo& info);

protected:  // members
    //! First associated cloud
    ccGenericPointCloud* m_cloud1;
    //! Second associated cloud
    ccGenericPointCloud* m_cloud2;

    //! Last report
    /** Only valid if clipboardPushButton is enabled
     **/
    ReportInfo m_lastReport;
};
