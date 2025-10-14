// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_2_5D_EDITOR_HEADER
#define CC_2_5D_EDITOR_HEADER

#include <ui_rasterizeDlg.h>

// ECV_DB_LIB
#include <ecvRasterGrid.h>

class ccBoundingBoxEditorDlg;
class ccGLWindow;
class QWidget;
class QFrame;
class QComboBox;

//! 2.5D data editor (generic interface)
class cc2Point5DimEditor {
public:
    //! Default constructor
    cc2Point5DimEditor();

    //! Destructor
    virtual ~cc2Point5DimEditor();

protected:  // standard methods
    //! Returns projection grid step
    virtual double getGridStep() const = 0;

    //! Returns projection dimension
    /** \return dimension as int (0: X, 1: Y, 2:Z)
     **/
    virtual unsigned char getProjectionDimension() const = 0;

    //! Returns type of projection
    virtual ccRasterGrid::ProjectionType getTypeOfProjection() const = 0;

    //! Returns custom bbox
    virtual ccBBox getCustomBBox() const;

    //! Declares whether the grid is up-to-date or not
    virtual void gridIsUpToDate(bool state) = 0;

    //! Updates the 2D display zoom
    virtual void update2DDisplayZoom(ccBBox& box);

protected:  // raster grid related stuff
    //! Show grid box editor and update
    /** \return whether the box was modified or not
     **/
    virtual bool showGridBoxEditor();

    //! Returns the grid size as a string
    virtual QString getGridSizeAsString() const;

    //! Returns the grid size
    virtual bool getGridSize(unsigned& width, unsigned& height) const;

    //! Creates the bounding-box editor
    void createBoundingBoxEditor(const ccBBox& gridBBox, QWidget* parent);

    //! Creates the 2D view
    void create2DView(QFrame* parentFrame);

    //! Returns the empty cell strategy (for a given combo-box)
    ccRasterGrid::EmptyCellFillOption getFillEmptyCellsStrategy(
            QComboBox* comboBox) const;

    //! Shortcut to ccRasterGrid::convertToCloud
    ccPointCloud* convertGridToCloud(
            const std::vector<ccRasterGrid::ExportableFields>& exportedFields,
            bool interpolateSF,
            bool interpolateColors,
            bool resampleInputCloudXY,
            bool resampleInputCloudZ,  // only considered if
                                       // resampleInputCloudXY is true!
            ccGenericPointCloud* inputCloud,
            bool fillEmptyCells,
            double emptyCellsHeight,
            bool exportToOriginalCS) const;

protected:  // members
    //! Raster grid
    ccRasterGrid m_grid;

    //! Associated dialog
    ccBoundingBoxEditorDlg* m_bbEditorDlg;

    //! 'Raster' cloud
    ccPointCloud* m_rasterCloud;
};

#endif  // CC_2_5D_EDITOR_HEADER
