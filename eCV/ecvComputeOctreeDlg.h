// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_COMPUTE_OCTREE_DLG_HEADER
#define ECV_COMPUTE_OCTREE_DLG_HEADER

#include <ui_computeOctreeDlg.h>

// ECV_DB_LIB
#include <ecvBBox.h>

class ccBoundingBoxEditorDlg;

//! Dialog for octree computation
class ccComputeOctreeDlg : public QDialog, public Ui::ComputeOctreeDialog {
    Q_OBJECT

public:
    //! Default constructor
    ccComputeOctreeDlg(const ccBBox& baseBBox,
                       double minCellSize,
                       QWidget* parent = 0);

    //! Computation mode
    enum ComputationMode { DEFAULT, MIN_CELL_SIZE, CUSTOM_BBOX };

    //! Returns octree computation mode
    ComputationMode getMode() const;

    //! Returns cell size at max level
    double getMinCellSize() const;

    //! Returns custom bbox
    ccBBox getCustomBBox() const;

protected:
    //! Associated dialog
    ccBoundingBoxEditorDlg* m_bbEditorDlg;
};

#endif  // CC_COMPUTE_OCTREE_DLG_HEADER
