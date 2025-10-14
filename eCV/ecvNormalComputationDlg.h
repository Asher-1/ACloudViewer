// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_normalComputationDlg.h>

// ECV_DB_LIB
#include <ecvNormalVectors.h>

class ccPointCloud;

//! Dialog for normal computation
class ccNormalComputationDlg : public QDialog, public Ui::NormalComputationDlg {
    Q_OBJECT

public:
    //! Default constructor
    /** \param withScanGrid whether the selection contains some structured point
    clouds
     *  \param withSensor whether the selection contains some sensors associated
    to the point clouds \param parent parent widget
    **/
    explicit ccNormalComputationDlg(bool withScanGrid,
                                    bool withSensor,
                                    QWidget* parent = nullptr);

    //! Returns the local model chosen for normal computation
    CV_LOCAL_MODEL_TYPES getLocalModel() const;

    //! Sets the local model chosen for normal computation
    void setLocalModel(CV_LOCAL_MODEL_TYPES model);

    //! Sets default value for local neighbourhood radius
    void setRadius(PointCoordinateType radius);

    //! Sets the preferred orientation
    void setPreferredOrientation(ccNormalVectors::Orientation orientation);

    //! Sets the currently selected cloud (required for 'auto' feature)
    void setCloud(ccPointCloud* cloud);

    //! Returns whether scan grids should be used for computation
    bool useScanGridsForComputation() const;

    //! Returns the min angle for grid triangles
    double getMinGridAngle_deg() const;

    //! Sets the min angle for grid triangles
    void setMinGridAngle_deg(double value);

    //! Returns local neighbourhood radius
    PointCoordinateType getRadius() const;

    //! Returns whether normals should be oriented or not
    bool orientNormals() const;

    //! Returns whether scan grids should be used for normals orientation
    bool useScanGridsForOrientation() const;

    //! Returns whether scan grids should be used for normals computation
    bool useSensorsForOrientation() const;

    //! Returns whether a preferred orientation should be used
    bool usePreferredOrientation() const;

    //! Returns the preferred orientation (if any)
    ccNormalVectors::Orientation getPreferredOrientation() const;

    //! Returns whether a Minimum Spanning Tree (MST) should be used for normals
    //! orientation
    bool useMSTOrientation() const;

    //! Returns the number of neighbors for Minimum Spanning Tree (MST)
    int getMSTNeighborCount() const;

    //! Sets the number of neighbors for Minimum Spanning Tree (MST)
    void setMSTNeighborCount(int n);

protected slots:

    //! On local model change
    void localModelChanged(int index);

    //! Automatically estimate the local surface radius
    void autoEstimateRadius();

protected:
    //! Selected cloud
    ccPointCloud* m_cloud;
};
