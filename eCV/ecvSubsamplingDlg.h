// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_SAMPLE_DLG_HEADER
#define ECV_SAMPLE_DLG_HEADER

// Qt
#include <QDialog>

// CV_CORE_LIB
#include <GenericProgressCallback.h>
#include <ReferenceCloud.h>

// GUI
#include <ui_subsamplingDlg.h>

class ccGenericPointCloud;

//! Subsampling cloud dialog
class ccSubsamplingDlg : public QDialog, public Ui::SubsamplingDialog {
    Q_OBJECT

public:
    //! Sub-sampling method
    enum CC_SUBSAMPLING_METHOD {
        RANDOM = 0,
        SPACE = 1,
        OCTREE = 2,
    };

    //! Default constructor
    ccSubsamplingDlg(unsigned maxPointCount,
                     double maxCloudRadius,
                     QWidget* parent = 0);

    //! Returns subsampled version of a cloud according to current parameters
    /** Should be called only once the dialog has been validated.
     **/
    cloudViewer::ReferenceCloud* getSampledCloud(
            ccGenericPointCloud* cloud,
            cloudViewer::GenericProgressCallback* progressCb = 0);

    //! Enables the SF modulation option (SPATIAL method)
    void enableSFModulation(ScalarType sfMin, ScalarType sfMax);

protected slots:

    void sliderMoved(int sliderPos);
    void samplingRateChanged(double value);
    void changeSamplingMethod(int index);

protected:  // methods
    //! Updates the dialog labels depending on the active mode
    void updateLabels();

protected:  // members
    //! Max point count (for RANDOM method)
    unsigned m_maxPointCount;

    //! Max radius (for SPACE method)
    double m_maxRadius;

    //! Scalar modulation
    bool m_sfModEnabled;
    //! Scalar modulation (min SF value)
    ScalarType m_sfMin;
    //! Scalar modulation (max SF value)
    ScalarType m_sfMax;
};

#endif  // ECV_SAMPLE_DLG_HEADER
