// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_GEOM_FEATURES_DIALOG_HEADER
#define ECV_GEOM_FEATURES_DIALOG_HEADER

// Local
#include "ecvLibAlgorithms.h"

// Qt
#include <ui_geomFeaturesDlg.h>

#include <QDialog>

//! Dialog for computing the density of a point clouds
class ccGeomFeaturesDlg : public QDialog, public Ui::GeomFeaturesDialog {
public:
    //! Default constructor
    explicit ccGeomFeaturesDlg(QWidget* parent = nullptr);

    //! Sets selected features
    void setSelectedFeatures(
            const ccLibAlgorithms::GeomCharacteristicSet& features);
    //! Returns selected features
    bool getSelectedFeatures(
            ccLibAlgorithms::GeomCharacteristicSet& features) const;
    //! Sets the default kernel radius (for 'precise' mode only)
    void setRadius(double r);
    //! Returns	the kernel radius (for 'precise' mode only)
    double getRadius() const;

    //! Sets the 'up direction' (and enables the group at the same time)
    void setUpDirection(const CCVector3& upDir);
    //! Returns the 'up direction' if any is defined (nullptr otherwise)
    CCVector3* getUpDirection() const;

    //! reset the whole dialog
    void reset();

protected:
    struct Option : ccLibAlgorithms::GeomCharacteristic {
        Option(QCheckBox* cb,
               cloudViewer::GeometricalAnalysisTools::GeomCharacteristic c,
               int option = 0)
            : ccLibAlgorithms::GeomCharacteristic(c, option), checkBox(cb) {}

        QCheckBox* checkBox = nullptr;
    };

    std::vector<Option> m_options;
};

#endif  // ECV_GEOM_FEATURES_DIALOG_HEADER
