// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_GRAPHICAL_SEGMENTATION_OPTIONS_DLG_HEADER
#define CC_GRAPHICAL_SEGMENTATION_OPTIONS_DLG_HEADER

// Qt
#include <QString>

// GUI
#include <ui_graphicalSegmentationOptionsDlg.h>

class ccGraphicalSegmentationOptionsDlg
    : public QDialog,
      public Ui::GraphicalSegmentationOptionsDlg {
    Q_OBJECT

public:
    //! Default constructor
    ccGraphicalSegmentationOptionsDlg(const QString windowTitle = QString(),
                                      QWidget* parent = nullptr);

    void accept();

    //! Returns the QSettings key to store the segmentation tool options
    static QString SegmentationToolOptionsKey() {
        return "SegmentationToolOptions";
    }
    //! Returns the QSettings key to store the 'remaining entity' suffix
    static QString RemainingSuffixKey() { return "Remaining"; }
    //! Returns the QSettings key to store the 'segmented entity' suffix
    static QString SegmentedSuffixKey() { return "Segmented"; }
};

#endif  // CC_GRAPHICAL_SEGMENTATION_OPTIONS_DLG_HEADER
