// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

#include "CVPluginAPI.h"

namespace Ui {
class RenderToFileDialog;
}

//! Dialog for screen to file rendering
class CVPLUGIN_LIB_API ccRenderToFileDlg : public QDialog {
    Q_OBJECT

public:
    //! Default constructor
    ccRenderToFileDlg(unsigned baseWidth,
                      unsigned baseHeight,
                      QWidget* parent = 0);

    ~ccRenderToFileDlg() override;

    //! Disable and hide the scale and overlay checkboxes
    void hideOptions();

    //! On dialog acceptance, returns requested zoom
    float getZoom() const;
    //! On dialog acceptance, returns requested output filename
    QString getFilename() const;
    //! On dialog acceptance, returns whether points should be scaled or not
    bool dontScalePoints() const;
    //! Whether overlay items should be rendered
    bool renderOverlayItems() const;

protected slots:

    void chooseFile();
    void updateInfo();
    void saveSettings();

protected:
    unsigned w;
    unsigned h;

    QString selectedFilter;
    QString currentPath;
    QString filters;

    Ui::RenderToFileDialog* m_ui;
};
