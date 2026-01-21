// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <ui_orthoSectionGenerationDlg.h>

#include <QDialog>

//! Dialog for generating orthogonal sections along a path (Section Extraction
//! Tool)
class ccOrthoSectionGenerationDlg : public QDialog,
                                    public Ui::OrthoSectionGenerationDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccOrthoSectionGenerationDlg(QWidget* parent = 0);

    //! Sets the path legnth
    void setPathLength(double l);

    //! Sets whether the generatrix should be automatically saved and removed
    void setAutoSaveAndRemove(bool state);
    //! Returns whether the generatrix should be automatically saved and removed
    bool autoSaveAndRemove() const;

    //! Sets the generation step
    void setGenerationStep(double s);
    //! Sets he sections width
    void setSectionsWidth(double w);

    //! Returns the generation step
    double getGenerationStep() const;
    //! Returns the sections width
    double getSectionsWidth() const;

protected slots:
    void onStepChanged(double);

protected:
    //! Path length
    double m_pathLength;
};
