// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QWidget>

class ccScalarField;
class ccHistogramWindow;

namespace Ui {
class SFEditDlg;
}

//! GUI scalar field interactor for properties list dialog
class sfEditDlg : public QWidget {
    Q_OBJECT

public:
    //! Default constructor
    explicit sfEditDlg(QWidget* parent = 0);

    ~sfEditDlg();

    //! Updates dialog with a given scalar field
    void fillDialogWith(ccScalarField* sf);

public:
    void minValSBChanged(double);
    void maxValSBChanged(double);
    void minSatSBChanged(double);
    void maxSatSBChanged(double);

    void minValHistoChanged(double);
    void maxValHistoChanged(double);
    void minSatHistoChanged(double);
    void maxSatHistoChanged(double);

    void nanInGrayChanged(bool);
    void alwaysShow0Changed(bool);
    void symmetricalScaleChanged(bool);
    void logScaleChanged(bool);

signals:

    //! Signal emitted when the SF display parameters have changed
    void entitySFHasChanged();

protected:
    // conversion between sliders (integer) and check box (double) values
    double dispSpin2slider(double val) const;
    double satSpin2slider(double val) const;
    double dispSlider2spin(int pos) const;
    double satSlider2spin(int pos) const;

    //! Associated scalar field
    ccScalarField* m_associatedSF;
    //! Associated scalar field histogram
    ccHistogramWindow* m_associatedSFHisto;

    Ui::SFEditDlg* m_ui;
};
