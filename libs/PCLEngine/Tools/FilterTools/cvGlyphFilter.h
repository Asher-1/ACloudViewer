// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvGenericFilter.h"

namespace Ui {
class cvGlyphFilterDlg;
}

class vtkActor2D;
class vtkGlyph3D;
class cvGlyphFilter : public cvGenericFilter {
    Q_OBJECT
public:
    explicit cvGlyphFilter(QWidget* parent = nullptr);

    virtual void apply() override;

    virtual ccHObject* getOutput() override;

    virtual void clearAllActor() override;

public:
    virtual void showInteractor(bool state) override;

protected:
    virtual void modelReady() override;
    virtual void createUi() override;
    virtual void initFilter() override;
    virtual void dataChanged() override;

private slots:
    void on_sizeSpinBox_valueChanged(double arg1);
    void on_shapeCombo_currentIndexChanged(int index);
    void onColorChanged(const QColor& clr);
    void on_displayEffectCombo_currentIndexChanged(int index);
    void on_labelGroupBox_toggled(bool arg1);
    void showLabels(bool show = true);
    void setLabelsColor(const QColor& clr);
    void on_modeCombo_currentIndexChanged(int index);

private:
    enum Shape { Arrow, Cone, Line, Cylinder, Sphere, Point };

    Ui::cvGlyphFilterDlg* m_configUi = nullptr;
    QColor m_glyphColor = Qt::white;
    QColor m_labelColor = Qt::white;
    double m_size = 3;
    int m_labelMode = 1;  // VTK_LABEL_SCALARS
    Shape m_shape = Arrow;

    vtkSmartPointer<vtkActor2D> m_labelActor;
    vtkSmartPointer<vtkGlyph3D> m_glyph3d;
    bool m_labelVisible = false;
};
