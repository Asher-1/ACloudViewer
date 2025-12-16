// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

namespace Ui {
class G3PointDialog;
}

class G3PointDialog : public QDialog {
    Q_OBJECT

public:
    explicit G3PointDialog(QString cloudName, QWidget *parent = nullptr);
    ~G3PointDialog();

    void readSettings();
    void writeSettings();

    void emitKNNChanged() { emit kNNEditingFinished(); }
    void emitSegment() { emit segment(); }
    void emitClusterAndOrClean() { emit clusterAndOrClean(); }
    void emitGetBorders() { emit getBorders(); }

    void emitAllClicked(bool state) { emit allClicked(state); }
    void emitOnlyOneClicked(bool state) { emit onlyOneClicked(state); }
    void emitOnlyOneChanged(int idx) { emit onlyOneChanged(idx); }
    void emitFit() { emit fit(); }
    void emitExportResults() { emit exportResults(); }
    void emitWolman() { emit wolman(); }
    void emitAngles() { emit angles(); }
    void emitTransparencyChanged(double transparency) {
        emit transparencyChanged(transparency);
    }
    void emitDrawSurfaces(bool state) { emit drawSurfaces(state); }
    void emitDrawLines(bool state) { emit drawLines(state); }
    void emitDrawPoints(bool state) { emit drawPoints(state); }
    void emitGLPointSizeChanged(int size) { emit glPointSize(size); }

    void emitSignals();

    double getMaxAngle1();
    double getMaxAngle2();
    double getMinFlatness();
    int getNMin();
    int getkNN();
    int getWolmanNbIter();
    int getAnglesNbBins();
    double getRadiusFactor();

    bool isSteepestSlope();
    void enableClusterAndOrClean(bool state);
    bool clusterIsChecked();
    bool cleanIsChecked();

    void setOnlyOneMax(int idx);
    void enableDrawPointsForOnlyOneGrain(bool state);

signals:
    void kNNEditingFinished();
    void segment();
    void clusterAndOrClean();
    void getBorders();
    void fit();
    void exportResults();
    void wolman();
    void angles();

    void allClicked(bool state);
    void onlyOneClicked(bool state);
    void onlyOneChanged(int idx);
    void transparencyChanged(double transparency);
    void drawSurfaces(bool state);
    void drawLines(bool state);
    void drawPoints(bool state);
    void glPointSize(int size);

private:
    Ui::G3PointDialog *ui;
};
