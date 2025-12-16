// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <qcustomplot.h>

#include <QWidget>

namespace Ui {
class AnglesCustomPlot;
}

class AnglesCustomPlot : public QCustomPlot {
    Q_OBJECT
public:
    explicit AnglesCustomPlot(const QVector<double>& data,
                              const QString& xLabel,
                              int nbBins,
                              QWidget* parent = nullptr);
    ~AnglesCustomPlot();

    bool computeHistogram(const QVector<double>& data);

    void createQCPBars(const QString& xLabel);

    void rescale();

    QSharedPointer<QCPBarsDataContainer> dataContainer() {
        return m_bars->data();
    }

    void emitCloseTab() { emit closeTab(); }

    void mouseDoubleClickEvent(QMouseEvent* event) override;

    void mousePressEvent(QMouseEvent* event) override;

signals:
    void closeTab();

private:
    Ui::AnglesCustomPlot* ui;

    QVector<double> m_axis;

    QVector<double> m_histogram;

    QString m_label;

    QCPBars* m_bars;

    int m_nbBins;
};
