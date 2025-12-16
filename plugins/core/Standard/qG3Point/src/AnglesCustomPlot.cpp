// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnglesCustomPlot.h"

#include <CVLog.h>
#include <ScalarField.h>

#include "ui_AnglesCustomPlot.h"

AnglesCustomPlot::AnglesCustomPlot(const QVector<double>& data,
                                   const QString& xLabel,
                                   int nbBins,
                                   QWidget* parent)
    : ui(new Ui::AnglesCustomPlot),
      m_label(xLabel),
      m_bars(nullptr),
      m_nbBins(nbBins) {
    setProperty("TypeOfCustomPlot", "AnglesCustomPlot");

    ui->setupUi(this);

    setWindowTitle(m_label);

    computeHistogram(data);

    createQCPBars(m_label + " [°]");

    m_bars->rescaleAxes();
}

AnglesCustomPlot::~AnglesCustomPlot() { delete ui; }

//! Default number of classes for associated histogram
static const unsigned MAX_HISTOGRAM_SIZE = 512;

bool AnglesCustomPlot::computeHistogram(const QVector<double>& data) {
    double minData = *std::min_element(data.begin(), data.end());
    double maxData = *std::max_element(data.begin(), data.end());
    double range = maxData - minData;

    if (range == 0 || data.size() == 0) {
        // can't build histogram of a flat field
        return false;
    } else {
        unsigned count = data.size();
        unsigned numberOfBins;
        if (m_nbBins == -1)  // compute number of bins automatically
        {
            numberOfBins = static_cast<unsigned>(
                    ceil(sqrt(static_cast<double>(count))));
            numberOfBins = std::max<unsigned>(
                    std::min<unsigned>(numberOfBins, MAX_HISTOGRAM_SIZE), 4);
        } else  // use number of bins
        {
            numberOfBins = m_nbBins;
        }

        m_axis.resize(numberOfBins);
        for (unsigned i = 0; i < numberOfBins; i++) {
            m_axis[i] = minData + i * range / numberOfBins;
        }

        // reserve memory
        try {
            m_histogram.resize(numberOfBins);
        } catch (const std::bad_alloc&) {
            CVLog::Warning("[computeHistogram] Failed to allocate histogram!");
        }

        std::fill(m_histogram.begin(), m_histogram.end(), 0);

        // compute histogram
        ScalarType step = static_cast<ScalarType>(numberOfBins) / range;
        for (unsigned i = 0; i < count; ++i) {
            const ScalarType& val = data[i];

            if (cloudViewer::ScalarField::ValidValue(val)) {
                unsigned bin = static_cast<unsigned>((val - minData) * step);
                ++m_histogram[std::min(bin, numberOfBins - 1)];
            }
        }
    }

    return true;
}

void AnglesCustomPlot::createQCPBars(const QString& xLabel) {
    // clear previous display
    if (m_bars) {
        this->removePlottable(m_bars);
        m_bars = nullptr;
    }

    m_bars = new QCPBars(xAxis, yAxis);
    m_bars->setAntialiased(
            false);  // gives more crisp, pixel aligned bar borders
    m_bars->setPen(QPen(QColor(0, 168, 140).lighter(130)));
    m_bars->setBrush(QColor(0, 168, 140));
    m_bars->setWidth(m_axis[1] - m_axis[0]);
    m_bars->setData(m_axis, m_histogram);
    xAxis->setLabel(xLabel);
    yAxis->setLabel("Counts");
    setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
}

void AnglesCustomPlot::rescale() {
    // set ranges appropriate to show data
    if (m_bars) {
        m_bars->rescaleAxes();
        replot();
    }
}

void AnglesCustomPlot::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        rescale();
    }
    QCustomPlot::mouseDoubleClickEvent(event);
}

void AnglesCustomPlot::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::RightButton) {
        QMenu* menu = new QMenu(this);
        QAction* action = new QAction("Close tab");
        menu->addAction(action);
        connect(action, &QAction::triggered, this,
                &AnglesCustomPlot::emitCloseTab);
        menu->popup(event->globalPos());
    }
    QCustomPlot::mousePressEvent(event);
}
