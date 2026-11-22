// SPDX-License-Identifier: GPL-3.0-or-later
// Original ACloudViewer API illustration generator; see LICENSE-NOTICE.md.
#include "qcustomplot.h"

#include <QApplication>
#include <QBrush>
#include <QColor>
#include <QFont>
#include <QPainter>
#include <QPen>
#include <QString>
#include <QVector>
#include <QtMath>

#include <cstdlib>
#include <iostream>

namespace {

const QColor canvas("#f4f7fa");
const QColor plotPaper("#ffffff");
const QColor ink("#273244");
const QColor grid("#dce3eb");
const QColor teal("#087f8c");
const QColor orange("#e07a2d");
const QColor blue("#3867a8");
const QColor green("#4c956c");
const QColor muted("#758195");

void stylePlot(QCustomPlot &plot, const QString &title) {
    plot.setBackground(canvas);
    plot.axisRect()->setBackground(plotPaper);
    plot.plotLayout()->insertRow(0);
    auto *heading = new QCPPlotTitle(&plot, title);
    QFont titleFont("DejaVu Sans", 13, QFont::DemiBold);
    heading->setFont(titleFont);
    heading->setTextColor(ink);
    plot.plotLayout()->addElement(0, 0, heading);

    plot.xAxis->setBasePen(QPen(ink, 1.2));
    plot.yAxis->setBasePen(QPen(ink, 1.2));
    plot.xAxis->setTickPen(QPen(muted, 1));
    plot.yAxis->setTickPen(QPen(muted, 1));
    plot.xAxis->setSubTickPen(QPen(grid, 0.8));
    plot.yAxis->setSubTickPen(QPen(grid, 0.8));
    plot.xAxis->setTickLabelColor(ink);
    plot.yAxis->setTickLabelColor(ink);
    plot.xAxis->setLabelColor(ink);
    plot.yAxis->setLabelColor(ink);
    QFont labelFont("DejaVu Sans", 9);
    plot.xAxis->setTickLabelFont(labelFont);
    plot.yAxis->setTickLabelFont(labelFont);
    plot.xAxis->setLabelFont(labelFont);
    plot.yAxis->setLabelFont(labelFont);
    plot.xAxis->grid()->setVisible(false);
    plot.yAxis->grid()->setVisible(true);
    plot.yAxis->grid()->setPen(QPen(grid, 0.9, Qt::DotLine));
    plot.xAxis->grid()->setSubGridVisible(false);
    plot.yAxis->grid()->setSubGridVisible(false);
    plot.legend->setVisible(false);
}

bool save(QCustomPlot &plot, const QString &path) {
    plot.replot();
    return plot.savePng(path, 760, 450, 1.0, -1);
}

void setPlotPosition(QCPItemPosition *position, double key, double value) {
    position->setType(QCPItemPosition::ptPlotCoords);
    position->setCoords(key, value);
}


void addIllustrationLegend(QCustomPlot &plot,
                          double left,
                          double top,
                          const QVector<QColor> &swatches,
                          const QStringList &labels) {
    if (swatches.size() != labels.size() || swatches.isEmpty()) return;
    const double line = 0.055;
    auto *box = new QCPItemRect(&plot);
    box->topLeft->setType(QCPItemPosition::ptAxisRectRatio);
    box->bottomRight->setType(QCPItemPosition::ptAxisRectRatio);
    box->topLeft->setCoords(left - 0.025, top - 0.03);
    box->bottomRight->setCoords(0.98, top + line * swatches.size() + 0.025);
    box->setBrush(QColor(255, 255, 255, 238));
    box->setPen(QPen(grid, 1));
    for (int i = 0; i < swatches.size(); ++i) {
        const double y = top + line * i;
        auto *mark = new QCPItemRect(&plot);
        mark->topLeft->setType(QCPItemPosition::ptAxisRectRatio);
        mark->bottomRight->setType(QCPItemPosition::ptAxisRectRatio);
        mark->topLeft->setCoords(left, y - 0.012);
        mark->bottomRight->setCoords(left + 0.025, y + 0.012);
        mark->setBrush(swatches[i]);
        mark->setPen(QPen(swatches[i].darker(125), 1));
        auto *label = new QCPItemText(&plot);
        label->position->setType(QCPItemPosition::ptAxisRectRatio);
        label->position->setCoords(left + 0.037, y);
        label->setPositionAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        label->setFont(QFont("DejaVu Sans", 8));
        label->setColor(ink);
        label->setPadding(QMargins(0, 0, 0, 0));
        label->setText(labels[i]);
    }
}

bool renderGraph(const QString &path) {
    QCustomPlot plot;
    plot.resize(760, 450);
    stylePlot(plot, "Original example · single-valued samples with a missing interval");
    plot.xAxis->setLabel("sample number");
    plot.yAxis->setLabel("normalized response");
    plot.xAxis->setRange(-1.0, 25.0);
    plot.yAxis->setRange(-1.2, 2.3);

    QVector<double> key, observed, reference;
    for (int i = 0; i <= 24; ++i) {
        const double x = i;
        key.push_back(x);
        reference.push_back(0.70 * qExp(-x / 18.0) * qSin(x * 0.62) + 0.22);
        const double noise = 0.12 * qSin(x * 1.91) + 0.04 * qCos(x * 0.37);
        observed.push_back((i >= 11 && i <= 14) ? qQNaN()
                                               : reference.back() + noise);
    }

    auto *model = plot.addGraph();
    model->setName("smooth reference");
    model->setData(key, reference);
    model->setPen(QPen(blue, 2.2));
    model->setLineStyle(QCPGraph::lsLine);
    model->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone));

    auto *measurement = plot.addGraph();
    measurement->setName("measured samples (gap retained)");
    measurement->setData(key, observed);
    measurement->setPen(QPen(teal, 1.8));
    measurement->setLineStyle(QCPGraph::lsLine);
    measurement->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, teal, teal, 6));
    addIllustrationLegend(plot, 0.69, 0.07,
                          QVector<QColor>{blue, teal},
                          QStringList{"reference", "observed samples"});

    plot.replot();
    return plot.savePng(path, 760, 450, 1.0, -1);
}

bool renderBars(const QString &path) {
    QCustomPlot plot;
    plot.resize(760, 450);
    stylePlot(plot, "Original example · grouped stage measurements");
    plot.xAxis->setLabel("processing stage");
    plot.yAxis->setLabel("elapsed time (ms)");
    plot.xAxis->setRange(0.35, 5.65);
    plot.yAxis->setRange(0, 110);

    plot.xAxis->setAutoTicks(false);
    plot.xAxis->setAutoTickLabels(false);
    plot.xAxis->setTickVector(QVector<double>{1, 2, 3, 4, 5});
    plot.xAxis->setTickVectorLabels(
            QVector<QString>{"decode", "filter", "align", "mesh", "export"});

    auto *group = new QCPBarsGroup(&plot);
    group->setSpacingType(QCPBarsGroup::stPlotCoords);
    group->setSpacing(0.035);

    const QVector<double> keys{1, 2, 3, 4, 5};
    const QVector<QVector<double>> samples{
        {42, 30, 78, 56, 35},
        {27, 55, 43, 31, 62},
        {61, 38, 51, 76, 44}
    };
    const QVector<QColor> colors{teal, orange, green};
    const QStringList names{"CPU path", "cached path", "alternate path"};
    for (int series = 0; series < samples.size(); ++series) {
        auto *bars = new QCPBars(plot.xAxis, plot.yAxis);
        bars->setName(names[series]);
        bars->setData(keys, samples[series]);
        bars->setBarsGroup(group);
        bars->setWidthType(QCPBars::wtPlotCoords);
        bars->setWidth(0.16);
        bars->setPen(QPen(colors[series].darker(125), 1));
        bars->setBrush(colors[series]);
    }
    addIllustrationLegend(plot, 0.62, 0.055,
                          QVector<QColor>{teal, orange, green},
                          QStringList{"CPU path", "cached path", "alternate path"});

    plot.replot();
    return plot.savePng(path, 760, 450, 1.0, -1);
}

bool renderBarsGroup(const QString &path) {
    QCustomPlot plot;
    plot.resize(760, 450);
    stylePlot(plot, "Original example · grouped instrument readings");
    plot.xAxis->setLabel("test batch");
    plot.yAxis->setLabel("throughput (units/s)");
    plot.xAxis->setRange(0.2, 5.8);
    plot.yAxis->setRange(0, 130);
    plot.xAxis->setAutoTicks(false);
    plot.xAxis->setAutoTickLabels(false);
    plot.xAxis->setTickVector(QVector<double>{1, 2, 3, 4, 5});
    plot.xAxis->setTickVectorLabels(QVector<QString>{"A", "B", "C", "D", "E"});
    auto *group = new QCPBarsGroup(&plot);
    group->setSpacingType(QCPBarsGroup::stPlotCoords);
    group->setSpacing(0.04);
    const QVector<double> keys{1, 2, 3, 4, 5};
    const QVector<QVector<double>> values{{94, 72, 108, 64, 87},
                                          {77, 106, 69, 91, 63},
                                          {61, 83, 96, 70, 110}};
    const QVector<QColor> colors{blue, orange, green};
    for (int i = 0; i < values.size(); ++i) {
        auto *bars = new QCPBars(plot.xAxis, plot.yAxis);
        bars->setData(keys, values[i]);
        bars->setBarsGroup(group);
        bars->setWidthType(QCPBars::wtPlotCoords);
        bars->setWidth(0.19);
        bars->setPen(QPen(colors[i].darker(130), 1));
        bars->setBrush(colors[i]);
    }
    addIllustrationLegend(plot, 0.69, 0.055, colors,
                          QStringList{"sensor north", "sensor center", "sensor south"});
    return save(plot, path);
}

bool renderCurve(const QString &path) {
    QCustomPlot plot;
    plot.resize(760, 450);
    stylePlot(plot, "Original example · one key may be visited more than once");
    plot.xAxis->setLabel("horizontal coordinate");
    plot.yAxis->setLabel("vertical coordinate");
    plot.xAxis->setRange(-1.5, 1.5);
    plot.yAxis->setRange(-1.35, 1.35);
    QVector<double> t, key, value;
    for (int i = 0; i <= 320; ++i) {
        const double a = 6.283185307179586 * i / 320.0;
        const double radius = 0.16 + 0.105 * a;
        t.push_back(a);
        key.push_back(radius * qCos(a));
        value.push_back(radius * qSin(a));
    }
    auto *curve = new QCPCurve(plot.xAxis, plot.yAxis);
    curve->setData(t, key, value);
    curve->setPen(QPen(teal, 2.5));
    curve->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone));
    return save(plot, path);
}

bool renderStatisticalBox(const QString &path) {
    QCustomPlot plot;
    plot.resize(760, 450);
    stylePlot(plot, "Original example · distribution summaries and outliers");
    plot.xAxis->setLabel("cohort");
    plot.yAxis->setLabel("measurement");
    plot.xAxis->setRange(0.2, 4.8);
    plot.yAxis->setRange(-10, 24);
    plot.xAxis->setAutoTicks(false);
    plot.xAxis->setAutoTickLabels(false);
    plot.xAxis->setTickVector(QVector<double>{1, 2, 3, 4});
    plot.xAxis->setTickVectorLabels(QVector<QString>{"amber", "cobalt", "jade", "plum"});
    const QVector<QColor> colors{teal, blue, orange, green};
    const double stats[4][5] = {{-6, -1, 3, 6, 11}, {-3, 1, 5, 8, 14},
                                {-8, -2, 2, 7, 13}, {-2, 4, 7, 10, 16}};
    for (int i = 0; i < 4; ++i) {
        auto *box = new QCPStatisticalBox(plot.xAxis, plot.yAxis);
        box->setData(i + 1, stats[i][0], stats[i][1], stats[i][2], stats[i][3], stats[i][4]);
        box->setOutliers(QVector<double>{stats[i][4] + 3, stats[i][0] - 2});
        box->setOutlierStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, colors[i], 6));
        box->setPen(QPen(colors[i].darker(140), 1.3));
        box->setBrush(QColor(colors[i].red(), colors[i].green(), colors[i].blue(), 90));
        box->setMedianPen(QPen(ink, 2.2));
    }
    return save(plot, path);
}

bool renderFinancial(const QString &path) {
    QCustomPlot plot;
    plot.resize(760, 450);
    stylePlot(plot, "Original example · open, high, low and close values");
    plot.xAxis->setLabel("session");
    plot.yAxis->setLabel("index value");
    plot.xAxis->setRange(0, 9);
    plot.yAxis->setRange(84, 118);
    plot.xAxis->setAutoTicks(false);
    plot.xAxis->setAutoTickLabels(false);
    QVector<double> key, open, high, low, close;
    const double o[] = {92, 95, 94, 99, 101, 98, 103, 107};
    const double c[] = {95, 93, 99, 101, 98, 104, 107, 105};
    for (int i = 0; i < 8; ++i) {
        key.push_back(i + 1); open.push_back(o[i]); close.push_back(c[i]);
        high.push_back(qMax(o[i], c[i]) + 2 + (i % 3));
        low.push_back(qMin(o[i], c[i]) - 2 - (i % 2));
    }
    plot.xAxis->setTickVector(key);
    QStringList sessions;
    for (int i = 1; i <= 8; ++i) sessions << QString("S%1").arg(i);
    plot.xAxis->setTickVectorLabels(sessions.toVector());
    auto *financial = new QCPFinancial(plot.xAxis, plot.yAxis);
    financial->setData(key, open, high, low, close);
    financial->setChartStyle(QCPFinancial::csCandlestick);
    financial->setWidth(0.58);
    financial->setTwoColored(true);
    financial->setPenPositive(QPen(teal.darker(120), 1));
    financial->setPenNegative(QPen(orange.darker(120), 1));
    financial->setBrushPositive(QBrush(teal));
    financial->setBrushNegative(QBrush(orange));
    return save(plot, path);
}

bool renderItemLine(const QString &path) {
    QCustomPlot plot; plot.resize(760, 450);
    stylePlot(plot, "Original example · anchored line between plot points");
    plot.xAxis->setLabel("longitude-like key"); plot.yAxis->setLabel("height-like value");
    plot.xAxis->setRange(0, 10); plot.yAxis->setRange(0, 10);
    auto *line = new QCPItemLine(&plot);
    setPlotPosition(line->start, 1, 2); setPlotPosition(line->end, 8, 7);
    line->setPen(QPen(teal, 3));
    line->setHead(QCPLineEnding(QCPLineEnding::esSpikeArrow, 10, 12));
    auto *a = new QCPItemEllipse(&plot); setPlotPosition(a->topLeft, .75, 1.75); setPlotPosition(a->bottomRight, 1.25, 2.25);
    a->setPen(QPen(orange, 2)); a->setBrush(QBrush(QColor(224,122,45,50)));
    auto *b = new QCPItemEllipse(&plot); setPlotPosition(b->topLeft, 7.75, 6.75); setPlotPosition(b->bottomRight, 8.25, 7.25);
    b->setPen(QPen(blue, 2)); b->setBrush(QBrush(QColor(56,103,168,50)));
    return save(plot,path);
}

bool renderItemStraightLine(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · one direction clips to the axis rectangle");
    plot.xAxis->setLabel("key"); plot.yAxis->setLabel("value"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(0,10);
    auto *line=new QCPItemStraightLine(&plot); setPlotPosition(line->point1,2,3); setPlotPosition(line->point2,7,7.2); line->setPen(QPen(blue,2.5));
    auto *text=new QCPItemText(&plot); setPlotPosition(text->position,4.2,7.7); text->setText("infinite straight direction"); text->setColor(ink); text->setFont(QFont("DejaVu Sans",9));
    return save(plot,path);
}

bool renderItemCurve(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · cubic item with independent control points");
    plot.xAxis->setLabel("key"); plot.yAxis->setLabel("value"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(0,10);
    auto *curve=new QCPItemCurve(&plot); setPlotPosition(curve->start,1,2); setPlotPosition(curve->startDir,2,8); setPlotPosition(curve->endDir,7,1); setPlotPosition(curve->end,9,7); curve->setPen(QPen(teal,3)); curve->setHead(QCPLineEnding(QCPLineEnding::esLineArrow,8,11));
    const QVector<QPointF> pts{{1,2},{2,8},{7,1},{9,7}};
    for(const auto &p:pts){auto *m=new QCPItemEllipse(&plot);setPlotPosition(m->topLeft,p.x()-.12,p.y()-.12);setPlotPosition(m->bottomRight,p.x()+.12,p.y()+.12);m->setPen(QPen(orange,2));m->setBrush(QBrush(Qt::white));}
    return save(plot,path);
}

bool renderItemRect(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · rectangle from two anchored corners");
    plot.xAxis->setLabel("key"); plot.yAxis->setLabel("value"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(0,10);
    auto *rect=new QCPItemRect(&plot);setPlotPosition(rect->topLeft,2,7.5);setPlotPosition(rect->bottomRight,7.5,3);rect->setPen(QPen(teal,2.5));rect->setBrush(QColor(8,127,140,55));
    return save(plot,path);
}

bool renderItemEllipse(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · ellipse bounded by opposite corners");
    plot.xAxis->setLabel("key"); plot.yAxis->setLabel("value"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(0,10);
    auto *ellipse=new QCPItemEllipse(&plot);setPlotPosition(ellipse->topLeft,2,8);setPlotPosition(ellipse->bottomRight,8,2);ellipse->setPen(QPen(blue,3));ellipse->setBrush(QColor(56,103,168,55));
    auto *center=new QCPItemEllipse(&plot);setPlotPosition(center->topLeft,4.85,4.85);setPlotPosition(center->bottomRight,5.15,5.15);center->setPen(QPen(orange,2));center->setBrush(orange);
    return save(plot,path);
}

bool renderItemText(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · rotated annotation at a plot position");
    plot.xAxis->setLabel("key"); plot.yAxis->setLabel("value"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(0,10);
    auto *band=new QCPItemRect(&plot);setPlotPosition(band->topLeft,1,8);setPlotPosition(band->bottomRight,9,2);band->setPen(QPen(grid,1));band->setBrush(QColor(8,127,140,28));
    auto *note=new QCPItemText(&plot);setPlotPosition(note->position,4.8,5);note->setText("calibration zone");note->setRotation(-24);note->setColor(ink);note->setFont(QFont("DejaVu Sans",14,QFont::DemiBold));note->setBrush(QColor(255,255,255,210));note->setPen(QPen(teal,1));note->setPadding(QMargins(8,5,8,5));
    return save(plot,path);
}

bool renderItemPixmap(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · generated pixmap anchored in data space");
    plot.xAxis->setLabel("key"); plot.yAxis->setLabel("value"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(0,10);
    QImage art(120,80,QImage::Format_ARGB32_Premultiplied);art.fill(QColor("#edf4f7"));
    QPainter painter(&art);painter.setRenderHint(QPainter::Antialiasing);painter.setPen(QPen(teal,4));painter.drawEllipse(QRect(10,12,44,44));painter.setBrush(QColor(224,122,45,150));painter.setPen(Qt::NoPen);painter.drawRoundedRect(QRect(62,18,46,38),8,8);painter.setPen(QPen(blue,3));painter.drawLine(8,68,112,68);painter.end();
    auto *pix=new QCPItemPixmap(&plot);setPlotPosition(pix->topLeft,2,8);setPlotPosition(pix->bottomRight,8,3);pix->setPixmap(QPixmap::fromImage(art));pix->setScaled(true,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);pix->setPen(QPen(ink,1));
    auto *label=new QCPItemText(&plot);setPlotPosition(label->position,5,2.3);label->setText("generated geometric texture");label->setColor(ink);label->setFont(QFont("DejaVu Sans",9));label->setPositionAlignment(Qt::AlignHCenter|Qt::AlignTop);
    return save(plot,path);
}

bool renderItemTracer(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · tracer interpolates along a data graph");
    plot.xAxis->setLabel("time (s)"); plot.yAxis->setLabel("signal"); plot.xAxis->setRange(0,12); plot.yAxis->setRange(-1.4,1.4);
    QVector<double> x,y;for(int i=0;i<=120;++i){double t=i/10.0;x<<t;y<<qSin(t*.8)+.2*qCos(t*2.1);}
    auto *graph=plot.addGraph();graph->setData(x,y);graph->setPen(QPen(blue,2));graph->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone));
    auto *tracer=new QCPItemTracer(&plot);tracer->setGraph(graph);tracer->setGraphKey(6.35);tracer->setInterpolating(true);tracer->setStyle(QCPItemTracer::tsCircle);tracer->setSize(13);tracer->setPen(QPen(orange,2));tracer->setBrush(Qt::white);
    auto *guide=new QCPItemLine(&plot);setPlotPosition(guide->start,6.35,-1.2);setPlotPosition(guide->end,6.35,.35);guide->setPen(QPen(muted,1,Qt::DashLine));
    return save(plot,path);
}

bool renderItemBracket(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · bracket marks a selected interval");
    plot.xAxis->setLabel("frequency (Hz)"); plot.yAxis->setLabel("response"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(-1.4,1.4);
    auto *g=plot.addGraph();QVector<double>x,y;for(int i=0;i<=100;++i){double t=i/10.0;x<<t<<y<<qSin(t*.75)*qExp(-t/16);};g->setData(x,y);g->setPen(QPen(blue,2));
    auto *bracket=new QCPItemBracket(&plot);setPlotPosition(bracket->left,2,-.8);setPlotPosition(bracket->right,7,-.8);bracket->setStyle(QCPItemBracket::bsSquare);bracket->setLength(14);bracket->setPen(QPen(orange,2));
    auto *text=new QCPItemText(&plot);setPlotPosition(text->position,4.5,-1.05);text->setText("selected band");text->setColor(ink);text->setFont(QFont("DejaVu Sans",9));text->setPositionAlignment(Qt::AlignHCenter|Qt::AlignTop);
    return save(plot,path);
}

bool renderBracketLength(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · bracket cap length is specified in pixels");
    plot.xAxis->setLabel("key"); plot.yAxis->setLabel("value"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(0,10);
    const QVector<double> lengths{6,14,24};const QVector<QColor> colors{teal,blue,orange};
    for(int i=0;i<3;++i){auto *b=new QCPItemBracket(&plot);setPlotPosition(b->left,2,8-i*2.6);setPlotPosition(b->right,8,8-i*2.6);b->setLength(lengths[i]);b->setStyle(QCPItemBracket::bsSquare);b->setPen(QPen(colors[i],2));auto *t=new QCPItemText(&plot);setPlotPosition(t->position,8.25,8-i*2.6);t->setText(QString::number(lengths[i])+" px");t->setColor(colors[i]);t->setFont(QFont("DejaVu Sans",9));}
    return save(plot,path);
}

bool renderLineEnding(const QString &path) {
    QCustomPlot plot; plot.resize(760,450);
    stylePlot(plot,"Original example · endpoint glyphs on measurement guides");
    plot.xAxis->setLabel("key"); plot.yAxis->setLabel("value"); plot.xAxis->setRange(0,10); plot.yAxis->setRange(0,10);
    const QVector<QCPLineEnding::EndingStyle> styles{QCPLineEnding::esNone,QCPLineEnding::esFlatArrow,QCPLineEnding::esSpikeArrow,QCPLineEnding::esDisc,QCPLineEnding::esSquare,QCPLineEnding::esDiamond,QCPLineEnding::esBar,QCPLineEnding::esHalfBar,QCPLineEnding::esSkewedBar};
    const QStringList labels{"none","flat","spike","disc","square","diamond","bar","half","skew"};
    const QVector<QColor> colors{muted,teal,blue,orange,green,teal,blue,orange,green};
    for(int i=0;i<styles.size();++i){double y=9.1-i*.95;auto *line=new QCPItemLine(&plot);setPlotPosition(line->start,2,y);setPlotPosition(line->end,6,y);line->setPen(QPen(colors[i],2));line->setHead(QCPLineEnding(styles[i],10,13));auto *text=new QCPItemText(&plot);setPlotPosition(text->position,6.4,y);text->setText(labels[i]);text->setColor(ink);text->setFont(QFont("DejaVu Sans",8));}
    return save(plot,path);
}

bool fillMap(QCPColorMap *map, int nx, int ny, bool coarse = false) {
    map->data()->setSize(nx, ny);
    map->data()->setRange(QCPRange(-2, 2), QCPRange(-1.5, 1.5));
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const double k = -2 + 4.0 * (x + .5) / nx;
            const double v = -1.5 + 3.0 * (y + .5) / ny;
            const double z = qExp(-1.4 * ((k - .75) * (k - .75) + (v - .45) * (v - .45)))
                           - .7 * qExp(-2.0 * ((k + .75) * (k + .75) + (v + .55) * (v + .55)))
                           + .14 * qSin(3*k + v);
            map->data()->setCell(x, y, z);
        }
    }
    map->setDataRange(QCPRange(-.9, 1.05));
    return true;
}

QCPColorGradient customGradient() {
    QCPColorGradient gradient;
    gradient.clearColorStops();
    gradient.setColorStopAt(0.0, QColor("#263c8c"));
    gradient.setColorStopAt(0.34, QColor("#22a8a1"));
    gradient.setColorStopAt(0.68, QColor("#f0bd4f"));
    gradient.setColorStopAt(1.0, QColor("#b43c56"));
    return gradient;
}

bool renderColorMap(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · scalar field from analytic samples");
    plot.xAxis->setLabel("position x");plot.yAxis->setLabel("position y");plot.xAxis->setRange(-2,2);plot.yAxis->setRange(-1.5,1.5);
    auto *map=new QCPColorMap(plot.xAxis,plot.yAxis);fillMap(map,72,52);
    map->setGradient(customGradient());map->setInterpolate(true);map->setTightBoundary(true);
    return save(plot,path);
}

bool renderColorMapInterpolate(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · cell values without interpolation");
    plot.xAxis->setLabel("grid coordinate x");plot.yAxis->setLabel("grid coordinate y");plot.xAxis->setRange(-2.3,2.3);plot.yAxis->setRange(-1.8,1.8);
    auto *map=new QCPColorMap(plot.xAxis,plot.yAxis);fillMap(map,8,6);
    map->setGradient(customGradient());map->setInterpolate(false);map->setTightBoundary(true);
    return save(plot,path);
}

bool renderColorMapTight(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · exact cell-boundary extents");
    plot.xAxis->setLabel("bounded key coordinate");plot.yAxis->setLabel("bounded value coordinate");
    auto *map=new QCPColorMap(plot.xAxis,plot.yAxis);fillMap(map,6,5);
    map->setGradient(customGradient());map->setInterpolate(false);map->setTightBoundary(true);
    plot.xAxis->setRange(-2.5,2.5);plot.yAxis->setRange(-2,2);
    auto *note=new QCPItemText(&plot);note->position->setType(QCPItemPosition::ptAxisRectRatio);note->position->setCoords(.03,.04);note->setText("6 × 5 cells; bounds are explicit");note->setColor(ink);note->setFont(QFont("DejaVu Sans",9));
    return save(plot,path);
}

bool renderGradient(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · custom four-stop color ramp");
    plot.xAxis->setLabel("position x");plot.yAxis->setLabel("position y");plot.xAxis->setRange(-2,2);plot.yAxis->setRange(-1.5,1.5);
    auto *map=new QCPColorMap(plot.xAxis,plot.yAxis);fillMap(map,48,32);
    map->setGradient(customGradient());map->setInterpolate(true);map->setTightBoundary(true);
    return save(plot,path);
}

bool renderGradientLevels(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · low versus dense gradient quantization");
    plot.xAxis->setLabel("position x");plot.yAxis->setLabel("position y");plot.xAxis->setRange(-2,2);plot.yAxis->setRange(-1.5,1.5);
    auto *map=new QCPColorMap(plot.xAxis,plot.yAxis);fillMap(map,54,38);
    QCPColorGradient gradient=customGradient();gradient.setLevelCount(9);
    map->setGradient(gradient);map->setInterpolate(true);map->setTightBoundary(true);
    return save(plot,path);
}

bool renderGradientPeriodic(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · periodic hue for wrapped phase data");
    plot.xAxis->setLabel("in-phase component");plot.yAxis->setLabel("quadrature component");plot.xAxis->setRange(-1.1,1.1);plot.yAxis->setRange(-1.1,1.1);
    auto *map=new QCPColorMap(plot.xAxis,plot.yAxis);map->data()->setSize(72,72);map->data()->setRange(QCPRange(-1,1),QCPRange(-1,1));
    for(int y=0;y<72;++y)for(int x=0;x<72;++x){double xx=-1+2.0*(x+.5)/72,yy=-1+2.0*(y+.5)/72;map->data()->setCell(x,y,qAtan2(yy,xx));}
    map->setDataRange(QCPRange(-3.141592653589793,3.141592653589793));
    QCPColorGradient gradient(QCPColorGradient::gpHues);gradient.setPeriodic(true);
    map->setGradient(gradient);map->setInterpolate(true);map->setTightBoundary(true);
    return save(plot,path);
}

bool renderColorScale(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · shared scale and scalar field");
    plot.xAxis->setLabel("distance x");plot.yAxis->setLabel("distance y");plot.xAxis->setRange(-2,2);plot.yAxis->setRange(-1.5,1.5);
    auto *map=new QCPColorMap(plot.xAxis,plot.yAxis);fillMap(map,52,36);
    auto *scale=new QCPColorScale(&plot);scale->setType(QCPAxis::atRight);scale->setDataRange(map->dataRange());scale->setGradient(customGradient());scale->setLabel("field value");scale->setBarWidth(24);
    plot.plotLayout()->addElement(1,1,scale);map->setColorScale(scale);
    return save(plot,path);
}

bool renderAxisNames(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · four named axes around one viewport");
    plot.xAxis->setLabel("primary horizontal · seconds");plot.yAxis->setLabel("primary vertical · amplitude");
    plot.xAxis->setRange(0,10);plot.yAxis->setRange(-1,1);
    auto *top=plot.axisRect()->addAxis(QCPAxis::atTop);top->setRange(0,100);top->setLabel("secondary horizontal · distance");top->setLabelColor(blue);top->setBasePen(QPen(blue));top->setTickLabelColor(blue);top->setTickPen(QPen(blue));
    auto *right=plot.axisRect()->addAxis(QCPAxis::atRight);right->setRange(-100,100);right->setLabel("secondary vertical · percent");right->setLabelColor(orange);right->setBasePen(QPen(orange));right->setTickLabelColor(orange);right->setTickPen(QPen(orange));
    auto *g=plot.addGraph(plot.xAxis,plot.yAxis);QVector<double>x,y;for(int i=0;i<=60;++i){double t=i/6.0;x<<t;y<<.78*qSin(t*.8);}g->setData(x,y);g->setPen(QPen(teal,2.5));
    return save(plot,path);
}

bool renderAxisRectSpacing(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · axis labels reserve plot margins");
    plot.xAxis->setLabel("short key label");plot.yAxis->setLabel("a deliberately longer value descriptor");
    plot.xAxis->setRange(0,10);plot.yAxis->setRange(0,10);
    plot.yAxis->setLabelPadding(22);plot.xAxis->setLabelPadding(18);
    plot.yAxis->setTickLabelPadding(9);plot.xAxis->setTickLabelPadding(8);
    auto *top=plot.axisRect()->addAxis(QCPAxis::atTop);top->setRange(0,10);top->setLabel("offset top axis");top->setOffset(18);top->setLabelColor(blue);top->setTickLabelColor(blue);top->setBasePen(QPen(blue));
    auto *g=plot.addGraph(plot.xAxis,plot.yAxis);g->setData(QVector<double>{0,2,4,6,8,10},QVector<double>{2,7,4,8,3,6});g->setPen(QPen(orange,2.5));
    return save(plot,path);
}

bool renderMarginGroup(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);plot.setBackground(canvas);
    plot.plotLayout()->insertRow(0);auto *title=new QCPPlotTitle(&plot,"Original example · coupled left margins align two plots");title->setFont(QFont("DejaVu Sans",13,QFont::DemiBold));title->setTextColor(ink);plot.plotLayout()->addElement(0,0,title);
    plot.plotLayout()->insertRow(2);
    auto *top=plot.axisRect();auto *bottom=new QCPAxisRect(&plot);
    plot.plotLayout()->addElement(2,0,bottom);
    auto *group=new QCPMarginGroup(&plot);top->setMarginGroup(QCP::msLeft,group);bottom->setMarginGroup(QCP::msLeft,group);
    top->axis(QCPAxis::atLeft)->setLabel("density");top->axis(QCPAxis::atBottom)->setLabel("time");
    bottom->axis(QCPAxis::atLeft)->setLabel("longer-unit-description");bottom->axis(QCPAxis::atBottom)->setLabel("time");
    top->axis(QCPAxis::atBottom)->setRange(0,10);bottom->axis(QCPAxis::atBottom)->setRange(0,10);
    top->axis(QCPAxis::atLeft)->setRange(0,10);bottom->axis(QCPAxis::atLeft)->setRange(-5,5);
    auto *a=new QCPGraph(top->axis(QCPAxis::atBottom),top->axis(QCPAxis::atLeft));a->setData(QVector<double>{0,2,4,6,8,10},QVector<double>{2,5,3,7,4,8});a->setPen(QPen(teal,2.2));
    auto *b=new QCPGraph(bottom->axis(QCPAxis::atBottom),bottom->axis(QCPAxis::atLeft));b->setData(QVector<double>{0,2,4,6,8,10},QVector<double>{-4,1,-2,3,-1,4});b->setPen(QPen(orange,2.2));
    return save(plot,path);
}

bool renderAdaptiveLine(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · dense line rendered with adaptive sampling");
    plot.xAxis->setLabel("sample coordinate");plot.yAxis->setLabel("signal");plot.xAxis->setRange(0,100);plot.yAxis->setRange(-1.2,1.2);
    QVector<double>x,y;for(int i=0;i<6000;++i){double t=100.0*i/5999;x<<t;y<<.73*qSin(t*1.24)+.16*qSin(t*9.1)+.04*qCos(t*19.7);}
    auto *g=plot.addGraph();g->setData(x,y);g->setAdaptiveSampling(true);g->setPen(QPen(teal,1.8));g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone));
    return save(plot,path);
}

bool renderAdaptiveScatter(const QString &path) {
    QCustomPlot plot;plot.resize(760,450);
    stylePlot(plot,"Original example · dense point cloud uses adaptive rendering");
    plot.xAxis->setLabel("sample coordinate");plot.yAxis->setLabel("residual");plot.xAxis->setRange(0,100);plot.yAxis->setRange(-1.2,1.2);
    QVector<double>x,y;for(int i=0;i<2400;++i){double t=100.0*i/2399;x<<t;y<<.9*qSin(t*.53)+.2*qCos(t*7.1);}
    auto *g=plot.addGraph();g->setData(x,y);g->setAdaptiveSampling(true);g->setLineStyle(QCPGraph::lsNone);g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCross,teal,3.5));
    return save(plot,path);
}


}  // namespace

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    if (argc != 2) {
        std::cerr << "usage: qcp_original_illustrations OUTPUT_DIR\n";
        return 2;
    }
    const QString output = QString::fromLocal8Bit(argv[1]);
    if (!QDir().mkpath(output)) return 3;
    struct Output { const char *name; bool (*render)(const QString &); };
    const Output outputs[] = {
        {"AxisNamesOverview.png", renderAxisNames},
        {"AxisRectSpacingOverview.png", renderAxisRectSpacing},
        {"QCPBars.png", renderBars},
        {"QCPBarsGroup.png", renderBarsGroup},
        {"QCPColorGradient-levelcount.png", renderGradientLevels},
        {"QCPColorGradient-periodic.png", renderGradientPeriodic},
        {"QCPColorGradient.png", renderGradient},
        {"QCPColorMap-interpolate.png", renderColorMapInterpolate},
        {"QCPColorMap-tightboundary.png", renderColorMapTight},
        {"QCPColorMap.png", renderColorMap},
        {"QCPColorScale.png", renderColorScale},
        {"QCPCurve.png", renderCurve},
        {"QCPFinancial.png", renderFinancial},
        {"QCPGraph.png", renderGraph},
        {"QCPItemBracket-length.png", renderBracketLength},
        {"QCPItemBracket.png", renderItemBracket},
        {"QCPItemCurve.png", renderItemCurve},
        {"QCPItemEllipse.png", renderItemEllipse},
        {"QCPItemLine.png", renderItemLine},
        {"QCPItemPixmap.png", renderItemPixmap},
        {"QCPItemRect.png", renderItemRect},
        {"QCPItemStraightLine.png", renderItemStraightLine},
        {"QCPItemText.png", renderItemText},
        {"QCPItemTracer.png", renderItemTracer},
        {"QCPLineEnding.png", renderLineEnding},
        {"QCPMarginGroup.png", renderMarginGroup},
        {"QCPStatisticalBox.png", renderStatisticalBox},
        {"adaptive-sampling-line.png", renderAdaptiveLine},
        {"adaptive-sampling-scatter.png", renderAdaptiveScatter},
    };
    for (const Output &output : outputs) {
        if (!output.render(QDir(argv[1]).filePath(output.name))) {
            std::cerr << "failed to render " << output.name << "\\n";
            return 10;
        }
    }
    return 0;
}
