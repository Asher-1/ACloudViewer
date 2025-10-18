// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGraphicalSegmentationOptionsDlg.h"

// Qt
#include <QSettings>

ccGraphicalSegmentationOptionsDlg::ccGraphicalSegmentationOptionsDlg(
        const QString windowTitle /*=QString()*/, QWidget* parent /*=nullptr*/)
    : QDialog(parent, Qt::Tool), Ui::GraphicalSegmentationOptionsDlg() {
    setupUi(this);

    QSettings settings;
    settings.beginGroup(SegmentationToolOptionsKey());
    QString remainingSuffix =
            settings.value(RemainingSuffixKey(), ".remaining").toString();
    QString segmentedSuffix =
            settings.value(SegmentedSuffixKey(), ".segmented").toString();
    settings.endGroup();

    remainingTextLineEdit->setText(remainingSuffix);
    segmentedTextLineEdit->setText(segmentedSuffix);

    if (!windowTitle.isEmpty()) {
        setWindowTitle(windowTitle);
    }
}

void ccGraphicalSegmentationOptionsDlg::accept() {
    QSettings settings;
    settings.beginGroup(SegmentationToolOptionsKey());
    settings.setValue(RemainingSuffixKey(), remainingTextLineEdit->text());
    settings.setValue(SegmentedSuffixKey(), segmentedTextLineEdit->text());
    settings.endGroup();

    QDialog::accept();
}
