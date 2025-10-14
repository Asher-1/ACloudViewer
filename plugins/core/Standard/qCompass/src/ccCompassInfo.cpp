// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccCompassInfo.h"

#include <QFile>
#include <QTextEdit>
#include <QTextStream>
#include <QVBoxLayout>

ccCompassInfo::ccCompassInfo(QWidget *parent) : QDialog(parent) {
    setFixedSize(800, 600);

    // setup GUI components
    QTextEdit *l = new QTextEdit;
    l->acceptRichText();
    l->setReadOnly(true);

    QDialogButtonBox *buttonBox =
            new QDialogButtonBox(QDialogButtonBox::Ok, Qt::Horizontal);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);

    QVBoxLayout *lt = new QVBoxLayout;
    lt->addWidget(l);
    lt->addWidget(buttonBox);
    setLayout(lt);

    // load text
    QFile file(":/CC/plugin/qCompass/info.html");
    if (file.open(QIODevice::ReadOnly)) {
        QTextStream in(&file);
        QString html = in.readAll();
        l->setText(html);
    } else {
        l->setText("Error loading documentation file...");
    }
}
