// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SpirulaDialog.h"

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <algorithm>

#include "SpirulaLocator.h"

SpirulaDialog::SpirulaDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("Spirula Studio"));
    setModal(false);

    auto* layout = new QVBoxLayout(this);

    auto* titleLabel =
            new QLabel(tr("<b>Spirula Studio</b> — 3D Gaussian Splatting "
                          "trainer (official upstream GUI)"),
                       this);
    layout->addWidget(titleLabel);

    m_statusLabel = new QLabel(this);
    m_statusLabel->setWordWrap(true);
    layout->addWidget(m_statusLabel);

    m_launchButton = new QPushButton(tr("Launch Spirula Studio GUI"), this);
    m_launchButton->setToolTip(
            tr("Start the official Spirula Studio application "
               "(training, editing, rendering, meshing)"));
    layout->addWidget(m_launchButton);

    m_importButton = new QPushButton(tr("Import Result to DB Tree…"), this);
    m_importButton->setToolTip(
            tr("Add a trained splat.ply / mesh.ply (or a run "
               "directory) to the ACloudViewer DB tree"));
    layout->addWidget(m_importButton);

    m_trainButton = new QPushButton(tr("Run Headless Training…"), this);
    m_trainButton->setToolTip(
            tr("Drive `spirula train` headlessly with live output and "
               "cancel support"));
    layout->addWidget(m_trainButton);

    auto* licenseLabel =
            new QLabel(tr("Standalone GPLv3 executable (zero patches; "
                          "sync = SPIRULA_TAG bump + smoke tests)."),
                       this);
    licenseLabel->setWordWrap(true);
    QFont licenseFont = licenseLabel->font();
    licenseFont.setPointSize(std::max(licenseFont.pointSize() - 1, 7));
    licenseLabel->setFont(licenseFont);
    layout->addWidget(licenseLabel);

    connect(m_launchButton, &QPushButton::clicked, this,
            &SpirulaDialog::launchRequested);
    connect(m_importButton, &QPushButton::clicked, this,
            &SpirulaDialog::importRequested);
    connect(m_trainButton, &QPushButton::clicked, this,
            &SpirulaDialog::trainRequested);
}

void SpirulaDialog::refreshStatus() {
    const QString exePath = spirula::locateExecutable();
    if (exePath.isEmpty()) {
        m_statusLabel->setText(
                tr("<span style=\"color:#c00\">Spirula Studio "
                   "executable not found.</span> Build it with "
                   "-DBUILD_SPIRULA=ON or place 'spirula' on the "
                   "PATH."));
        m_launchButton->setEnabled(false);
        return;
    }
    m_statusLabel->setText(tr("Executable: <code>%1</code>").arg(exePath));
    m_launchButton->setEnabled(true);
}
