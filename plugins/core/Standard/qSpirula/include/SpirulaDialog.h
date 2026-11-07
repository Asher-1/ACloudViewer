// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

class QLabel;
class QPushButton;

//! Single entry dialog for the qSpirula plugin.
//!
//! The plugin exposes ONE toolbar action (matching the qFreeSplatter pattern);
//! both capabilities — launching the official upstream GUI and importing its
//! outputs into the DB tree — live here as buttons, so the toolbar stays
//! uncluttered and the upstream icon appears exactly once.
class SpirulaDialog : public QDialog {
    Q_OBJECT

public:
    explicit SpirulaDialog(QWidget* parent = nullptr);

    //! Refreshes the status line from SpirulaLocator (called before show()).
    void refreshStatus();

signals:
    void launchRequested();
    void importRequested();
    void trainRequested();

private:
    QLabel* m_statusLabel = nullptr;
    QPushButton* m_launchButton = nullptr;
    QPushButton* m_importButton = nullptr;
    QPushButton* m_trainButton = nullptr;
};
