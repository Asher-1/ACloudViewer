// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvStdPluginInterface.h>

#include <QTranslator>
#include <memory>

class QAction;
class SpirulaDialog;
class SpirulaTrainDialog;

//! qSpirula — process-boundary integration of Spirula Studio, the upstream
//! 3D Gaussian Splatting trainer (standalone `spirula` executable, GPLv3).
//!
//! The plugin owns no training logic. It exposes ONE toolbar action (the
//! official upstream icon) opening a dialog whose buttons launch the official
//! GUI binary and import the training outputs (splat.ply / mesh.ply) into the
//! DB tree, so the official UI evolution rides the SPIRULA_TAG upgrade for
//! free and the plugin carries no source patches.
class qSpirula : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qSpirula" FILE
                          "../info.json")

public:
    explicit qSpirula(QObject* parent = nullptr);
    // defined out-of-line: m_dialog is a unique_ptr of a forward-declared type
    ~qSpirula() override;

    // inherited from ccStdPluginInterface
    QList<QAction*> getActions() override;

private slots:
    //! Open the Spirula Studio dialog (launch + import entries live there).
    void showDialog();
    //! Start the official Spirula Studio GUI (startDetached — the plugin does
    //! not own the process; training results cross back as files).
    void launchSpirulaStudio();

    //! Import a specific training output (run directory / splat.ply /
    //! mesh.ply) — the shared entry point for the dialog picker and the
    //! headless-training hand-off.
    void importPath(const QString& path);

    //! Pick a training output (run directory / step-*.ckpt / splat.ply /
    //! mesh.ply) and add it to the DB tree.
    void importResult();

private:
    QAction* m_action = nullptr;
    QTranslator m_translator;
    std::unique_ptr<SpirulaDialog> m_dialog;
    std::unique_ptr<SpirulaTrainDialog> m_trainDialog;
};
