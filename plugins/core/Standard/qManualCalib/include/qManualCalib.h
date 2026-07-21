// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvStdPluginInterface.h"

class ManualSensorCalibDlg;
class ManualAvmAdjustDlg;

class qManualCalib : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qManualCalib" FILE
                          "../info.json")

public:
    explicit qManualCalib(QObject* parent = nullptr);
    ~qManualCalib() override = default;

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

protected slots:
    void doSensorCalib();
    void doAvmAdjust();

protected:
    QAction* m_actionSensorCalib = nullptr;
    QAction* m_actionAvmAdjust = nullptr;

    ManualSensorCalibDlg* m_sensorCalibDlg = nullptr;
    ManualAvmAdjustDlg* m_avmAdjustDlg = nullptr;
};
