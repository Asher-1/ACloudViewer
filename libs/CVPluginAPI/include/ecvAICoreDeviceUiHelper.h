// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Device-combo projection helpers for AICore-based plugin dialogs.
//
// Split from ecvAICoreUiHelper.h on purpose: that header must stay
// Qt-only (optional consumers such as video_base include it without an
// AICore include path), while these helpers consume the public
// aicore/backend_capi.h registry and therefore belong only to targets
// that link AICore. Header-only: no out-of-line symbols, nothing to
// link beyond libAICore itself.
// ----------------------------------------------------------------------------

#pragma once

#include <QComboBox>
#include <QStandardItemModel>
#include <QString>

#include "aicore/backend_capi.h"

namespace ecvAICoreUi {

/// Fill a device combo from the AICore runtime backend registry
/// (aicore/backend_capi.h). This is the SINGLE projection point of the
/// platform's registered backends: the combo lists only devices the core
/// can resolve (Metal -> CPU on macOS, CUDA/Vulkan -> CPU on
/// Linux/Windows per the build) — never a hardcoded list. Item data
/// carries the id passed to the C API; the label is display-only and the
/// registry default ("auto") is pre-selected. A default tooltip describes
/// the auto resolution order; callers may override it afterwards with
/// task-specific semantics.
inline void populateDeviceCombo(QComboBox* combo) {
    if (!combo) return;
    for (int i = 0; i < aicore_device_count(); ++i) {
        const aicore_device_info* dev = aicore_device_at(i);
        if (!dev || !dev->id) continue;
        combo->addItem(QString::fromUtf8(dev->label),
                       QString::fromUtf8(dev->id));
        if (dev->is_default) combo->setCurrentIndex(i);
    }
    combo->setToolTip(
            QObject::tr("Inference backend. 'auto' resolves to %1; a device "
                        "the runtime cannot resolve fails context "
                        "admission.")
                    .arg(QString::fromUtf8(aicore_auto_device_order())));
}

/// Model-aware variant: devices the task's C ABI rejects through
/// aicore_model_device_info_query stay VISIBLE but DISABLED, with the
/// rejection reason as their tooltip. Device support is a runtime fact
/// owned by the AICore task ABI — never hand-write a device whitelist
/// in a plugin.
inline void populateDeviceCombo(QComboBox* combo, aicore_model_kind model) {
    if (!combo) return;
    for (int i = 0; i < aicore_device_count(); ++i) {
        const aicore_device_info* dev = aicore_device_at(i);
        if (!dev || !dev->id) continue;
        combo->addItem(QString::fromUtf8(dev->label),
                       QString::fromUtf8(dev->id));
        aicore_model_device_info info;
        info.struct_size = sizeof(info);
        info.abi_version = 1;
        if (aicore_model_device_info_query(model, dev->id, &info) != 0) {
            const QString reason = QString::fromUtf8(
                    aicore_backend_last_error());
            if (auto* items = qobject_cast<QStandardItemModel*>(
                        combo->model())) {
                if (auto* item = items->item(combo->count() - 1)) {
                    item->setEnabled(false);
                    if (!reason.isEmpty()) {
                        item->setToolTip(reason);
                    }
                }
            }
        }
        if (dev->is_default) combo->setCurrentIndex(i);
    }
    combo->setToolTip(
            QObject::tr("Inference backend. 'auto' resolves to %1; a device "
                        "the runtime cannot resolve fails context "
                        "admission.")
                    .arg(QString::fromUtf8(aicore_auto_device_order())));
}

/// Device id of the combo's current item, ready for the C API. Returns
/// "auto" on a null combo or an empty item data value — never leak a
/// display label into a runtime device request.
inline QString currentDeviceId(QComboBox* combo) {
    if (!combo) return QStringLiteral("auto");
    const QString id = combo->currentData().toString();
    return id.isEmpty() ? QStringLiteral("auto") : id;
}

/// Restore a persisted device id by identity (item data). Unknown ids
/// (e.g. a device the current build does not register) keep the registry
/// default instead of failing.
inline void restoreDeviceCombo(QComboBox* combo, const QString& deviceId) {
    if (!combo) return;
    const int idx = combo->findData(deviceId);
    if (idx >= 0) combo->setCurrentIndex(idx);
}

}  // namespace ecvAICoreUi
