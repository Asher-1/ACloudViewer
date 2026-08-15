// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QComboBox>
#include <QCursor>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <algorithm>
#include <filesystem>

#include "BagDiscovery.h"
#include "RosBagReader.h"
#include "mcalib_portability.h"

namespace mcalib {
namespace ui {

inline QString initialBagBrowsePath(const QString& last_path) {
    if (last_path.isEmpty()) return QString();
    const QFileInfo info(last_path);
    if (info.isDir()) return info.absoluteFilePath();
    return info.absolutePath();
}

inline QString pickBagInputPath(QWidget* parent, const QString& last_path) {
    QMenu menu(parent);
    menu.setTitle(QObject::tr("Open ROS Bag"));

    QString selected;
    const QString browse_path = initialBagBrowsePath(last_path);

    auto pick_file = [&]() {
        selected = QFileDialog::getOpenFileName(
                parent, QObject::tr("Open ROS Bag File"), browse_path,
                QObject::tr("ROS Bag (*.bag)"));
    };
    auto pick_dir = [&]() {
        selected = QFileDialog::getExistingDirectory(
                parent, QObject::tr("Open ROS Bag Directory"), browse_path,
                QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    };

    menu.addAction(QObject::tr("Bag file..."), pick_file);
    menu.addAction(QObject::tr("Bag directory..."), pick_dir);
    menu.exec(QCursor::pos());
    return selected;
}

inline int pickBagSessionIndex(QWidget* parent,
                               const BagDiscoveryResult& discovery,
                               const std::string& preferred_session_key = {}) {
    if (discovery.sessions.size() <= 1) return 0;

    QDialog dialog(parent);
    dialog.setWindowTitle(QObject::tr("Select Bag Session"));
    dialog.setModal(true);

    auto* layout = new QVBoxLayout(&dialog);
    layout->addWidget(new QLabel(QObject::tr(
            "Multiple recording sessions were found. Select one to load:")));

    auto* combo = new QComboBox(&dialog);
    int default_index = static_cast<int>(discovery.sessions.size()) - 1;
    for (size_t i = 0; i < discovery.sessions.size(); ++i) {
        const auto& session = discovery.sessions[i];
        QString label = QString::fromStdString(session.session_key);
        label += QObject::tr(" (%1 bags)").arg(session.bag_paths.size());
        combo->addItem(label);
        if (!preferred_session_key.empty() &&
            session.session_key == preferred_session_key) {
            default_index = static_cast<int>(i);
        }
    }
    combo->setCurrentIndex(default_index);
    layout->addWidget(combo);

    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal,
            &dialog);
    layout->addWidget(buttons);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog,
                     &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog,
                     &QDialog::reject);

    if (dialog.exec() != QDialog::Accepted) return -1;
    return combo->currentIndex();
}

inline bool openResolvedBag(mcalib::RosBagReader& reader,
                            const mcalib::BagResolveResult& resolved) {
    if (!resolved.ok || resolved.source_bags.empty()) return false;
    // Pre-merged single bag, or legacy one-file session.
    if (resolved.source_bags.size() == 1 ||
        resolved.layout == mcalib::BagLayoutType::SingleFile) {
        return reader.open(resolved.source_bags.front());
    }
    return reader.openMulti(resolved.source_bags);
}

inline void removeMergedBagTempFile(const std::string& path) {
    if (path.empty()) return;
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

inline QString layoutDescription(BagLayoutType layout) {
    switch (layout) {
        case BagLayoutType::SingleFile:
            return QObject::tr("merged/single bag");
        case BagLayoutType::FlatTopicGroup:
            return QObject::tr("flat topic-group bags");
        case BagLayoutType::NestedTopicGroup:
            return QObject::tr("nested topic-group bags");
        case BagLayoutType::LegacyMultiBag:
            return QObject::tr("legacy multi-bag directory");
        default:
            return QObject::tr("unknown layout");
    }
}

inline std::string preferredSessionKeyFromInput(const std::string& input_path) {
    namespace fs = std::filesystem;
    const fs::path input(input_path);
    if (!fs::is_regular_file(input) || input.extension() != ".bag") {
        return {};
    }
    const std::string stem = input.stem().string();
    std::string session_key = extractTopicGroupSessionKey(stem);
    if (session_key.empty()) {
        session_key = stem;
    }
    return session_key;
}

}  // namespace ui
}  // namespace mcalib
