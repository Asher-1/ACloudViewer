// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QCheckBox>
#include <QDialog>
#include <QFile>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>
#include <algorithm>

#include "AnnotationShapeBatch.h"
#include "AnnotationWindow.h"

void AnnotationWindow::editShapes() {
    if (m_files->count() == 0 || m_image.isNull() || m_inferenceRunning ||
        m_batchRunning || !confirmSave())
        return;
    QJsonArray selected;
    for (int index : m_canvas->selectedIndices())
        selected.append(m_canvas->shapes()[index].toJson());
    QStringList images;
    for (int row = 0; row < m_files->count(); ++row)
        images.append(m_files->item(row)->data(Qt::UserRole).toString());

    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("AnnotationShapeManager"));
    dialog.setWindowTitle(tr("Shape Manager"));
    dialog.setFixedSize(480, 240);
    auto* layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(24, 24, 24, 24);
    const QStringList names{tr("Delete All Annotations"),
                            tr("Delete All Images with Annotations"),
                            tr("Remove Selected Shapes"),
                            tr("Add Selected Shapes")};
    QVector<QCheckBox*> choices;
    for (int i = 0; i < names.size(); ++i) {
        auto* choice = new QCheckBox(names[i], &dialog);
        choice->setObjectName(
                QStringLiteral("AnnotationShapeOperation%1").arg(i));
        choice->setEnabled(i < 2 || !selected.isEmpty());
        layout->addWidget(choice);
        choices.append(choice);
    }
    for (auto* choice : choices)
        connect(choice, &QCheckBox::toggled, &dialog,
                [choices, choice](bool checked) {
                    if (checked)
                        for (auto* other : choices)
                            if (other != choice) other->setChecked(false);
                });
    layout->addStretch();
    auto* range = new QHBoxLayout;
    auto* from = new QSpinBox(&dialog);
    from->setObjectName(QStringLiteral("AnnotationShapeFrom"));
    from->setRange(1, images.size());
    from->setValue(std::max(1, m_files->currentRow() + 1));
    auto* to = new QSpinBox(&dialog);
    to->setObjectName(QStringLiteral("AnnotationShapeTo"));
    to->setRange(0, images.size());
    to->setSpecialValueText(QStringLiteral(" "));
    from->setFixedWidth(104);
    to->setFixedWidth(104);
    range->addWidget(new QLabel(tr("From:"), &dialog));
    range->addWidget(from);
    range->addWidget(new QLabel(tr("To:"), &dialog));
    range->addWidget(to);
    auto* go = new QPushButton(tr("Go"), &dialog);
    go->setObjectName(QStringLiteral("AnnotationShapeGo"));
    range->addWidget(go);
    layout->addLayout(range);
    AnnotationShapeBatchResult result;
    auto operation = AnnotationShapeBatchOperation::DeleteAnnotations;
    connect(go, &QPushButton::clicked, &dialog, [&]() {
        if (to->value() == 0 || from->value() > to->value()) {
            QMessageBox::warning(
                    &dialog, tr("Invalid Range"),
                    to->value() == 0 ? tr("Please specify the end frame index")
                                     : tr("Start index cannot be greater than "
                                          "end index"));
            return;
        }
        int choice = -1;
        for (int i = 0; i < choices.size(); ++i)
            if (choices[i]->isChecked()) choice = i;
        if (choice < 0) {
            QMessageBox::information(
                    &dialog, tr("No Operation"),
                    tr("Please select an operation to perform"));
            return;
        }
        if (QMessageBox::question(&dialog, tr("Confirm Operation"),
                                  tr("Are you sure you want to perform '%1' on "
                                     "frames %2 to %3?")
                                          .arg(names[choice])
                                          .arg(from->value())
                                          .arg(to->value()),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::No) != QMessageBox::Yes)
            return;
        operation = static_cast<AnnotationShapeBatchOperation>(choice);
        go->setEnabled(false);
        QProgressDialog progress(tr("Processing..."), tr("Cancel"), 0,
                                 to->value() - from->value() + 1, &dialog);
        progress.setWindowModality(Qt::WindowModal);
        progress.setMinimumWidth(400);
        result = applyAnnotationShapeBatch(images, m_outputDir, from->value(),
                                           to->value(), operation, selected,
                                           [&](int done, int) {
                                               progress.setValue(done);
                                               QApplication::processEvents();
                                               return !progress.wasCanceled();
                                           });
        dialog.accept();
    });
    if (dialog.exec() != QDialog::Accepted) return;
    // Reload from disk after batch changes; these are not canvas undo entries.
    const QString current = m_imagePath;
    const bool movedImages =
            operation == AnnotationShapeBatchOperation::DeleteImages &&
            result.files;
    if (movedImages) {
        closeImage();
        refreshFileList();
        if (m_files->count()) m_files->setCurrentRow(0);
    } else {
        openFile(m_openedLabelPath.isEmpty() ||
                                 !QFile::exists(m_openedLabelPath)
                         ? current
                         : m_openedLabelPath);
        updateFileStatus();
    }
    m_status->setText(
            tr("Processed %1 files, %2 shapes; skipped %3 out-of-bounds "
               "shapes%4")
                    .arg(result.files)
                    .arg(result.shapes)
                    .arg(result.skipped)
                    .arg(result.canceled ? tr(" (canceled)") : QString()));
    if (!result.errors.isEmpty())
        QMessageBox::warning(
                this, tr("Shape Manager"),
                tr("Unable to process:\n%1").arg(result.errors.join('\n')));
}
