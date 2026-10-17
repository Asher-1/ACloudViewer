// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QJsonObject>

class QLineEdit;
class QListWidget;
class QVBoxLayout;

/** Source-compatible label editing dialog: the text field doubles as the
 *  list filter (label_completion), the list can be sorted (sort_labels),
 *  the text field can be hidden (show_label_text_field), and the dialog
 *  resizes to content (fit_to_content.column/row). When a Label Flags
 *  template defines flags for a label, checkbox fields are rendered for
 *  the shape being labeled, mirroring the source label dialog. */
class AnnotationLabelDialog final : public QDialog {
    Q_OBJECT
public:
    explicit AnnotationLabelDialog(const QStringList& labels,
                                   bool sortLabels,
                                   bool showTextField,
                                   const QString& completion,
                                   bool fitToColumn,
                                   bool fitToRow,
                                   const QString& selected,
                                   QWidget* parent = nullptr);

    /** Render a checkbox per flag for the labeled shape; checked states
     *  initialize from the shape's current flags object. */
    void setFlagFields(const QStringList& flags,
                       const QJsonObject& currentFlags);
    QString selectedLabel() const;
    QJsonObject selectedFlags() const;

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    void rebuildList();

    QLineEdit* m_edit = nullptr;
    QListWidget* m_list = nullptr;
    QVBoxLayout* m_flagLayout = nullptr;
    QStringList m_labels;
    QStringList m_flagNames;
    QJsonObject m_currentFlags;
    bool m_sortLabels = true;
    QString m_completion;
};
