// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QFrame>
#include <QString>
class QTableWidget;
namespace PpocrTable {
bool isTable(const QString& label, const QString& content);
void load(QTableWidget* table, const QString& content);
QString tokens(const QTableWidget* table);
}  // namespace PpocrTable
class PpocrTableEditor final : public QFrame {
    Q_OBJECT
public:
    explicit PpocrTableEditor(const QString& content,
                              QWidget* parent = nullptr);
    QString content() const;

signals:
    void saveRequested(const QString& content);
    void cancelRequested();
    void geometryChanged();

protected:
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    void scheduleGeometry();
    bool m_geometryPending = false;
    QTableWidget* m_table;
};
