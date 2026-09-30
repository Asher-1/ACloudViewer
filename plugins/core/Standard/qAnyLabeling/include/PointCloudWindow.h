// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QMainWindow>
#include <QMap>
#include <QPointer>
#include <QSet>

#include "PointCloudDocument.h"

class QListWidget;
class QComboBox;
class QSpinBox;
class QLabel;
class QListWidgetItem;
class PointCloudView;

class PointCloudWindow final : public QMainWindow {
public:
    explicit PointCloudWindow(QWidget* parent = nullptr);
    void openFile(const QString& path);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void openDirectory();
    void selectFrame(int index);
    void save();
    void saveAs();
    bool confirmSaved();
    void applySelection(const QVector<int>& indices);
    void refresh();
    void updateVisibility();
    void loadClasses();
    void saveClasses();
    void addClass();
    void editClass();
    void removeClass();
    void editClassDefinition(bool editing);
    void refreshClassCombo(quint16 selected = 0);
    bool classConfigDirty() const;
    bool writeClassDefinitions(const QString& path, QString* error) const;

    PointCloudDocument m_document;
    QStringList m_files;
    QListWidget* m_fileList = nullptr;
    QListWidget* m_classList = nullptr;
    QListWidget* m_instanceList = nullptr;
    PointCloudView* m_view = nullptr;
    QComboBox* m_class = nullptr;
    QComboBox* m_operation = nullptr;
    QSpinBox* m_instance = nullptr;
    QLabel* m_status = nullptr;
    QString m_outputDirectory;
    bool m_changingFile = false;
    QSet<quint16> m_hiddenSemantics;
    QSet<quint16> m_hiddenInstances;
    QMap<quint16, QString> m_classNames{{0, QStringLiteral("Unlabeled")}};
    QMap<quint16, QColor> m_classColors{{0, QColor(QStringLiteral("#808080"))}};
    QMap<quint16, QString> m_savedClassNames{{0, QStringLiteral("Unlabeled")}};
    QMap<quint16, QColor> m_savedClassColors{
            {0, QColor(QStringLiteral("#808080"))}};
    QString m_classConfigPath;
};
