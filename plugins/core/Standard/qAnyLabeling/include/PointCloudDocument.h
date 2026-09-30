// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QString>
#include <QVector4D>
#include <QVector>

struct PointCloudFrame {
    QString path;
    QString labelPath;
    QVector<QVector4D> points;
    QVector<QColor> rgb;
    QVector<quint32> labels;
    bool hasIntensity = true;
    bool labelExists = false;
};

bool loadPointCloudFrame(const QString& path,
                         PointCloudFrame* frame,
                         QString* error,
                         const QString& labelPath = QString());
bool savePointCloudLabels(const PointCloudFrame& frame,
                          QString* error,
                          const QString& outputPath = QString());

class PointCloudDocument {
public:
    enum Operation {
        AssignSemantic,
        CreateInstance,
        AddToInstance,
        RemoveFromInstance,
        SplitInstance
    };
    void setFrame(PointCloudFrame frame);
    const PointCloudFrame& frame() const { return m_frame; }
    void setLabelPath(const QString& path) {
        m_frame.labelPath = path;
        m_frame.labelExists = true;
    }
    bool dirty() const { return m_dirty; }
    bool canUndo() const { return !m_undo.isEmpty(); }
    bool canRedo() const { return !m_redo.isEmpty(); }
    void markSaved();
    bool assign(const QVector<int>& indices,
                quint16 semantic,
                quint16 instance);
    bool operate(const QVector<int>& indices,
                 quint16 semantic,
                 quint16 instance,
                 Operation operation,
                 QString* error = nullptr);
    bool undo();
    bool redo();

private:
    struct Edit {
        QVector<int> indices;
        QVector<quint32> before;
        QVector<quint32> after;
    };
    void apply(const Edit& edit, bool forward);
    PointCloudFrame m_frame;
    QVector<quint32> m_baseline;
    QVector<Edit> m_undo;
    QVector<Edit> m_redo;
    bool m_dirty = false;
};
