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
#include <functional>

#include "PointCloudCuboid.h"

struct PointCloudFrame {
    QString path;
    QString labelPath;
    QVector<QVector4D> points;
    QVector<QColor> rgb;
    QVector<quint32> labels;
    bool hasIntensity = true;
    bool labelExists = false;
    QVector<PointCloudCuboid> cuboids;
    QString cuboidPath;
    bool cuboidExists = false;
};

bool loadPointCloudFrame(const QString& path,
                         PointCloudFrame* frame,
                         QString* error,
                         const QString& labelPath = QString(),
                         const std::function<bool()>& cancelled = {});
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
        SplitInstance,
        CreateCuboid
    };
    void setFrame(PointCloudFrame frame);
    const PointCloudFrame& frame() const { return m_frame; }
    void setLabelPath(const QString& path, bool exists = true) {
        m_frame.labelPath = path;
        m_frame.labelExists = exists;
    }
    bool dirty() const { return m_dirty || cuboidsDirty(); }
    bool labelsDirty() const { return m_dirty; }
    bool cuboidsDirty() const { return m_frame.cuboids != m_cuboidBaseline; }
    int nextCuboidId() const;
    bool setCuboid(const PointCloudCuboid& cuboid, QString* error = nullptr);
    bool deleteCuboid(int id, QString* error = nullptr);
    bool replaceCuboids(const QVector<PointCloudCuboid>& cuboids,
                        QString* error = nullptr);
    void markCuboidsSaved(const QString& path);
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
        bool cuboids = false;
        QVector<PointCloudCuboid> boxesBefore;
        QVector<PointCloudCuboid> boxesAfter;
        QVector<int> indices;
        QVector<quint32> before;
        QVector<quint32> after;
    };
    void apply(const Edit& edit, bool forward);
    bool commitCuboids(const QVector<PointCloudCuboid>& boxes);
    PointCloudFrame m_frame;
    QVector<quint32> m_baseline;
    QVector<PointCloudCuboid> m_cuboidBaseline;
    QVector<Edit> m_undo;
    QVector<Edit> m_redo;
    bool m_dirty = false;
};
