// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QJsonObject>
#include <QObject>
#include <QString>
#include <vector>

#include "CV_db.h"

class ecvGenericGLDisplay;

/// KD-tree–based view layout model.
///
/// Directly mirrors ParaView's vtkSMViewLayoutProxy:
///   - Binary tree stored in a heap-indexed std::vector<Cell>
///   - Each cell is either a Split (HORIZONTAL/VERTICAL + fraction) or a Leaf
///   - Leaf cells optionally hold a view pointer
///   - Layout changes fire layoutChanged() so UI widgets can rebuild
///
/// Index arithmetic (same as ParaView):
///   firstChild(i)  = 2*i + 1
///   secondChild(i) = 2*i + 2
///   parent(i)      = (i - 1) / 2   (i > 0)
class CV_DB_LIB_API ecvViewLayoutProxy : public QObject {
    Q_OBJECT

public:
    enum Direction { NONE = 0, VERTICAL, HORIZONTAL };

    explicit ecvViewLayoutProxy(QObject* parent = nullptr);
    ~ecvViewLayoutProxy() override;

    // ================================================================
    // KD-tree operations (ParaView-compatible API)
    // ================================================================

    /// Split a leaf cell. Returns the first child index, or -1 on failure.
    /// The existing view (if any) moves to firstChild; secondChild is empty.
    int split(int location, Direction direction, double fraction = 0.5);

    int splitVertical(int location, double fraction = 0.5) {
        return split(location, VERTICAL, fraction);
    }
    int splitHorizontal(int location, double fraction = 0.5) {
        return split(location, HORIZONTAL, fraction);
    }

    /// Assign a view to a leaf cell. Returns true on success.
    bool assignView(int location, ecvGenericGLDisplay* view);

    /// Find an empty leaf (or split to create one) and assign the view.
    /// hint is the preferred location; -1 means "anywhere".
    int assignViewToAnyCell(ecvGenericGLDisplay* view, int hint = -1);

    /// Remove a view from its cell. Returns the cell index, or -1.
    int removeView(ecvGenericGLDisplay* view);

    /// Remove the view at a specific cell. Returns true on success.
    bool removeViewAt(int location);

    /// Collapse an empty leaf: promote its sibling to the parent.
    /// Returns true on success.
    bool collapse(int location);

    /// Swap views between two leaf cells.
    bool swapCells(int location1, int location2);

    /// Set the split fraction at a split cell.
    bool setSplitFraction(int location, double fraction);

    /// Equalize all splits (50/50).
    void equalize();

    /// Equalize splits along a specific direction only.
    void equalize(Direction direction);

    /// Maximize a single cell (hide all others). Pass -1 to restore.
    bool maximizeCell(int location);
    void restoreMaximizedState();
    int maximizedCell() const { return m_maximizedCell; }

    // ================================================================
    // Query
    // ================================================================

    bool isSplitCell(int location) const;
    Direction splitDirection(int location) const;
    double splitFraction(int location) const;

    ecvGenericGLDisplay* getView(int location) const;
    int getViewLocation(ecvGenericGLDisplay* view) const;
    bool containsView(ecvGenericGLDisplay* view) const {
        return getViewLocation(view) != -1;
    }

    std::vector<ecvGenericGLDisplay*> getViews() const;
    int viewCount() const;

    /// Get the first empty leaf under root. Returns -1 if none.
    int getEmptyCell(int root = 0) const;

    /// Get a splittable cell and suggested direction.
    int getSplittableCell(int root, Direction& suggestedDir) const;

    // ================================================================
    // Index arithmetic (static, same as ParaView)
    // ================================================================

    static int firstChild(int location) { return 2 * location + 1; }
    static int secondChild(int location) { return 2 * location + 2; }
    static int parent(int location) {
        return location > 0 ? (location - 1) / 2 : -1;
    }

    // ================================================================
    // Metadata
    // ================================================================

    QString name() const { return m_name; }
    void setName(const QString& name);

    // ================================================================
    // Serialization
    // ================================================================

    QJsonObject saveState() const;
    bool loadState(const QJsonObject& state);

    /// Reset to a single empty root cell.
    void reset();

signals:
    /// Emitted whenever the layout tree structure or view assignments change.
    /// UI widgets should call reload() in response.
    void layoutChanged();

    void nameChanged(const QString& name);

private:
    struct Cell {
        Direction direction = NONE;
        double splitFraction = 0.5;
        ecvGenericGLDisplay* view = nullptr;
    };

    bool isValidLocation(int location) const;
    bool isLeaf(int location) const;
    void ensureSize(int location);
    void shrink();
    void moveSubtree(int dst, int src);
    void notifyChanged();
    void equalizeRecursive(int location, Direction filterDir);

    std::vector<Cell> m_tree;
    int m_maximizedCell = -1;
    QString m_name;
};
