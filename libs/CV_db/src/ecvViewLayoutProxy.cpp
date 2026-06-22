// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewLayoutProxy.h"

#include <QJsonArray>
#include <QJsonObject>
#include <algorithm>

#include "ecvGenericGLDisplay.h"
#include "ecvLayoutUndoCommand.h"
#include "ecvUndoManager.h"
#include "ecvViewManager.h"

void ecvViewLayoutProxy::setUndoManager(ecvUndoManager* mgr) {
    m_undoManager = mgr;
}

ecvViewLayoutProxy::ecvViewLayoutProxy(QObject* parent) : QObject(parent) {
    m_tree.resize(1);
}

ecvViewLayoutProxy::~ecvViewLayoutProxy() = default;

// ============================================================================
// Helpers
// ============================================================================

bool ecvViewLayoutProxy::isValidLocation(int location) const {
    return location >= 0 && location < static_cast<int>(m_tree.size());
}

bool ecvViewLayoutProxy::isLeaf(int location) const {
    return isValidLocation(location) && m_tree[location].direction == NONE;
}

void ecvViewLayoutProxy::ensureSize(int location) {
    if (location >= static_cast<int>(m_tree.size())) {
        m_tree.resize(location + 1);
    }
}

void ecvViewLayoutProxy::shrink() {
    while (!m_tree.empty() && m_tree.back().direction == NONE &&
           m_tree.back().view == nullptr &&
           static_cast<int>(m_tree.size()) > 1) {
        m_tree.pop_back();
    }
}

void ecvViewLayoutProxy::moveSubtree(int dst, int src) {
    if (!isValidLocation(src)) {
        if (isValidLocation(dst)) {
            m_tree[dst] = Cell{};
        }
        return;
    }

    ensureSize(dst);
    m_tree[dst] = m_tree[src];
    m_tree[src] = Cell{};

    if (m_tree[dst].direction != NONE) {
        moveSubtree(firstChild(dst), firstChild(src));
        moveSubtree(secondChild(dst), secondChild(src));
    }
}

void ecvViewLayoutProxy::notifyChanged() {
    m_maximizedCell = -1;
    emit layoutChanged();
}

// ============================================================================
// Split
// ============================================================================

int ecvViewLayoutProxy::split(int location,
                              Direction direction,
                              double fraction) {
    if (direction == NONE) return -1;
    if (!isValidLocation(location)) return -1;
    if (!isLeaf(location)) return -1;

    beginUndoSet(QStringLiteral("Split View"));
    fraction = std::clamp(fraction, 0.0, 1.0);

    int child1 = firstChild(location);
    int child2 = secondChild(location);
    ensureSize(child2);

    m_tree[child1].view = m_tree[location].view;
    m_tree[location].view = nullptr;
    m_tree[location].direction = direction;
    m_tree[location].splitFraction = fraction;

    notifyChanged();
    endUndoSet();
    return child1;
}

// ============================================================================
// View assignment
// ============================================================================

bool ecvViewLayoutProxy::assignView(int location, ecvGenericGLDisplay* view) {
    if (!isValidLocation(location)) return false;
    if (!isLeaf(location)) return false;
    if (m_tree[location].view != nullptr && m_tree[location].view != view)
        return false;

    m_tree[location].view = view;
    notifyChanged();
    return true;
}

int ecvViewLayoutProxy::assignViewToAnyCell(ecvGenericGLDisplay* view,
                                            int hint) {
    int existing = getViewLocation(view);
    if (existing >= 0) return existing;

    int emptyCell = -1;
    if (hint >= 0) emptyCell = getEmptyCell(hint);
    if (emptyCell < 0) emptyCell = getEmptyCell(0);

    if (emptyCell >= 0) {
        assignView(emptyCell, view);
        return emptyCell;
    }

    Direction suggestedDir = HORIZONTAL;
    int splittable = getSplittableCell(0, suggestedDir);
    if (splittable < 0) return -1;

    int child = split(splittable, suggestedDir);
    if (child < 0) return -1;

    int newCell = secondChild(parent(child));
    assignView(newCell, view);
    return newCell;
}

int ecvViewLayoutProxy::removeView(ecvGenericGLDisplay* view) {
    int location = getViewLocation(view);
    if (location < 0) return -1;
    m_tree[location].view = nullptr;
    notifyChanged();
    return location;
}

bool ecvViewLayoutProxy::removeViewAt(int location) {
    if (!isValidLocation(location)) return false;
    if (!isLeaf(location)) return false;
    m_tree[location].view = nullptr;
    notifyChanged();
    return true;
}

// ============================================================================
// Collapse
// ============================================================================

bool ecvViewLayoutProxy::collapse(int location) {
    if (!isLeaf(location)) return false;
    if (m_tree[location].view != nullptr) return false;
    if (location == 0) return false;

    beginUndoSet(QStringLiteral("Close View"));
    int par = parent(location);
    int sibling =
            (location == firstChild(par)) ? secondChild(par) : firstChild(par);

    moveSubtree(par, sibling);
    shrink();
    notifyChanged();
    endUndoSet();
    return true;
}

// ============================================================================
// Swap
// ============================================================================

bool ecvViewLayoutProxy::swapCells(int location1, int location2) {
    if (!isLeaf(location1) || !isLeaf(location2)) return false;

    beginUndoSet(QStringLiteral("Swap Views"));
    std::swap(m_tree[location1].view, m_tree[location2].view);
    notifyChanged();
    endUndoSet();
    return true;
}

// ============================================================================
// Split fraction
// ============================================================================

bool ecvViewLayoutProxy::setSplitFraction(int location, double fraction) {
    if (!isValidLocation(location)) return false;
    if (m_tree[location].direction == NONE) return false;

    m_tree[location].splitFraction = std::clamp(fraction, 0.0, 1.0);
    return true;
}

// ============================================================================
// Equalize
// ============================================================================

void ecvViewLayoutProxy::equalize() {
    beginUndoSet(QStringLiteral("Equalize Views"));
    equalize(NONE);
    endUndoSet();
}

void ecvViewLayoutProxy::equalize(Direction direction) {
    beginUndoSet(QStringLiteral("Equalize Views"));
    equalizeRecursive(0, direction);
    notifyChanged();
    endUndoSet();
}

void ecvViewLayoutProxy::equalizeRecursive(int location, Direction filterDir) {
    if (!isValidLocation(location)) return;
    if (m_tree[location].direction == NONE) return;

    if (filterDir == NONE || m_tree[location].direction == filterDir) {
        m_tree[location].splitFraction = 0.5;
    }

    equalizeRecursive(firstChild(location), filterDir);
    equalizeRecursive(secondChild(location), filterDir);
}

// ============================================================================
// Maximize
// ============================================================================

bool ecvViewLayoutProxy::maximizeCell(int location) {
    if (!isLeaf(location)) return false;
    beginUndoSet(QStringLiteral("Maximize View"));
    m_maximizedCell = location;
    emit layoutChanged();
    endUndoSet();
    return true;
}

void ecvViewLayoutProxy::restoreMaximizedState() {
    if (m_maximizedCell >= 0) {
        beginUndoSet(QStringLiteral("Restore Layout"));
        m_maximizedCell = -1;
        emit layoutChanged();
        endUndoSet();
    }
}

// ============================================================================
// Query
// ============================================================================

bool ecvViewLayoutProxy::isSplitCell(int location) const {
    return isValidLocation(location) && m_tree[location].direction != NONE;
}

ecvViewLayoutProxy::Direction ecvViewLayoutProxy::splitDirection(
        int location) const {
    if (!isValidLocation(location)) return NONE;
    return m_tree[location].direction;
}

double ecvViewLayoutProxy::splitFraction(int location) const {
    if (!isValidLocation(location)) return 0.5;
    return m_tree[location].splitFraction;
}

ecvGenericGLDisplay* ecvViewLayoutProxy::getView(int location) const {
    if (!isValidLocation(location)) return nullptr;
    return m_tree[location].view;
}

int ecvViewLayoutProxy::getViewLocation(ecvGenericGLDisplay* view) const {
    if (!view) return -1;
    for (int i = 0; i < static_cast<int>(m_tree.size()); ++i) {
        if (m_tree[i].view == view) return i;
    }
    return -1;
}

std::vector<ecvGenericGLDisplay*> ecvViewLayoutProxy::getViews() const {
    std::vector<ecvGenericGLDisplay*> result;
    for (const auto& cell : m_tree) {
        if (cell.view) result.push_back(cell.view);
    }
    return result;
}

int ecvViewLayoutProxy::viewCount() const {
    int count = 0;
    for (const auto& cell : m_tree) {
        if (cell.view) ++count;
    }
    return count;
}

int ecvViewLayoutProxy::getEmptyCell(int root) const {
    if (!isValidLocation(root)) return -1;
    if (isLeaf(root)) {
        return m_tree[root].view == nullptr ? root : -1;
    }
    int left = getEmptyCell(firstChild(root));
    if (left >= 0) return left;
    return getEmptyCell(secondChild(root));
}

int ecvViewLayoutProxy::getSplittableCell(int root,
                                          Direction& suggestedDir) const {
    if (!isValidLocation(root)) return -1;
    if (isLeaf(root)) {
        suggestedDir = HORIZONTAL;
        return root;
    }
    int left = getSplittableCell(firstChild(root), suggestedDir);
    if (left >= 0) return left;
    return getSplittableCell(secondChild(root), suggestedDir);
}

// ============================================================================
// Metadata
// ============================================================================

void ecvViewLayoutProxy::setName(const QString& name) {
    if (m_name != name) {
        m_name = name;
        emit nameChanged(m_name);
    }
}

// ============================================================================
// Serialization
// ============================================================================

QJsonObject ecvViewLayoutProxy::saveState() const {
    QJsonArray cells;
    for (int i = 0; i < static_cast<int>(m_tree.size()); ++i) {
        const auto& cell = m_tree[i];
        QJsonObject obj;
        obj["index"] = i;
        obj["direction"] = static_cast<int>(cell.direction);
        obj["fraction"] = cell.splitFraction;
        obj["view_id"] = cell.view ? cell.view->getUniqueID() : -1;
        if (cell.direction == NONE && cell.view) {
            const QJsonObject camJson = cell.view->saveLayoutCameraState();
            if (!camJson.isEmpty()) {
                obj[QStringLiteral("camera")] = camJson;
            }
        }
        cells.append(obj);
    }

    QJsonObject state;
    state["name"] = m_name;
    state["cells"] = cells;
    state["maximized_cell"] = m_maximizedCell;
    return state;
}

bool ecvViewLayoutProxy::loadState(const QJsonObject& state) {
    m_name = state["name"].toString();
    m_maximizedCell = state["maximized_cell"].toInt(-1);

    QJsonArray cells = state["cells"].toArray();
    m_tree.clear();
    m_tree.resize(cells.size());

    for (const auto& val : cells) {
        QJsonObject obj = val.toObject();
        int idx = obj["index"].toInt();
        if (idx >= 0 && idx < static_cast<int>(m_tree.size())) {
            m_tree[idx].direction =
                    static_cast<Direction>(obj["direction"].toInt());
            m_tree[idx].splitFraction = obj["fraction"].toDouble(0.5);

            int viewId = obj["view_id"].toInt(-1);
            if (viewId >= 0) {
                auto* view = ecvViewManager::instance().findView(viewId);
                if (view) {
                    m_tree[idx].view = view;
                    const QJsonObject camObj =
                            obj.value(QStringLiteral("camera")).toObject();
                    if (!camObj.isEmpty()) {
                        view->loadLayoutCameraState(camObj);
                    }
                }
            }
        }
    }

    emit layoutChanged();
    return true;
}

void ecvViewLayoutProxy::reset() {
    m_tree.clear();
    m_tree.resize(1);
    m_maximizedCell = -1;
    m_undoNesting = 0;
    m_pendingBeforeState = QJsonObject();
    m_pendingLabel.clear();
    emit layoutChanged();
}

// ============================================================================
// Undo grouping (global stack via ecvUndoManager)
// ============================================================================

void ecvViewLayoutProxy::beginUndoSet(const QString& label) {
    if (m_undoNesting == 0) {
        m_pendingBeforeState = saveState();
        m_pendingLabel = label;
    }
    ++m_undoNesting;
}

void ecvViewLayoutProxy::endUndoSet() {
    if (m_undoNesting <= 0) return;
    --m_undoNesting;
    if (m_undoNesting == 0 && m_undoManager) {
        QJsonObject afterState = saveState();
        if (m_pendingBeforeState != afterState) {
            m_undoManager->push(new ecvLayoutUndoCommand(
                    this, m_pendingBeforeState, afterState, m_pendingLabel));
        }
        m_pendingBeforeState = QJsonObject();
        m_pendingLabel.clear();
    }
}
