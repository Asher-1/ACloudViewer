// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef RELATION_FILTER_WIDGET_H
#define RELATION_FILTER_WIDGET_H

#include <QJsonArray>
#include <QJsonObject>
#include <QWidget>

class QTableWidget;
class QTableWidgetItem;
class QComboBox;
class QDoubleSpinBox;
class QPushButton;
class QLabel;
class QLineEdit;

/**
 * @brief Widget for filtering and displaying RelateAnything triplets
 *
 * Displays relationship triplets (subject, predicate, object) with scores
 * Allows filtering by:
 * - Relation type (semantic vs spatial)
 * - Confidence score threshold
 * - Subject/Object/Predicate text search
 */
class RelationFilterWidget : public QWidget {
    Q_OBJECT

public:
    explicit RelationFilterWidget(QWidget* parent = nullptr);

    // Load relation triplets from RelateAnything inference result
    void loadRelations(const QJsonArray& relations);

    // Get currently visible (filtered) relations
    QJsonArray visibleRelations() const;

    // Clear all relations
    void clear();

    // Write the currently visible rows as CSV (atomic; UTF-8). Returns false
    // and fills *error when the file cannot be fully written.
    bool saveCsv(const QString& path, QString* error = nullptr) const;

signals:
    // Emitted when filter changes
    void filterChanged(const QJsonArray& visibleRelations);

    // Emitted when a row is clicked; "row" is the index in the visible list,
    // which is also the index of the edge in the canvas overlay.
    void relationSelected(const QJsonObject& relation);

private slots:
    void onTypeFilterChanged(int index);
    void onScoreThresholdChanged(double value);
    void onSearchTextChanged(const QString& text);
    void onClearFilter();
    void onExportTriplets();
    void onRowActivated(int row, int column);

private:
    void updateFilteredView();
    bool passesFilter(const QJsonObject& relation) const;
    void addRelationRow(const QJsonObject& relation);

    QTableWidget* m_relationTable = nullptr;
    QComboBox* m_typeFilter = nullptr;  // All / Semantic / Spatial
    QDoubleSpinBox* m_scoreThresholdSpinBox = nullptr;
    QLineEdit* m_searchEdit = nullptr;
    QPushButton* m_clearBtn = nullptr;
    QPushButton* m_exportBtn = nullptr;
    QLabel* m_countLabel = nullptr;

    QJsonArray m_allRelations;  // Complete relation list
    QString m_searchText;
    QString m_selectedType = QStringLiteral("all");  // all/semantic/spatial
    double m_scoreThresholdValue = 0.0;
};

#endif  // RELATION_FILTER_WIDGET_H
