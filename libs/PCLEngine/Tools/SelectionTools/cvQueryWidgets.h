// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QWidget>
#include <QMap>
#include <QString>
#include <QRegularExpression>

/**
 * @brief cvQueryValueWidget - Dynamic value input widget
 * 
 * Based on ParaView's pqValueWidget, this widget dynamically creates
 * different UI components based on the ValueType. It supports:
 * - NO_VALUE: No input (for operators like "is min", "is max")
 * - SINGLE_VALUE: Single text input
 * - COMMA_SEPARATED_VALUES: Single text input for comma-separated values
 * - RANGE_PAIR: Two inputs for min/max range
 * - LOCATION: Three inputs for X/Y/Z coordinates
 * - LOCATION_WITH_TOLERANCE: X/Y/Z coordinates + tolerance value
 */
class cvQueryValueWidget : public QWidget {
    Q_OBJECT

public:
    enum ValueType {
        NO_VALUE,
        SINGLE_VALUE,
        COMMA_SEPARATED_VALUES,
        RANGE_PAIR,
        LOCATION_WITH_TOLERANCE,  // Must be before LOCATION (ParaView order)
        LOCATION,
    };

    explicit cvQueryValueWidget(QWidget* parent = nullptr);
    ~cvQueryValueWidget() override = default;

    /**
     * Set the type of value widget. This rebuilds the UI with appropriate
     * input components.
     */
    void setType(ValueType type);

    /**
     * Get current values as a map (key -> value).
     * Keys are: "value", "value_min", "value_max", "value_x", "value_y", "value_z", "value_tolerance"
     */
    QMap<QString, QString> values() const;

    /**
     * Set values from a map (used when parsing existing expressions)
     */
    void setValues(const QMap<QString, QString>& values);

    /**
     * Clear all input fields
     */
    void clear();

signals:
    void valueChanged();

private:
    void rebuildUI();

    ValueType m_type;
    QMap<QString, QLineEdit*> m_lineEdits;
};

/**
 * @brief cvQueryConditionWidget - Single query condition widget
 * 
 * Based on ParaView's pqQueryWidget, represents one query condition row:
 * [Term Combo] [Operator Combo] [Value Widget]
 * 
 * The widget automatically updates the operator list and value widget
 * based on the selected term type.
 */
class cvQueryConditionWidget : public QWidget {
    Q_OBJECT

public:
    enum TermType {
        ARRAY,                      // Regular array field (ID, NormalX, etc.)
        POINT_NEAREST_TO,          // Point nearest to location
        CELL_CONTAINING_POINT,     // Cell containing a point
    };

    explicit cvQueryConditionWidget(QWidget* parent = nullptr);
    ~cvQueryConditionWidget() override = default;

    /**
     * Update the widget with available terms from data
     */
    void updateTerms(const QStringList& arrayNames, 
                     const QMap<QString, int>& arrayComponents,
                     bool isPointData);

    /**
     * Get the generated query expression (e.g., "NormalX >= 0.5")
     */
    QString expression() const;

    /**
     * Set expression and parse it to update UI
     */
    void setExpression(const QString& expr);

    /**
     * Clear all selections
     */
    void clear();

signals:
    void conditionChanged();

private slots:
    void onTermChanged(int index);
    void onOperatorChanged(int index);

private:
    void populateOperators(TermType termType);
    void updateValueWidget();
    
    QString currentTerm() const;
    TermType currentTermType() const;
    void setCurrentTerm(const QString& term);

    // Helper to add operator with its metadata
    void addOperator(const QString& text, 
                     cvQueryValueWidget::ValueType valueType,
                     const QString& expressionTemplate);

    // Helper to add term with its metadata
    void addTerm(const QString& text, TermType type, const QString& internalName);

    QComboBox* m_termCombo;
    QComboBox* m_operatorCombo;
    cvQueryValueWidget* m_valueWidget;

    // Role constants for combo box item data
    static constexpr int TermTypeRole = Qt::UserRole;
    static constexpr int NameRole = Qt::UserRole + 1;
    static constexpr int ValueTypeRole = Qt::UserRole;
    static constexpr int ExprTemplateRole = Qt::UserRole + 1;
    static constexpr int ExprRegExRole = Qt::UserRole + 2;
};

/**
 * @brief Helper functions for expression formatting and parsing
 */
namespace QueryExpressionUtils {
    /**
     * Format expression template with values
     * e.g., fmt("{term} >= {value}", {{"term", "NormalX"}, {"value", "0.5"}}) 
     *       -> "NormalX >= 0.5"
     */
    QString formatExpression(const QString& templateStr, const QMap<QString, QString>& values);

    /**
     * Create regex pattern from expression template for parsing
     */
    QRegularExpression createRegex(const QString& templateStr);

    /**
     * Split compound expression by AND operator (&)
     * e.g., "(NormalX >= 0.5) & (NormalY <= 1.0)" -> ["NormalX >= 0.5", "NormalY <= 1.0"]
     */
    QStringList splitByAnd(const QString& expression);
}


