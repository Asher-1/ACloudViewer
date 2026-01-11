// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvQueryWidgets.h"

#include <CVLog.h>

//=============================================================================
// cvQueryValueWidget Implementation
//=============================================================================

cvQueryValueWidget::cvQueryValueWidget(QWidget* parent)
    : QWidget(parent), m_type(NO_VALUE) {
    rebuildUI();
}

void cvQueryValueWidget::setType(ValueType type) {
    if (m_type != type) {
        m_type = type;
        rebuildUI();
    }
}

QMap<QString, QString> cvQueryValueWidget::values() const {
    QMap<QString, QString> result;
    for (auto it = m_lineEdits.constBegin(); it != m_lineEdits.constEnd();
         ++it) {
        result[it.key()] = it.value()->text();
    }
    return result;
}

void cvQueryValueWidget::setValues(const QMap<QString, QString>& values) {
    for (auto it = values.constBegin(); it != values.constEnd(); ++it) {
        if (m_lineEdits.contains(it.key())) {
            m_lineEdits[it.key()]->setText(it.value());
        }
    }
}

void cvQueryValueWidget::clear() {
    for (auto* edit : m_lineEdits) {
        edit->clear();
    }
}

void cvQueryValueWidget::rebuildUI() {
    // Delete existing widgets
    qDeleteAll(findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly));
    m_lineEdits.clear();
    delete layout();

    switch (m_type) {
        case NO_VALUE:
            // No UI needed
            break;

        case SINGLE_VALUE:
        case COMMA_SEPARATED_VALUES: {
            auto* vbox = new QVBoxLayout(this);
            vbox->setContentsMargins(0, 0, 0, 0);
            vbox->setSpacing(3);

            auto* edit = new QLineEdit(this);
            edit->setObjectName("value");
            if (m_type == SINGLE_VALUE) {
                edit->setPlaceholderText(tr("value"));
            } else {
                edit->setPlaceholderText(tr("comma separated values"));
            }
            vbox->addWidget(edit);
            m_lineEdits["value"] = edit;

            connect(edit, &QLineEdit::textChanged, this,
                    &cvQueryValueWidget::valueChanged);
            break;
        }

        case RANGE_PAIR: {
            auto* hbox = new QHBoxLayout(this);
            hbox->setContentsMargins(0, 0, 0, 0);
            hbox->setSpacing(3);

            auto* editMin = new QLineEdit(this);
            editMin->setObjectName("value_min");
            editMin->setPlaceholderText(tr("minimum"));
            m_lineEdits["value_min"] = editMin;

            auto* label = new QLabel(tr("and"), this);
            label->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

            auto* editMax = new QLineEdit(this);
            editMax->setObjectName("value_max");
            editMax->setPlaceholderText(tr("maximum"));
            m_lineEdits["value_max"] = editMax;

            hbox->addWidget(editMin, 1);
            hbox->addWidget(label, 0);
            hbox->addWidget(editMax, 1);

            connect(editMin, &QLineEdit::textChanged, this,
                    &cvQueryValueWidget::valueChanged);
            connect(editMax, &QLineEdit::textChanged, this,
                    &cvQueryValueWidget::valueChanged);
            break;
        }

        case LOCATION:
        case LOCATION_WITH_TOLERANCE: {
            auto* grid = new QGridLayout(this);
            grid->setContentsMargins(0, 0, 0, 0);
            grid->setVerticalSpacing(3);
            grid->setHorizontalSpacing(3);

            auto* editX = new QLineEdit(this);
            editX->setObjectName("value_x");
            editX->setPlaceholderText(tr("X"));
            m_lineEdits["value_x"] = editX;

            auto* editY = new QLineEdit(this);
            editY->setObjectName("value_y");
            editY->setPlaceholderText(tr("Y"));
            m_lineEdits["value_y"] = editY;

            auto* editZ = new QLineEdit(this);
            editZ->setObjectName("value_z");
            editZ->setPlaceholderText(tr("Z"));
            m_lineEdits["value_z"] = editZ;

            grid->addWidget(editX, 0, 0);
            grid->addWidget(editY, 0, 1);
            grid->addWidget(editZ, 0, 2);

            connect(editX, &QLineEdit::textChanged, this,
                    &cvQueryValueWidget::valueChanged);
            connect(editY, &QLineEdit::textChanged, this,
                    &cvQueryValueWidget::valueChanged);
            connect(editZ, &QLineEdit::textChanged, this,
                    &cvQueryValueWidget::valueChanged);

            if (m_type == LOCATION_WITH_TOLERANCE) {
                auto* editTolerance = new QLineEdit(this);
                editTolerance->setObjectName("value_tolerance");
                editTolerance->setPlaceholderText(tr("within epsilon"));
                m_lineEdits["value_tolerance"] = editTolerance;
                grid->addWidget(editTolerance, 1, 0, 1, 3);

                connect(editTolerance, &QLineEdit::textChanged, this,
                        &cvQueryValueWidget::valueChanged);
            }
            break;
        }
    }
}

//=============================================================================
// cvQueryConditionWidget Implementation
//=============================================================================

cvQueryConditionWidget::cvQueryConditionWidget(QWidget* parent)
    : QWidget(parent),
      m_termCombo(new QComboBox(this)),
      m_operatorCombo(new QComboBox(this)),
      m_valueWidget(new cvQueryValueWidget(this)) {
    m_termCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_operatorCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);

    auto* hbox = new QHBoxLayout(this);
    hbox->setContentsMargins(0, 0, 0, 0);
    hbox->setSpacing(3);
    hbox->addWidget(m_termCombo, 0, Qt::AlignTop);
    hbox->addWidget(m_operatorCombo, 0, Qt::AlignTop);
    hbox->addWidget(m_valueWidget, 1, Qt::AlignTop);

    connect(m_termCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &cvQueryConditionWidget::onTermChanged);
    connect(m_operatorCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &cvQueryConditionWidget::onOperatorChanged);
    connect(m_valueWidget, &cvQueryValueWidget::valueChanged, this,
            &cvQueryConditionWidget::conditionChanged);
}

void cvQueryConditionWidget::updateTerms(
        const QStringList& arrayNames,
        const QMap<QString, int>& arrayComponents,
        bool isPointData) {
    const QSignalBlocker blocker(m_termCombo);
    m_termCombo->clear();

    // Add ID first (ParaView style)
    addTerm(tr("ID"), ARRAY, "id");

    // Add arrays
    for (const QString& arrayName : arrayNames) {
        int numComponents = arrayComponents.value(arrayName, 1);

        if (numComponents == 1) {
            // Single component array
            addTerm(arrayName, ARRAY, arrayName);
        } else if (numComponents > 1) {
            // Multi-component: add magnitude first, then components (lowercase
            // 'magnitude' to match ParaView)
            addTerm(QString("%1 (magnitude)").arg(arrayName), ARRAY,
                    QString("mag(%1)").arg(arrayName));

            // Component names
            QStringList compNames;
            if (numComponents == 3) {
                compNames << "X" << "Y" << "Z";
            } else if (numComponents == 4) {
                compNames << "X" << "Y" << "Z" << "W";
            } else if (numComponents == 2) {
                compNames << "X" << "Y";
            } else {
                for (int i = 0; i < numComponents; ++i) {
                    compNames << QString::number(i);
                }
            }

            for (int i = 0; i < numComponents && i < compNames.size(); ++i) {
                addTerm(QString("%1 (%2)").arg(arrayName, compNames[i]), ARRAY,
                        QString("%1[:,%2]").arg(arrayName).arg(i));
            }
        }
    }

    // Add Point/Cell location query
    if (isPointData) {
        addTerm(tr("Point"), POINT_NEAREST_TO, "inputs");
    } else {
        addTerm(tr("Cell"), CELL_CONTAINING_POINT, "inputs");
    }

    // Initialize operators for first term
    if (m_termCombo->count() > 0) {
        m_termCombo->setCurrentIndex(0);
        onTermChanged(0);
    }
}

QString cvQueryConditionWidget::expression() const {
    QString expr = m_operatorCombo->currentData(ExprTemplateRole).toString();
    if (expr.isEmpty()) {
        return QString();
    }

    QMap<QString, QString> values = m_valueWidget->values();
    values["term"] = currentTerm();

    return QueryExpressionUtils::formatExpression(expr, values);
}

void cvQueryConditionWidget::setExpression(const QString& expr) {
    // Try to match against all operators
    for (int termType = CELL_CONTAINING_POINT; termType >= 0; --termType) {
        populateOperators(static_cast<TermType>(termType));

        for (int i = 0; i < m_operatorCombo->count(); ++i) {
            QRegularExpression regex =
                    m_operatorCombo->itemData(i, ExprRegExRole)
                            .toRegularExpression();
            QRegularExpressionMatch match = regex.match(expr);

            if (match.hasMatch()) {
                // Found a match!
                setCurrentTerm(match.captured("term"));
                m_operatorCombo->setCurrentIndex(i);

                // Extract values
                QMap<QString, QString> values;
                for (const QString& key :
                     {"value", "value_min", "value_max", "value_x", "value_y",
                      "value_z", "value_tolerance"}) {
                    QString capturedValue = match.captured(key);
                    if (!capturedValue.isNull() && !capturedValue.isEmpty()) {
                        values[key] = capturedValue;
                    }
                }
                m_valueWidget->setValues(values);
                return;
            }
        }
    }

    // Couldn't parse - reset to defaults
    if (m_termCombo->count() > 0) {
        m_termCombo->setCurrentIndex(0);
    }
    if (m_operatorCombo->count() > 0) {
        m_operatorCombo->setCurrentIndex(0);
    }
    m_valueWidget->clear();
}

void cvQueryConditionWidget::clear() {
    if (m_termCombo->count() > 0) {
        m_termCombo->setCurrentIndex(0);
    }
    if (m_operatorCombo->count() > 0) {
        m_operatorCombo->setCurrentIndex(0);
    }
    m_valueWidget->clear();
}

void cvQueryConditionWidget::onTermChanged(int index) {
    Q_UNUSED(index);
    populateOperators(currentTermType());
    updateValueWidget();
    emit conditionChanged();
}

void cvQueryConditionWidget::onOperatorChanged(int index) {
    Q_UNUSED(index);
    updateValueWidget();
    emit conditionChanged();
}

void cvQueryConditionWidget::populateOperators(TermType termType) {
    const QSignalBlocker blocker(m_operatorCombo);
    m_operatorCombo->clear();

    using VT = cvQueryValueWidget::ValueType;

    switch (termType) {
        case ARRAY:
            addOperator("is", VT::SINGLE_VALUE, "{term} == {value}");
            addOperator("is in range", VT::RANGE_PAIR,
                        "({term} > {value_min}) & ({term} < {value_max})");
            addOperator("is one of", VT::COMMA_SEPARATED_VALUES,
                        "isin({term}, [{value}])");
            addOperator("is >=", VT::SINGLE_VALUE, "{term} >= {value}");
            addOperator("is <=", VT::SINGLE_VALUE, "{term} <= {value}");
            addOperator("is min", VT::NO_VALUE, "{term} == min({term})");
            addOperator("is max", VT::NO_VALUE, "{term} == max({term})");
            addOperator("is min per block", VT::NO_VALUE,
                        "{term} == min_per_block({term})");
            addOperator("is max per block", VT::NO_VALUE,
                        "{term} == max_per_block({term})");
            addOperator("is NaN", VT::NO_VALUE, "isnan({term})");
            addOperator("is <= mean", VT::NO_VALUE, "{term} <= mean({term})");
            addOperator("is >= mean", VT::NO_VALUE, "{term} >= mean({term})");
            addOperator("is mean", VT::SINGLE_VALUE,
                        "abs({term} - mean({term})) <= {value}");
            break;

        case POINT_NEAREST_TO:
            addOperator("nearest to", VT::LOCATION_WITH_TOLERANCE,
                        "pointIsNear([({value_x}, {value_y}, {value_z}),], "
                        "{value_tolerance}, {term})");
            break;

        case CELL_CONTAINING_POINT:
            addOperator("containing", VT::LOCATION,
                        "cellContainsPoint({term}, [({value_x}, {value_y}, "
                        "{value_z}),])");
            break;
    }
}

void cvQueryConditionWidget::updateValueWidget() {
    auto valueType = static_cast<cvQueryValueWidget::ValueType>(
            m_operatorCombo->currentData(ValueTypeRole).toInt());
    m_valueWidget->setType(valueType);
}

QString cvQueryConditionWidget::currentTerm() const {
    return m_termCombo->currentData(NameRole).toString();
}

cvQueryConditionWidget::TermType cvQueryConditionWidget::currentTermType()
        const {
    return static_cast<TermType>(
            m_termCombo->currentData(TermTypeRole).toInt());
}

void cvQueryConditionWidget::setCurrentTerm(const QString& term) {
    for (int i = 0; i < m_termCombo->count(); ++i) {
        if (m_termCombo->itemData(i, NameRole).toString() == term) {
            m_termCombo->setCurrentIndex(i);
            return;
        }
    }

    // Term not found - add as unknown
    m_termCombo->insertItem(0, term + "(?)");
    m_termCombo->setItemData(0, ARRAY, TermTypeRole);
    m_termCombo->setItemData(0, term, NameRole);
    m_termCombo->setCurrentIndex(0);
}

void cvQueryConditionWidget::addOperator(
        const QString& text,
        cvQueryValueWidget::ValueType valueType,
        const QString& expressionTemplate) {
    int index = m_operatorCombo->count();
    m_operatorCombo->addItem(text);
    m_operatorCombo->setItemData(index, valueType, ValueTypeRole);
    m_operatorCombo->setItemData(index, expressionTemplate, ExprTemplateRole);

    // Create regex for parsing
    QRegularExpression regex =
            QueryExpressionUtils::createRegex(expressionTemplate);
    m_operatorCombo->setItemData(index, regex, ExprRegExRole);
}

void cvQueryConditionWidget::addTerm(const QString& text,
                                     TermType type,
                                     const QString& internalName) {
    int index = m_termCombo->count();
    m_termCombo->addItem(text);
    m_termCombo->setItemData(index, type, TermTypeRole);
    m_termCombo->setItemData(index, internalName, NameRole);
}

//=============================================================================
// QueryExpressionUtils Implementation
//=============================================================================

namespace QueryExpressionUtils {

QString formatExpression(const QString& templateStr,
                         const QMap<QString, QString>& values) {
    QString result = templateStr;

    // Check for empty values (except term which should always exist)
    for (auto it = values.constBegin(); it != values.constEnd(); ++it) {
        if (it.key() != "term" && it.value().isEmpty()) {
            return QString();  // Invalid expression
        }
    }

    // Replace all placeholders
    for (auto it = values.constBegin(); it != values.constEnd(); ++it) {
        QString placeholder = QString("{%1}").arg(it.key());
        result.replace(placeholder, it.value());
    }

    return result;
}

QRegularExpression createRegex(const QString& templateStr) {
    QString pattern = templateStr;

    // Escape special regex characters in the template
    pattern.replace("(", "\\(").replace(")", "\\)");
    pattern.replace("[", "\\[").replace("]", "\\]");
    pattern.replace("+", "\\+").replace("*", "\\*");
    pattern.replace(".", "\\.");

    // Define capture patterns for different types (matching ParaView exactly)
    QMap<QString, QString> capturePatterns;
    // Note: the term can include "accl", "mag(accl)", "accl[:,0]", etc.
    // Hence the pattern is not simply "\w+".
    capturePatterns["term"] = R"==(\w+|\w+\(\w+\)|\w+\[:,\d+\])==";
    // Value pattern allows: alphanumeric, dot, underscore, comma, space, hyphen
    capturePatterns["value"] = R"([a-zA-Z0-9\._,\s\-]+)";
    // Numeric patterns allow: alphanumeric, underscore, dot, hyphen
    capturePatterns["value_min"] = R"([a-zA-Z0-9_.\-]+)";
    capturePatterns["value_max"] = R"([a-zA-Z0-9_.\-]+)";
    capturePatterns["value_x"] = R"([a-zA-Z0-9_.\-]+)";
    capturePatterns["value_y"] = R"([a-zA-Z0-9_.\-]+)";
    capturePatterns["value_z"] = R"([a-zA-Z0-9_.\-]+)";
    capturePatterns["value_tolerance"] = R"([a-zA-Z0-9_.\-]+)";

    // Replace placeholders with named capture groups
    // First occurrence gets the full capture group, subsequent ones are
    // back-references
    QMap<QString, bool> usedKeys;
    for (auto it = capturePatterns.constBegin();
         it != capturePatterns.constEnd(); ++it) {
        QString placeholder = QString("{%1}").arg(it.key());
        int firstOccurrence = pattern.indexOf(placeholder);

        if (firstOccurrence != -1) {
            // Replace first occurrence with named capture group
            pattern.replace(firstOccurrence, placeholder.length(),
                            QString("(?<%1>%2)").arg(it.key(), it.value()));
            usedKeys[it.key()] = true;

            // Replace remaining occurrences with back-reference
            pattern.replace(placeholder, QString("\\g{%1}").arg(it.key()));
        }
    }

    QRegularExpression regex("^" + pattern + "$");
    if (!regex.isValid()) {
        CVLog::Warning(QString("[QueryExpressionUtils] Invalid regex: %1")
                               .arg(regex.errorString()));
    }

    return regex;
}

QStringList splitByAnd(const QString& expression) {
    QStringList result;
    int parentCount = 0;
    int start = 0;

    for (int pos = 0; pos < expression.size(); ++pos) {
        if (expression[pos] == '(') {
            ++parentCount;
        } else if (expression[pos] == ')') {
            --parentCount;
        } else if (parentCount == 0 && expression[pos] == '&') {
            QString term = expression.mid(start, pos - start).trimmed();
            if (!term.isEmpty()) {
                // Remove outer parentheses if present
                if (term.startsWith('(') && term.endsWith(')')) {
                    term = term.mid(1, term.length() - 2).trimmed();
                }
                result.append(term);
            }
            start = pos + 1;
        }
    }

    // Add last term
    if (start < expression.size()) {
        QString term = expression.mid(start).trimmed();
        if (!term.isEmpty()) {
            if (term.startsWith('(') && term.endsWith(')')) {
                term = term.mid(1, term.length() - 2).trimmed();
            }
            result.append(term);
        }
    }

    return result;
}

}  // namespace QueryExpressionUtils
