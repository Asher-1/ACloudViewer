// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RelationFilterWidget.h"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSaveFile>
#include <QTableWidget>
#include <QVBoxLayout>

RelationFilterWidget::RelationFilterWidget(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);

    // Filter controls row
    auto* filterRow = new QHBoxLayout;

    filterRow->addWidget(new QLabel(tr("Type:")));
    m_typeFilter = new QComboBox;
    m_typeFilter->addItem(tr("All"), QStringLiteral("all"));
    m_typeFilter->addItem(tr("Semantic"), QStringLiteral("semantic"));
    m_typeFilter->addItem(tr("Spatial"), QStringLiteral("spatial"));
    filterRow->addWidget(m_typeFilter);

    filterRow->addWidget(new QLabel(tr("Min Score:")));
    m_scoreThresholdSpinBox = new QDoubleSpinBox;
    m_scoreThresholdSpinBox->setRange(0.0, 1.0);
    m_scoreThresholdSpinBox->setSingleStep(0.05);
    m_scoreThresholdSpinBox->setValue(0.0);
    m_scoreThresholdSpinBox->setDecimals(2);
    filterRow->addWidget(m_scoreThresholdSpinBox);

    m_searchEdit = new QLineEdit;
    m_searchEdit->setPlaceholderText(tr("Search subject/predicate/object..."));
    filterRow->addWidget(m_searchEdit, 1);

    m_clearBtn = new QPushButton(tr("Clear Filter"));
    filterRow->addWidget(m_clearBtn);

    layout->addLayout(filterRow);

    // Relation table
    setObjectName(QStringLiteral("AnnotationRelationFilter"));
    m_relationTable = new QTableWidget;
    m_relationTable->setObjectName(QStringLiteral("AnnotationRelationTable"));
    m_relationTable->setColumnCount(5);
    m_relationTable->setHorizontalHeaderLabels({tr("Subject"), tr("Predicate"),
                                                tr("Object"), tr("Type"),
                                                tr("Score")});
    m_relationTable->horizontalHeader()->setStretchLastSection(false);
    m_relationTable->horizontalHeader()->setSectionResizeMode(
            0, QHeaderView::Stretch);
    m_relationTable->horizontalHeader()->setSectionResizeMode(
            1, QHeaderView::Stretch);
    m_relationTable->horizontalHeader()->setSectionResizeMode(
            2, QHeaderView::Stretch);
    m_relationTable->horizontalHeader()->setSectionResizeMode(
            3, QHeaderView::ResizeToContents);
    m_relationTable->horizontalHeader()->setSectionResizeMode(
            4, QHeaderView::ResizeToContents);
    m_relationTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_relationTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_relationTable->setAlternatingRowColors(true);
    layout->addWidget(m_relationTable);

    // Status and export row
    auto* bottomRow = new QHBoxLayout;
    m_countLabel = new QLabel(tr("0 relations"));
    bottomRow->addWidget(m_countLabel);
    bottomRow->addStretch();

    m_exportBtn = new QPushButton(tr("Export to CSV"));
    bottomRow->addWidget(m_exportBtn);

    layout->addLayout(bottomRow);

    // Connect signals
    connect(m_typeFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &RelationFilterWidget::onTypeFilterChanged);
    connect(m_scoreThresholdSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &RelationFilterWidget::onScoreThresholdChanged);
    connect(m_searchEdit, &QLineEdit::textChanged, this,
            &RelationFilterWidget::onSearchTextChanged);
    connect(m_clearBtn, &QPushButton::clicked, this,
            &RelationFilterWidget::onClearFilter);
    connect(m_exportBtn, &QPushButton::clicked, this,
            &RelationFilterWidget::onExportTriplets);
    connect(m_relationTable, &QTableWidget::cellClicked, this,
            &RelationFilterWidget::onRowActivated);
}

void RelationFilterWidget::loadRelations(const QJsonArray& relations) {
    m_allRelations = relations;
    updateFilteredView();
}

QJsonArray RelationFilterWidget::visibleRelations() const {
    QJsonArray visible;
    for (const QJsonValue& val : m_allRelations) {
        if (!val.isObject()) continue;
        QJsonObject relation = val.toObject();
        if (passesFilter(relation)) {
            visible.append(relation);
        }
    }
    return visible;
}

void RelationFilterWidget::clear() {
    m_allRelations = QJsonArray();
    m_relationTable->setRowCount(0);
    m_countLabel->setText(tr("0 relations"));
}

void RelationFilterWidget::onTypeFilterChanged(int index) {
    Q_UNUSED(index);
    m_selectedType = m_typeFilter->currentData().toString();
    updateFilteredView();
}

void RelationFilterWidget::onScoreThresholdChanged(double value) {
    m_scoreThresholdValue = value;
    updateFilteredView();
}

void RelationFilterWidget::onSearchTextChanged(const QString& text) {
    m_searchText = text.toLower();
    updateFilteredView();
}

void RelationFilterWidget::onClearFilter() {
    m_typeFilter->setCurrentIndex(0);  // All
    m_scoreThresholdSpinBox->setValue(0.0);
    m_searchEdit->clear();
}

namespace {
QString csvField(QString text) {
    // RFC 4180 quoting; also quote leading spreadsheet-formula characters so
    // an object label cannot turn into a formula when the CSV is opened.
    const bool formula =
            !text.isEmpty() && QStringLiteral("=+-@").contains(text[0]);
    if (formula) text.prepend(QLatin1Char('\''));
    if (text.contains(QLatin1Char(',')) || text.contains(QLatin1Char('"')) ||
        text.contains(QLatin1Char('\n')) || text.contains(QLatin1Char('\r')))
        return QLatin1Char('"') +
               text.replace(QLatin1Char('"'), QStringLiteral("\"\"")) +
               QLatin1Char('"');
    return text;
}
}  // namespace

bool RelationFilterWidget::saveCsv(const QString& path, QString* error) const {
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        if (error) *error = file.errorString();
        return false;
    }
    // Encode explicitly so Qt 5 and Qt 6 write identical UTF-8 bytes.
    QByteArray data =
            QByteArrayLiteral("Subject,Predicate,Object,Type,Score\n");
    for (int row = 0; row < m_relationTable->rowCount(); ++row) {
        QStringList fields;
        for (int col = 0; col < 5; ++col)
            fields << csvField(m_relationTable->item(row, col)->text());
        data += fields.join(QLatin1Char(',')).toUtf8() + '\n';
    }
    if (file.write(data) != data.size() || !file.commit()) {
        if (error) *error = file.errorString();
        return false;
    }
    return true;
}

void RelationFilterWidget::onExportTriplets() {
    if (m_relationTable->rowCount() == 0) {
        QMessageBox::information(this, tr("Export"),
                                 tr("No relations to export."));
        return;
    }
    const QString path =
            QFileDialog::getSaveFileName(this, tr("Export Relation Triplets"),
                                         QString(), tr("CSV Files (*.csv)"));
    if (path.isEmpty()) return;
    QString error;
    if (!saveCsv(path, &error))
        QMessageBox::critical(this, tr("Export Failed"),
                              tr("Failed to write %1: %2").arg(path, error));
}

void RelationFilterWidget::onRowActivated(int row, int column) {
    Q_UNUSED(column);
    const QJsonArray visible = visibleRelations();
    if (row < 0 || row >= visible.size()) return;
    QJsonObject relation = visible.at(row).toObject();
    relation.insert(QStringLiteral("row"), row);
    emit relationSelected(relation);
}

void RelationFilterWidget::updateFilteredView() {
    m_relationTable->setRowCount(0);

    int visibleCount = 0;
    for (const QJsonValue& val : m_allRelations) {
        if (!val.isObject()) continue;
        QJsonObject relation = val.toObject();
        if (passesFilter(relation)) {
            addRelationRow(relation);
            ++visibleCount;
        }
    }

    m_countLabel->setText(tr("%1 / %2 relations")
                                  .arg(visibleCount)
                                  .arg(m_allRelations.size()));
    emit filterChanged(visibleRelations());
}

bool RelationFilterWidget::passesFilter(const QJsonObject& relation) const {
    // Type filter
    if (m_selectedType != QStringLiteral("all")) {
        QString type = relation.value("type").toString().toLower();
        if (type != m_selectedType) return false;
    }

    // Score threshold
    double score = relation.value("score").toDouble();
    if (score < m_scoreThresholdValue) return false;

    // Text search
    if (!m_searchText.isEmpty()) {
        QString subject = relation.value("subject").toString().toLower();
        QString predicate = relation.value("predicate").toString().toLower();
        QString object = relation.value("object").toString().toLower();
        if (!subject.contains(m_searchText) &&
            !predicate.contains(m_searchText) &&
            !object.contains(m_searchText)) {
            return false;
        }
    }

    return true;
}

void RelationFilterWidget::addRelationRow(const QJsonObject& relation) {
    int row = m_relationTable->rowCount();
    m_relationTable->insertRow(row);

    m_relationTable->setItem(
            row, 0, new QTableWidgetItem(relation.value("subject").toString()));
    m_relationTable->setItem(
            row, 1,
            new QTableWidgetItem(relation.value("predicate").toString()));
    m_relationTable->setItem(
            row, 2, new QTableWidgetItem(relation.value("object").toString()));
    m_relationTable->setItem(
            row, 3, new QTableWidgetItem(relation.value("type").toString()));
    m_relationTable->setItem(
            row, 4,
            new QTableWidgetItem(QString::number(
                    relation.value("score").toDouble(), 'f', 3)));
}
