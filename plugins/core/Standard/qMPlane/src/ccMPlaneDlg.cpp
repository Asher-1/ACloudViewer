// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccMPlaneDlg.h"

// Qt
#include <qheaderview.h>
#include <qpushbutton.h>
#include <qscrollbar.h>
#include <qsizepolicy.h>

// Sizing of Qt tables inspired by:
// https://stackoverflow.com/questions/8766633/how-to-determine-the-correct-size-of-a-qtablewidget

ccMPlaneDlg::ccMPlaneDlg(QWidget *parent /*=0*/)
    : ccOverlayDialog(parent), Ui::MPlaneDialog() {
    setupUi(this);

    QPalette p;
    p.setColor(backgroundRole(), QColor(240, 240, 240, 200));
    setPalette(p);
    setAutoFillBackground(true);

    // Create the TabWidget + Tabs
    m_tabWidget = new QTabWidget();
    layout()->addWidget(m_tabWidget);

    createPlaneFittingTab();
    createMeasurementTab();

    m_tabWidget->addTab(m_tabPlaneFitting, "Point selection");
    m_tabWidget->addTab(m_tabMeasurement, "Measurement");

    // Create the close button
    QPushButton *buttonClose = new QPushButton("Close");
    layout()->addWidget(buttonClose);

    // Qt Signal <-> Slot connections
    connect(buttonClose, &QPushButton::clicked, this,
            &ccMPlaneDlg::onCloseButtonPressed);
    connect(m_tabWidget, &QTabWidget::currentChanged, this,
            &ccMPlaneDlg::onTabChanged);

    m_tabWidget->setTabEnabled(1, false);
    adjustSize();
}

void ccMPlaneDlg::createPlaneFittingTab() {
    // Create the tab
    m_tabPlaneFitting = new QWidget();
    m_tabPlaneFitting->setLayout(new QVBoxLayout());

    // Create the table
    m_pointTableWidget = new QTableWidget();
    m_tabPlaneFitting->layout()->addWidget(m_pointTableWidget);
    initializeFittingPointTable();

    m_pointTableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_pointTableWidget->setFocusPolicy(Qt::NoFocus);
    m_pointTableWidget->setSelectionMode(QAbstractItemView::NoSelection);
    m_pointTableWidget->setSizePolicy(QSizePolicy::Minimum,
                                      QSizePolicy::Minimum);
    m_pointTableWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_pointTableWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_pointTableWidget->horizontalHeader()->setStyleSheet(
            "QHeaderView::section { background-color: #EDEDED; }");
    m_pointTableWidget->verticalHeader()->setStyleSheet(
            "QHeaderView::section { background-color: #EDEDED; }");

    m_pointTableWidget->resizeColumnsToContents();
    m_pointTableMinWidth = static_cast<unsigned int>(
            (m_pointTableWidget->horizontalHeader()->length() +
             m_pointTableWidget->verticalHeader()->width() +
             m_pointTableWidget->verticalScrollBar()->width()) *
            1.5);
    m_pointTableMinHeight = static_cast<unsigned int>(
            m_pointTableWidget->verticalHeader()->length() +
            m_pointTableWidget->horizontalHeader()->height() +
            m_pointTableWidget->horizontalScrollBar()->height());
    m_pointTableWidget->setFixedSize(m_pointTableMinWidth,
                                     m_pointTableMinHeight);

    // Stretch the dialog by default
    m_pointTableWidget->horizontalHeader()->setSectionResizeMode(
            QHeaderView::Stretch);
    m_pointTableWidget->verticalHeader()->setSectionResizeMode(
            QHeaderView::Stretch);
}

void ccMPlaneDlg::createMeasurementTab() {
    // Create the tab
    m_tabMeasurement = new QWidget();
    m_tabMeasurement->setLayout(new QVBoxLayout());

    // Create the table
    m_measurementTableWidget = new QTableWidget();
    m_radioButtonSignedMeasurement = new QRadioButton("Signed measurement");
    m_radioButtonUnsignedMeasurement = new QRadioButton("Unsigned measurement");
    m_checkBoxShowNormal = new QCheckBox("Show normal vector");

    QWidget *savingWidget = new QWidget();
    QHBoxLayout *hLayout = new QHBoxLayout();
    hLayout->setSpacing(0);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    hLayout->setMargin(0);
#endif
    hLayout->setContentsMargins(0, 0, 0, 0);

    QPushButton *buttonSaving = new QPushButton("Save");
    buttonSaving->setIcon(QIcon(":/CC/plugin/qMPlane/images/ccSave.png"));
    buttonSaving->setContentsMargins(0, 0, 0, 0);

    savingWidget->setLayout(hLayout);
    savingWidget->layout()->addWidget(buttonSaving);
    hLayout->addStretch();

    m_tabMeasurement->layout()->addWidget(m_radioButtonSignedMeasurement);
    m_tabMeasurement->layout()->addWidget(m_radioButtonUnsignedMeasurement);
    m_tabMeasurement->layout()->addWidget(m_checkBoxShowNormal);
    m_tabMeasurement->layout()->addWidget(m_measurementTableWidget);
    m_tabMeasurement->layout()->addWidget(savingWidget);

    m_measurementTableWidget->setRowCount(3);
    m_measurementTableWidget->setColumnCount(2);
    QStringList header1;
    header1 << "Measurement"
            << "Distance";
    m_measurementTableWidget->setHorizontalHeaderLabels(header1);
    m_radioButtonUnsignedMeasurement->setChecked(true);

    m_measurementTableWidget->setSizePolicy(QSizePolicy::Minimum,
                                            QSizePolicy::Minimum);
    m_measurementTableWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_measurementTableWidget->setHorizontalScrollBarPolicy(
            Qt::ScrollBarAlwaysOn);
    m_measurementTableWidget->horizontalHeader()->setStyleSheet(
            "QHeaderView::section { background-color: #EDEDED; }");

    m_measurementTableWidget->resizeColumnsToContents();
    m_measurementTableWidget->setFixedSize(
            m_pointTableMinWidth,
            m_measurementTableWidget->verticalHeader()->length() +
                    m_measurementTableWidget->horizontalHeader()->height() +
                    m_measurementTableWidget->horizontalScrollBar()->height() *
                            1.5);

    // Stretch the dialog by default
    m_measurementTableWidget->horizontalHeader()->setSectionResizeMode(
            QHeaderView::Stretch);
    m_measurementTableWidget->verticalHeader()->setSectionResizeMode(
            QHeaderView::Stretch);
    m_measurementTableWidget->verticalHeader()->hide();

    connect(m_measurementTableWidget, &QTableWidget::itemChanged, this,
            &ccMPlaneDlg::onMeasurementPointNameChanged);
    connect(m_radioButtonSignedMeasurement, &QRadioButton::clicked, this,
            &ccMPlaneDlg::onRadioButtonClicked);
    connect(m_radioButtonUnsignedMeasurement, &QRadioButton::clicked, this,
            &ccMPlaneDlg::onRadioButtonClicked);
    connect(m_checkBoxShowNormal, &QCheckBox::clicked, this,
            &ccMPlaneDlg::onShowNormalCheckBox);
    connect(buttonSaving, &QCheckBox::clicked, this,
            &ccMPlaneDlg::onSaveButtonClicked);
}

int ccMPlaneDlg::getFittingPointContentWidth() const {
    m_pointTableWidget->sizeHint();
    return m_pointTableWidget->columnWidth(0) +
           m_pointTableWidget->columnWidth(1) +
           m_pointTableWidget->columnWidth(2) +
           m_pointTableWidget->columnWidth(3);
}

int ccMPlaneDlg::getFittingPointTableWidth() const {
    return m_pointTableWidget->width() -
           m_pointTableWidget->verticalHeader()->width() -
           m_pointTableWidget->verticalScrollBar()->width();
}

QPushButton *ccMPlaneDlg::createDeleteButton() {
    QPushButton *deleteButton = new QPushButton();
    deleteButton->setIcon(QIcon(":/CC/plugin/qMPlane/images/icon_delete.png"));
    deleteButton->setStyleSheet(
            "QPushButton {border-style: outset; border-width: 0px;}");
    connect(deleteButton, SIGNAL(clicked()), this,
            SLOT(onDeleteButtonClicked()));
    return deleteButton;
}

void ccMPlaneDlg::addFittingPoint(int rowIndex, const CCVector3 &point) {
    m_pointTableWidget->horizontalHeader()->setSectionResizeMode(
            QHeaderView::ResizeToContents);

    if ((rowIndex + 1) > m_pointTableWidget->rowCount()) {
        m_pointTableWidget->insertRow(m_pointTableWidget->rowCount());
    }

    QTableWidgetItem itemPrefab;
    itemPrefab.setTextAlignment(Qt::AlignCenter);

    m_pointTableWidget->setVerticalHeaderItem(
            rowIndex, new QTableWidgetItem(QString("Point %1").arg(rowIndex)));

    QTableWidgetItem *xItem = itemPrefab.clone();
    xItem->setText(QString::number(point.x, 'f', 3));
    m_pointTableWidget->setItem(rowIndex, 0, xItem);

    QTableWidgetItem *yItem = itemPrefab.clone();
    yItem->setText(QString::number(point.y, 'f', 3));
    m_pointTableWidget->setItem(rowIndex, 1, yItem);

    QTableWidgetItem *zItem = itemPrefab.clone();
    zItem->setText(QString::number(point.z, 'f', 3));
    m_pointTableWidget->setItem(rowIndex, 2, zItem);

    m_pointTableWidget->setCellWidget(rowIndex, 3, createDeleteButton());

    m_pointTableWidget->resizeColumnToContents(0);
    m_pointTableWidget->resizeColumnToContents(1);
    m_pointTableWidget->resizeColumnToContents(2);

    // Finally check if actual content is smaller than the table
    // If so -> apply stretch to fill the available area
    if (getFittingPointContentWidth() <= getFittingPointTableWidth()) {
        m_pointTableWidget->horizontalHeader()->setSectionResizeMode(
                QHeaderView::Stretch);
    }
}

void ccMPlaneDlg::selectFittingPoint(unsigned int rowIndex) {
    m_pointTableWidget->selectRow(rowIndex);
}

void ccMPlaneDlg::clearFittingPoints() {
    m_pointTableWidget->setRowCount(0);
    initializeFittingPointTable();
}

void ccMPlaneDlg::addMeasurementPoint(const QString &name, float distance) {
    QTableWidgetItem itemPrefab;
    itemPrefab.setTextAlignment(Qt::AlignCenter);

    m_measurementTableWidget->blockSignals(true);

    unsigned int rowCount = m_measurementTableWidget->rowCount();
    m_measurementTableWidget->insertRow(rowCount);

    QTableWidgetItem *idItem = itemPrefab.clone();
    idItem->setText(name);
    m_measurementTableWidget->setItem(rowCount, 0, idItem);

    QTableWidgetItem *distanceItem = itemPrefab.clone();
    distanceItem->setText(QString::number(distance, 'f', 3));
    distanceItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    m_measurementTableWidget->setItem(rowCount, 1, distanceItem);
    m_measurementTableWidget->selectRow(rowCount);

    m_measurementTableWidget->blockSignals(false);
}

void ccMPlaneDlg::renameMeasurement(const QString &name,
                                    unsigned int rowIndex) {
    m_measurementTableWidget->blockSignals(true);
    m_measurementTableWidget->item(rowIndex, 0)->setText(name);
    m_measurementTableWidget->blockSignals(false);
}

void ccMPlaneDlg::enableMeasurementTab(bool enable) {
    m_tabWidget->setTabEnabled(1, enable);
}

void ccMPlaneDlg::clearMeasurementPoints() {
    m_measurementTableWidget->setRowCount(0);
}

bool ccMPlaneDlg::isSignedMeasurement() const {
    return m_radioButtonSignedMeasurement->isChecked();
}

bool ccMPlaneDlg::isNormalVectorChecked() const {
    return m_checkBoxShowNormal->isChecked();
}

void ccMPlaneDlg::initializeFittingPointTable() {
    m_pointTableWidget->setRowCount(6);
    m_pointTableWidget->setColumnCount(4);
    QStringList header;
    header << "x"
           << "y"
           << "z"
           << "";
    m_pointTableWidget->setHorizontalHeaderLabels(header);
    for (unsigned int i = 0; i < 6; ++i) {
        m_pointTableWidget->setVerticalHeaderItem(
                i, new QTableWidgetItem(QString("Point %1").arg(i)));
    }
}

void ccMPlaneDlg::onCloseButtonPressed() { emit signalCloseButtonPressed(); }

void ccMPlaneDlg::onTabChanged(int tab) { emit signalTabChanged(tab); }

void ccMPlaneDlg::onRadioButtonClicked() {
    emit signalMeasurementModeChanged();
}

void ccMPlaneDlg::onDeleteButtonClicked() {
    QWidget *w = qobject_cast<QWidget *>(sender());
    if (w) {
        int index = m_pointTableWidget->indexAt(w->pos()).row();
        m_pointTableWidget->removeRow(index);
        emit signalFittingPointDelete(index);
    }
}

void ccMPlaneDlg::onShowNormalCheckBox(bool checked) {
    emit signalShowNormalCheckBoxClicked(m_checkBoxShowNormal->isChecked());
}

void ccMPlaneDlg::onSaveButtonClicked() { emit signalSaveButtonClicked(); }

void ccMPlaneDlg::onMeasurementPointNameChanged(QTableWidgetItem *item) {
    emit signalMeasureNameChanged(item);
}
