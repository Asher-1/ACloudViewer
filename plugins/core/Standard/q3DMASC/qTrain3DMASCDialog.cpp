// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qTrain3DMASCDialog.h"

// Qt
#include <CVLog.h>
#include <QtCompat.h>

#include <QDateTime>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <QStandardPaths>
#include <QTableWidgetItem>

// System
#include <assert.h>

#include <iostream>

static const int FeatureImportanceColumn = 1;

Train3DMASCDialog::Train3DMASCDialog(QWidget* parent /*=nullptr*/)
    : QDialog(parent),
      Ui::Train3DMASCDialog(),
      classifierSaved(false),
      saveRequested(false),
      traceFileConfigured(false),
      m_traceFile(nullptr),
      run(0) {
    setupUi(this);

    QDateTime dateTime = QDateTime::currentDateTime();
    m_baseName = "3dmasc_" + dateTime.toString("yyyyMMdd_hh") + "h" +
                 dateTime.toString("mm");

    readSettings();

    connect(closePushButton, SIGNAL(clicked()), this, SLOT(onClose()));
    connect(savePushButton, SIGNAL(clicked()), this, SLOT(onSave()));
    connect(exportToolButton, SIGNAL(clicked()), this, SLOT(onExportResults()));
}

Train3DMASCDialog::~Train3DMASCDialog() {
    writeSettings();
    closeTraceFile();
    for (auto m : toDeleteLater) {
        if (m != nullptr) delete m;
    }
}

void Train3DMASCDialog::readSettings() {
    QSettings settings;
    settings.beginGroup("3DMASC");
    bool keepAttributes = settings.value("keepAttributes", false).toBool();
    this->keepAttributesCheckBox->setChecked(keepAttributes);
    bool saveTrace = settings.value("saveTrace", false).toBool();
    setCheckBoxSaveTrace(saveTrace);
}

void Train3DMASCDialog::writeSettings() {
    QSettings settings;
    settings.beginGroup("3DMASC");
    settings.setValue("keepAttributes", keepAttributesCheckBox->isChecked());
    settings.setValue("saveTrace", checkBox_keepTraces->isChecked());
}

void Train3DMASCDialog::clearResults() {
    resultLabel->clear();
    tableWidget->clear();
}

int Train3DMASCDialog::addFeature(QString name,
                                  float importance,
                                  bool isChecked /*=true*/) {
    int index = tableWidget->rowCount();
    tableWidget->setRowCount(index + 1);

    QTableWidgetItem* nameItem = new QTableWidgetItem(name);
    nameItem->setCheckState(isChecked ? Qt::Checked : Qt::Unchecked);
    tableWidget->setItem(index, 0, nameItem);

    QTableWidgetItem* importanceItem = new QTableWidgetItem(
            std::isnan(importance) ? QString() : QString::number(importance));
    tableWidget->setItem(index, 1, importanceItem);

    return index;
}

int Train3DMASCDialog::addScale(double scale, bool isChecked /*=true*/) {
    int index = tableWidgetScales->rowCount();
    tableWidgetScales->setRowCount(index + 1);

    QTableWidgetItem* nameItem = new QTableWidgetItem(QString::number(scale));
    nameItem->setCheckState(isChecked ? Qt::Checked : Qt::Unchecked);
    tableWidgetScales->setItem(index, 0, nameItem);

    return index;
}

void Train3DMASCDialog::scaleStateChanged(QTableWidgetItem* item) {
    // if the scale is not checked, remove automatically features based on this
    // scale
    QString scale = "_SC" + item->text() + "_";
    for (int row = 0; row < tableWidget->rowCount(); row++) {
        QTableWidgetItem* nameItem = tableWidget->item(row, 0);
        QString name = nameItem->text();
        if (name.contains(scale)) nameItem->setCheckState(item->checkState());
    }
}

void Train3DMASCDialog::connectScaleSelectionToFeatureSelection() {
    connect(tableWidgetScales, &QTableWidget::itemChanged, this,
            &Train3DMASCDialog::scaleStateChanged);
}

void Train3DMASCDialog::setResultText(QString text) {
    resultLabel->setText(text);
}

void Train3DMASCDialog::setFirstRunDone() {
    runPushButton->setText(tr("Retry"));
    savePushButton->setEnabled(true);
}

bool Train3DMASCDialog::isFeatureSelected(QString featureName) const {
    for (int index = 0; index < tableWidget->rowCount(); ++index) {
        QTableWidgetItem* item = tableWidget->item(index, 0);
        if (item->text() == featureName) {
            return (item->checkState() == Qt::Checked);
        }
    }

    assert(false);
    return false;
}

void Train3DMASCDialog::sortByFeatureImportance() {
    tableWidget->sortByColumn(FeatureImportanceColumn, Qt::DescendingOrder);
}

void Train3DMASCDialog::setFeatureImportance(QString featureName,
                                             float importance) {
    for (int index = 0; index < tableWidget->rowCount(); ++index) {
        if (tableWidget->item(index, 0)->text() == featureName) {
            QTableWidgetItem* item =
                    tableWidget->item(index, FeatureImportanceColumn);
            item->setText(std::isnan(importance)
                                  ? QString()
                                  : QString::number(importance, 'f', 6));
            return;
        }
    }

    assert(false);
}

void Train3DMASCDialog::onClose() {
    if (!classifierSaved &&
        QMessageBox::question(this, "Classifier not saved",
                              "Classifier not saved. Do you confirm you want "
                              "to close the tool?",
                              QMessageBox::Yes,
                              QMessageBox::No) == QMessageBox::No)
        return;

    reject();
}

void Train3DMASCDialog::onSave() {
    saveRequested = true;
    accept();
}

void Train3DMASCDialog::onExportResults(QString filePath /*=""*/) {
    QSettings settings;
    settings.beginGroup("3DMASC");
    QString outputPath =
            settings.value("FilePath", QCoreApplication::applicationDirPath())
                    .toString();
    QString outputFilename = QFileDialog::getSaveFileName(
            this, "Export feature importance matrix", outputPath, "*.csv");
    if (outputFilename.isNull()) {
        // process cancelled by the user
        return;
    }
    settings.setValue("FilePath", QFileInfo(outputFilename).absolutePath());
    settings.endGroup();

    // save the file
    QFile file(outputFilename);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        QMessageBox::critical(
                this, "Error",
                "Failed to open file for writing: " + outputFilename);
        return;
    }

    QTextStream stream(&file);
    stream << "Feature;Importance" << QtCompat::endl;
    for (int index = 0; index < tableWidget->rowCount(); ++index) {
        QString featureName = tableWidget->item(index, 0)->text();
        QString importance =
                tableWidget->item(index, FeatureImportanceColumn)->text();
        stream << featureName << ";" << importance << QtCompat::endl;
    }
}

void Train3DMASCDialog::addConfusionMatrixAndSaveTraces(
        ConfusionMatrix* confusionMatrix) {
    toDeleteLater.push_back(confusionMatrix);
    saveTraces(confusionMatrix);
}

void Train3DMASCDialog::setInputFilePath(QString filePath) {
    m_parameterFilePath = filePath;
}

void Train3DMASCDialog::setCheckBoxSaveTrace(bool state) {
    checkBox_keepTraces->setChecked(state);
}

bool Train3DMASCDialog::openTraceFile() {
    QString traceFileName;
    QString traceFilePath;

    // get currentPath
    QFileInfo info(m_parameterFilePath);
    QDir parameterDir = QDir(info.path());

    QFileDialog dialog(this);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    // Qt5.15+ and Qt6: Use new API (DirectoryOnly is deprecated)
    dialog.setFileMode(QFileDialog::Directory);
    dialog.setOption(QFileDialog::ShowDirsOnly, true);
#else
    // Qt5.12-5.14: Use old API
    dialog.setFileMode(QFileDialog::DirectoryOnly);
#endif
    dialog.setWindowTitle("Choose a valid directory for the traces");
    dialog.setDirectory(
            QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation)
                    .at(0));

    if (!m_traceFile)  // create trace file if it does not exists already
    {
        m_tracePath = parameterDir.absolutePath() + "/" + m_baseName;
        traceFileName = m_baseName + ".txt";

        if (parameterDir.exists(m_baseName)) {
            QDateTime dateTime = QDateTime::currentDateTime();
            m_baseName += "min" + dateTime.toString("ss") + "s";
        }
        if (!parameterDir.mkdir(m_baseName))  // create a specific directory to
                                              // store the traces
        {
            CVLog::Error("impossible to save in the default directory: " +
                         m_tracePath);
            // impossible to save in the default directory, you have to propose
            // another path
            if (dialog.exec())
                m_tracePath = dialog.selectedFiles().at(0);
            else
                return false;
        } else
            CVLog::Print("directory for traces created: " + m_tracePath);

        traceFilePath = m_tracePath + "/" + traceFileName;
        m_traceFile = new QFile(traceFilePath);

        if (!m_traceFile->open(QIODevice::WriteOnly | QIODevice::Text)) {
            CVLog::Error("impossible to open trace file: " + traceFilePath);
            delete m_traceFile;
            m_tracePath.clear();
            return false;
        }
    }

    if (m_traceFile->isOpen()) {
        traceFileConfigured = true;
        CVLog::Print("save trace in: " + traceFilePath);
        m_traceStream.setDevice(m_traceFile);
        m_traceStream << "run overallAccuracy\n";
        return true;
    } else {
        return false;
    }
}

bool Train3DMASCDialog::closeTraceFile() {
    if (m_traceFile)
        if (m_traceFile->isOpen()) {
            CVLog::Print("[3DMASC] traces stored in: " +
                         m_traceFile->fileName());
            m_traceFile->close();
        }

    return true;
}

void Train3DMASCDialog::saveTraces(ConfusionMatrix* confusionMatrix) {
    run++;  // increment the run number
    confusionMatrix->setSessionRun(m_baseName, run);
    if (checkBox_keepTraces->isChecked()) {
        if (!traceFileConfigured)  // if the trace file is not configured yet,
                                   // do it
        {
            if (!openTraceFile()) return;
        }
        // save the trace

        // save the run number and the overall accuracy
        if (m_traceStream.device())
            m_traceStream << run << " " << confusionMatrix->getOverallAccuracy()
                          << QtCompat::endl;
        confusionMatrix->save(m_tracePath + "/" + "run_" +
                              QString::number(run) + "_confusion_matrix.txt");
    }
}

bool Train3DMASCDialog::getSaveTrace() {
    return checkBox_keepTraces->isChecked();
}

QString Train3DMASCDialog::getTracePath() {
    if (traceFileConfigured) {
        QFileInfo fi(*m_traceFile);
        return fi.absoluteDir().absolutePath();
    } else if (openTraceFile()) {
        QFileInfo fi(*m_traceFile);
        return fi.absoluteDir().absolutePath();
    } else
        return QString();
}

int Train3DMASCDialog::getRun() { return run; }
