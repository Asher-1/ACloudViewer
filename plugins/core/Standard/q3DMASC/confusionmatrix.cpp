// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "confusionmatrix.h"

#include <CVConst.h>
#include <CVLog.h>
#include <QtCompat.h>

#include <QBrush>
#include <QFile>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <set>

#include "ui_confusionmatrix.h"

static QColor GetColor(double value, double r1, double g1, double b1) {
    double r0 = 255.0;
    double g0 = 255.0;
    double b0 = 255.0;
    if (value < 0.05) {
        value = 0.05;
    } else if (value > 0.95) {
        value = 0.95;
    }
    int r = static_cast<int>((r1 - r0) * value + r0);
    int g = static_cast<int>((g1 - g0) * value + g0);
    int b = static_cast<int>((b1 - b0) * value + b0);
    //	CVLog::Warning("value " + QString::number(value) + " (" +
    // QString::number(r) + ", " + QString::number(g) + ", " +
    // QString::number(b)
    //+ ")");
    return QColor(r, g, b);
}

ConfusionMatrix::ConfusionMatrix(const std::vector<ScalarType>& actual,
                                 const std::vector<ScalarType>& predicted)
    : nbClasses(0),
      ui(new Ui::ConfusionMatrix),
      m_overallAccuracy(0.0f)

{
    ui->setupUi(this);
    this->setWindowFlag(Qt::WindowStaysOnTopHint);

    compute(actual, predicted);

    this->ui->tableWidget->resizeColumnsToContents();
    this->ui->tableWidget->setSizeAdjustPolicy(
            QAbstractScrollArea::AdjustToContents);
    QSize tableSize = this->ui->tableWidget->sizeHint();
    QSize widgetSize = QSize(tableSize.width() + 30, tableSize.height() + 50);
    this->setMinimumSize(widgetSize);
}

ConfusionMatrix::~ConfusionMatrix() {
    delete ui;
    ui = nullptr;
}

void ConfusionMatrix::computePrecisionRecallF1Score(
        cv::Mat& matrix, cv::Mat& precisionRecallF1Score, cv::Mat& vec_TP_FN) {
    int nbClasses = matrix.rows;

    // compute precision
    for (int predictedIdx = 0; predictedIdx < nbClasses; predictedIdx++) {
        float TP = 0;
        float FP = 0;
        for (int realIdx = 0; realIdx < nbClasses; realIdx++) {
            if (realIdx == predictedIdx)
                TP = matrix.at<int>(realIdx, realIdx);
            else
                FP += matrix.at<int>(realIdx, predictedIdx);
        }
        float TP_FP = TP + FP;
        if (TP_FP == 0)
            precisionRecallF1Score.at<float>(predictedIdx, PRECISION) =
                    NAN_VALUE;
        else
            precisionRecallF1Score.at<float>(predictedIdx, PRECISION) =
                    TP / TP_FP;
    }

    // compute recall
    for (int realIdx = 0; realIdx < nbClasses; realIdx++) {
        float TP = 0;
        float FN = 0;
        for (int predictedIdx = 0; predictedIdx < nbClasses; predictedIdx++) {
            if (realIdx == predictedIdx)
                TP = matrix.at<int>(realIdx, realIdx);
            else
                FN += matrix.at<int>(realIdx, predictedIdx);
        }
        float TP_FN = TP + FN;
        if (TP_FN == 0)
            precisionRecallF1Score.at<float>(realIdx, RECALL) = NAN_VALUE;
        else
            precisionRecallF1Score.at<float>(realIdx, RECALL) = TP / TP_FN;
        vec_TP_FN.at<int>(realIdx, 0) = TP_FN;
    }

    // compute F1-score
    for (int realIdx = 0; realIdx < nbClasses; realIdx++) {
        float den = precisionRecallF1Score.at<float>(realIdx, PRECISION) +
                    precisionRecallF1Score.at<float>(realIdx, RECALL);
        if (den == 0)
            precisionRecallF1Score.at<float>(realIdx, F1_SCORE) = NAN_VALUE;
        else
            precisionRecallF1Score.at<float>(realIdx, F1_SCORE) =
                    2 * precisionRecallF1Score.at<float>(realIdx, PRECISION) *
                    precisionRecallF1Score.at<float>(realIdx, RECALL) / den;
    }
}

float ConfusionMatrix::computeOverallAccuracy(cv::Mat& matrix) {
    int nbClasses = matrix.rows;
    float totalTrue = 0;
    float totalFalse = 0;

    m_overallAccuracy = 0.0;

    for (int realIdx = 0; realIdx < nbClasses; realIdx++) {
        for (int predictedIdx = 0; predictedIdx < nbClasses; predictedIdx++) {
            if (realIdx == predictedIdx)
                totalTrue += matrix.at<int>(realIdx, realIdx);
            else
                totalFalse += matrix.at<int>(realIdx, predictedIdx);
        }
    }
    if ((totalTrue + totalFalse) != 0)
        m_overallAccuracy = totalTrue / (totalTrue + totalFalse);
    else
        m_overallAccuracy = NAN_VALUE;

    return m_overallAccuracy;
}

void ConfusionMatrix::compute(const std::vector<ScalarType>& actual,
                              const std::vector<ScalarType>& predicted) {
    int idxActual;
    int idxPredicted;
    int actualClass;
    int predictedClass;

    // get the set of classes with the contents of the actual classes
    std::set<ScalarType> classes(actual.begin(), actual.end());
    int nbClasses = static_cast<int>(classes.size());
    confusionMatrix = cv::Mat(nbClasses, nbClasses, CV_32S, cv::Scalar(0));
    precisionRecallF1Score = cv::Mat(nbClasses, 3, CV_32F, cv::Scalar(0));
    cv::Mat vec_TP_FN(nbClasses, 1, CV_32S, cv::Scalar(0));

    // fill the confusion matrix
    for (int i = 0; i < actual.size(); i++) {
        actualClass = actual.at(i);
        idxActual = std::distance(classes.begin(), classes.find(actualClass));
        predictedClass = predicted.at(i);
        idxPredicted =
                std::distance(classes.begin(), classes.find(predictedClass));
        confusionMatrix.at<int>(idxActual, idxPredicted)++;
    }

    // compute precision recall F1-score
    computePrecisionRecallF1Score(confusionMatrix, precisionRecallF1Score,
                                  vec_TP_FN);
    float overallAccuracy = computeOverallAccuracy(confusionMatrix);

    // display the overall accuracy
    this->ui->label_overallAccuracy->setText(
            QString::number(overallAccuracy, 'g', 2));

    std::set<ScalarType>::iterator itB = classes.begin();
    std::set<ScalarType>::iterator itE = classes.end();
    class_numbers.assign(itB, itE);

    // BUILD THE QTABLEWIDGET

    this->ui->tableWidget->setColumnCount(
            2 + nbClasses +
            3);  // +2 for titles, +3 for precision / recall / F1-score
    this->ui->tableWidget->setRowCount(2 + nbClasses);
    // create a font for the table widgets
    QFont font;
    font.setBold(true);
    QTableWidgetItem* newItem = nullptr;
    // set the row and column names
    this->ui->tableWidget->setSpan(0, 0, 2, 2);          // empty area
    this->ui->tableWidget->setSpan(0, 2, 1, nbClasses);  // 'Predicted' header
    this->ui->tableWidget->setSpan(2, 0, nbClasses, 1);  // 'Actual' header
    this->ui->tableWidget->setSpan(0, 2 + nbClasses, 1, 3);  // empty area
    // Predicted
    newItem = new QTableWidgetItem("Predicted");
    newItem->setFont(font);
    newItem->setBackground(Qt::lightGray);
    newItem->setTextAlignment(Qt::AlignCenter);
    this->ui->tableWidget->setItem(0, 2, newItem);
    // Real
    newItem = new QTableWidgetItem("Real");
    newItem->setFont(font);
    newItem->setBackground(Qt::lightGray);
    newItem->setTextAlignment(Qt::AlignCenter);
    this->ui->tableWidget->setItem(2, 0, newItem);
    // add precision / recall / F1-score headers
    newItem = new QTableWidgetItem("Precision");
    newItem->setToolTip("TP / (TP + FP)");
    newItem->setFont(font);
    this->ui->tableWidget->setItem(1, 2 + nbClasses + PRECISION, newItem);
    newItem = new QTableWidgetItem("Recall");
    newItem->setToolTip("TP / (TP + FN)");
    newItem->setFont(font);
    this->ui->tableWidget->setItem(1, 2 + nbClasses + RECALL, newItem);
    newItem = new QTableWidgetItem("F1-score");
    newItem->setToolTip(
            "Harmonic mean of precision and recall (the closer to 1 the "
            "better)\n2 x precision x recall / (precision + recall)");
    newItem->setFont(font);
    this->ui->tableWidget->setItem(1, 2 + nbClasses + F1_SCORE, newItem);
    // add column names and row names
    for (int idx = 0; idx < class_numbers.size(); idx++) {
        QString str = QString::number(class_numbers[idx]);
        newItem = new QTableWidgetItem(str);
        newItem->setFont(font);
        this->ui->tableWidget->setItem(1, 2 + idx, newItem);
        newItem = new QTableWidgetItem(str);
        newItem->setFont(font);
        this->ui->tableWidget->setItem(2 + idx, 1, newItem);
    }

    // FILL THE QTABLEWIDGET

    // add the confusion matrix values
    for (int row = 0; row < nbClasses; row++)
        for (int column = 0; column < nbClasses; column++) {
            double val = confusionMatrix.at<int>(row, column);
            QTableWidgetItem* newItem =
                    new QTableWidgetItem(QString::number(val));
            if (row == column) {
                newItem->setBackground(
                        GetColor(val / vec_TP_FN.at<int>(row, 0), 0, 128, 255));
            } else {
                newItem->setBackground(
                        GetColor(val / vec_TP_FN.at<int>(row, 0), 200, 50, 50));
            }
            this->ui->tableWidget->setItem(2 + row, +2 + column, newItem);
        }

    // set precision / recall / F1-score values
    for (int realIdx = 0; realIdx < nbClasses; realIdx++) {
        newItem = new QTableWidgetItem(QString::number(
                precisionRecallF1Score.at<float>(realIdx, PRECISION), 'g', 2));
        this->ui->tableWidget->setItem(2 + realIdx, 2 + nbClasses + PRECISION,
                                       newItem);
        newItem = new QTableWidgetItem(QString::number(
                precisionRecallF1Score.at<float>(realIdx, RECALL), 'g', 2));
        this->ui->tableWidget->setItem(2 + realIdx, 2 + nbClasses + RECALL,
                                       newItem);
        newItem = new QTableWidgetItem(QString::number(
                precisionRecallF1Score.at<float>(realIdx, F1_SCORE), 'g', 2));
        this->ui->tableWidget->setItem(2 + realIdx, 2 + nbClasses + F1_SCORE,
                                       newItem);
    }
}

void ConfusionMatrix::setSessionRun(QString session, int run) {
    QString label;

    label = session + " / " + QString::number(run);

    this->ui->label_sessionRun->setText(label);
}

bool ConfusionMatrix::save(QString filePath) {
    QFile file(filePath);

    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        CVLog::Error("impossible to open file: " + filePath);
        return false;
    }

    QTextStream stream(&file);
    stream << "# columns: predicted classes\n# rows: actual classes\n";
    stream << "# last three colums: precision / recall / F1-score\n";
    for (auto class_number : class_numbers) {
        stream << class_number << " ";
    }
    stream << QtCompat::endl;
    for (int row = 0; row < confusionMatrix.rows; row++) {
        stream << class_numbers.at(row) << " ";
        for (int col = 0; col < confusionMatrix.cols; col++) {
            stream << confusionMatrix.at<int>(row, col) << " ";
        }
        stream << precisionRecallF1Score.at<float>(row, PRECISION) << " ";
        stream << precisionRecallF1Score.at<float>(row, RECALL) << " ";
        stream << precisionRecallF1Score.at<float>(row, F1_SCORE)
               << QtCompat::endl;
    }

    file.close();

    return true;
}

float ConfusionMatrix::getOverallAccuracy() { return m_overallAccuracy; }
