// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qCanupo2DViewDialog.h"

#include "qCanupo2DWidget.h"

// local
#include "qCanupoTools.h"

// CV_DB_LIB
#include <CVLog.h>
#include <ecvMainAppInterface.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// Qt
#include <QApplication>
#include <QColor>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>

// system
#include <assert.h>

#include <cmath>

static bool s_firstDisplay = true;

qCanupo2DViewDialog::qCanupo2DViewDialog(
        const CorePointDescSet* descriptors1,
        const CorePointDescSet* descriptors2,
        QString cloud1Name,
        QString cloud2Name,
        int class1 /*=1*/,
        int class2 /*=2*/,
        const CorePointDescSet* evaluationDescriptors /*=0*/,
        ecvMainAppInterface* app /*=0*/)
    : QDialog(app ? app->getActiveWindow() : nullptr),
      Ui::Canupo2DViewDialog(),
      m_app(app),
      m_classifierSaved(false),
      m_descriptors1(descriptors1),
      m_descriptors2(descriptors2),
      m_evaluationDescriptors(evaluationDescriptors),
      m_class1(class1),
      m_cloud1Name(cloud1Name),
      m_cloud2Name(cloud2Name),
      m_class2(class2),
      m_cloud(nullptr),
      m_poly(nullptr),
      m_polyVertices(nullptr),
      m_selectedPointIndex(-1),
      m_pickingRadius(5) {
    setupUi(this);

    // update legend
    cloud1NameLabel->setText(QString("class %1: ").arg(m_class1) +
                             m_cloud1Name);  // blue points
    cloud2NameLabel->setText(QString("class %1: ").arg(m_class2) +
                             m_cloud2Name);  // red points

    s_firstDisplay = true;

    // setup native 2D view (QPainter-based, no VTK/OpenGL dependency)
    {
        m_2dView = new qCanupo2DWidget(this);
        viewFrame->setLayout(new QHBoxLayout());
        viewFrame->layout()->setContentsMargins(0, 0, 0, 0);
        viewFrame->layout()->addWidget(m_2dView);

        connect(m_2dView, &qCanupo2DWidget::leftButtonClicked, this,
                &qCanupo2DViewDialog::addOrSelectPoint);
        connect(m_2dView, &qCanupo2DWidget::rightButtonClicked, this,
                &qCanupo2DViewDialog::removePoint);
        connect(m_2dView, &qCanupo2DWidget::mouseMoved, this,
                &qCanupo2DViewDialog::moveSelectedPoint);
        connect(m_2dView, &qCanupo2DWidget::buttonReleased, this,
                &qCanupo2DViewDialog::deselectPoint);
    }

    updateScalesList(true);

    connect(resetToolButton, SIGNAL(clicked()), this, SLOT(resetBoundary()));
    connect(statisticsToolButton, SIGNAL(clicked()), this,
            SLOT(computeStatistics()));
    connect(savePushButton, SIGNAL(clicked()), this, SLOT(saveClassifier()));
    connect(donePushButton, SIGNAL(clicked()), this, SLOT(checkBeforeAccept()));
    connect(pointSizeSpinBox, SIGNAL(valueChanged(int)), this,
            SLOT(setPointSize(int)));
    connect(scalesCountSpinBox, SIGNAL(valueChanged(int)), this,
            SLOT(onScalesCountSpinBoxChanged(int)));
}

qCanupo2DViewDialog::~qCanupo2DViewDialog() { reset(); }

void qCanupo2DViewDialog::updateScalesList(bool firstTime) {
    if (!m_descriptors1 || !m_descriptors2) {
        scalesListLineEdit->setText("Invalid descriptors!");
        scalesCountSpinBox->setEnabled(false);
    } else {
        const std::vector<float>& allScales = m_descriptors1->scales();
        int maxScaleCount = static_cast<int>(allScales.size());
        scalesCountSpinBox->setRange(1, maxScaleCount);
        if (firstTime) scalesCountSpinBox->setValue(maxScaleCount);

        int currentScaleCount = scalesCountSpinBox->value();
        QStringList scalesList;
        for (int i = 0; i < currentScaleCount; ++i)
            scalesList << QString::number(
                    allScales[maxScaleCount - currentScaleCount + i]);

        scalesListLineEdit->setText(scalesList.join(" "));
        scalesCountSpinBox->setEnabled(true);
    }
}

void qCanupo2DViewDialog::reset() {
    if (m_poly) delete m_poly;
    m_poly = nullptr;
    m_polyVertices = nullptr;

    if (m_cloud) delete m_cloud;
    m_cloud = nullptr;

    if (m_2dView) {
        m_2dView->setCloud(nullptr);
        m_2dView->setPolyline(nullptr);
        m_2dView->update();
    }
}

void qCanupo2DViewDialog::getActiveScales(std::vector<float>& scales) const {
    scales.clear();

    if (!m_descriptors1) return;

    const std::vector<float>& allScales = m_descriptors1->scales();
    int maxScaleCount = static_cast<int>(allScales.size());
    int currentScaleCount = scalesCountSpinBox->value();
    assert(currentScaleCount >= 1 && currentScaleCount <= maxScaleCount);
    currentScaleCount = std::min<int>(currentScaleCount, maxScaleCount);
    scales.resize(currentScaleCount);
    for (int i = 0; i < currentScaleCount; ++i) {
        scales[i] = allScales[maxScaleCount - currentScaleCount + i];
    }
}

// Training in progress
static bool s_training = false;
static int s_trainingValue = 0;

bool qCanupo2DViewDialog::trainClassifier() {
    if (!m_descriptors1 || !m_descriptors2) return false;

    s_training = true;
    s_trainingValue = scalesCountSpinBox->value();
    statisticsToolButton->setEnabled(false);
    setEnabled(false);
    QApplication::processEvents();

    std::vector<float> scales;
    getActiveScales(scales);

    // reset display
    reset();

    // Reset classifier and re-train it!
    m_classifier = Classifier();
    m_classifier.class1 = m_class1;
    m_classifier.class2 = m_class2;

    m_cloud = new ccPointCloud("CANUPO projections");
    if (!qCanupoTools::TrainClassifier(m_classifier, *m_descriptors1,
                                       *m_descriptors2, scales, m_cloud,
                                       m_evaluationDescriptors, m_app)) {
        delete m_cloud;
        m_cloud = nullptr;

        s_training = false;
        setEnabled(true);

        return false;
    }

    // Set cloud on the 2D view
    m_2dView->setCloud(m_cloud);

    // Show reference points as markers
    m_2dView->clearMarkers();
    m_2dView->addMarker(m_classifier.refPointPos.x, m_classifier.refPointPos.y,
                        QColor(255, 0, 0), 6.0);
    m_2dView->addMarker(m_classifier.refPointNeg.x, m_classifier.refPointNeg.y,
                        QColor(0, 0, 255), 6.0);

    // update/create boundary representation
    resetBoundary();

    if (s_firstDisplay) {
        updateZoom();
        s_firstDisplay = false;
    }

    s_training = false;
    setEnabled(true);

    statisticsToolButton->setEnabled(true);

    return true;
}

void qCanupo2DViewDialog::onScalesCountSpinBoxChanged(int value) {
    Q_UNUSED(value);
    if (s_training) {
        scalesCountSpinBox->blockSignals(true);
        scalesCountSpinBox->setValue(s_trainingValue);
        scalesCountSpinBox->blockSignals(false);
    } else {
        updateScalesList(false);
        trainClassifier();
    }
}

void qCanupo2DViewDialog::computeStatistics() {
    qCanupoTools::EvalParameters params;
    std::vector<float> scales;
    getActiveScales(scales);

    Classifier classifier = m_classifier;
    updateClassifierPath(classifier);

    if (qCanupoTools::EvaluateClassifier(classifier, *m_descriptors1,
                                         *m_descriptors2, scales, params)) {
        QStringList info;
        info << QString("Class %1 (%2)").arg(m_class1).arg(m_cloud1Name);
        info << QString("\tTotal: %1").arg(params.true1 + params.false1);
        info << QString("\tTruly classified: %1").arg(params.true1);
        info << QString("\tFalsely classified: %1").arg(params.false1);
        info << QString("\tDist. to boundary: %1 +/- %2")
                        .arg(params.mu1)
                        .arg(sqrt(params.var1));
        info << QString("");
        info << QString("Class %1 (%2)").arg(m_class2).arg(m_cloud2Name);
        info << QString("\tTotal: %1").arg(params.true2 + params.false2);
        info << QString("\tTruly classified: %1").arg(params.true2);
        info << QString("\tFalsely classified: %1").arg(params.false2);
        info << QString("\tDist. to boundary: %1 +/- %2")
                        .arg(params.mu2)
                        .arg(sqrt(params.var2));
        info << QString("");
        info << QString("Balanced accuracy (ba) = %1").arg(params.ba());
        info << QString("Fisher Discriminant Ratio (fdr) = %1")
                        .arg(params.fdr());

        QMessageBox::information(this, "Statistics", info.join("\n"),
                                 QMessageBox::Ok);
    }
}

void qCanupo2DViewDialog::setPointSize(int value) {
    if (m_2dView) {
        m_2dView->setPointSize(value);
    }
}

void qCanupo2DViewDialog::checkBeforeAccept() {
    if (!m_classifierSaved) {
        if (QMessageBox::warning(this, "Classifier has not been saved!",
                                 "Do you really want to close the dialog "
                                 "before saving the classifier?",
                                 QMessageBox::Yes,
                                 QMessageBox::No) == QMessageBox::No)
            return;
    }

    accept();
}

void qCanupo2DViewDialog::resetBoundary() {
    if (!m_poly) {
        assert(!m_polyVertices);
        m_polyVertices = new ccPointCloud("vertices");
        m_poly = new ccPolyline(m_polyVertices);
        m_poly->addChild(m_polyVertices);
        m_poly->setColor(ecvColor::magenta);
        m_poly->showColors(true);
        m_poly->setWidth(2);
        m_poly->showVertices(true);
        m_poly->setVertexMarkerWidth(4);
    }

    m_poly->clear();
    m_polyVertices->clear();

    unsigned pathLength = static_cast<unsigned>(m_classifier.path.size());
    if (pathLength > 1) {
        m_polyVertices->reserve(pathLength);
        m_poly->reserve(pathLength);
        for (unsigned i = 0; i < pathLength; ++i) {
            m_polyVertices->addPoint(CCVector3(m_classifier.path[i].x,
                                               m_classifier.path[i].y, 0));
            m_poly->addPointIndex(i);
        }
    }

    if (m_2dView) {
        m_2dView->setPolyline(m_poly);
        m_2dView->update();
    }
}

void qCanupo2DViewDialog::saveClassifier() {
    QSettings settings("qCanupo");
    settings.beginGroup("Classif");
    QString currentPath =
            settings.value("MscCurrentPath", QApplication::applicationDirPath())
                    .toString();

    QString filename = QFileDialog::getSaveFileName(this, "Save Classifier",
                                                    currentPath, "*.prm");
    if (filename.isEmpty()) return;

    Classifier classifier = m_classifier;
    updateClassifierPath(classifier);

    QString error;
    if (classifier.save(filename, error)) {
        m_classifierSaved = true;
        if (m_app)
            m_app->dispToConsole(
                    QString("Classifier file saved: '%1'").arg(filename));
    } else {
        if (m_app)
            m_app->dispToConsole(error,
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }

    currentPath = QFileInfo(filename).absolutePath();
    settings.setValue("MscCurrentPath", currentPath);
}

void qCanupo2DViewDialog::updateClassifierPath(Classifier& classifier) const {
    if (m_poly) {
        classifier.path.resize(m_poly->size());
        for (unsigned i = 0; i < m_poly->size(); ++i) {
            const CCVector3* P = m_poly->getPoint(i);
            classifier.path[i] = Classifier::Point2D(P->x, P->y);
        }
    }
}

void qCanupo2DViewDialog::updateZoom() {
    if (m_2dView) {
        m_2dView->zoomFit();
    }
}

void qCanupo2DViewDialog::setPickingRadius(int radius) {
    m_pickingRadius = std::max(radius, 1);
}

CCVector3 qCanupo2DViewDialog::getClickPos(int x, int y) const {
    if (!m_2dView) {
        assert(false);
        return CCVector3(0, 0, 0);
    }
    QPointF wp = m_2dView->screenToWorld(x, y);
    return CCVector3(static_cast<PointCoordinateType>(wp.x()),
                     static_cast<PointCoordinateType>(wp.y()), 0);
}

int qCanupo2DViewDialog::getClosestVertex(int x, int y, CCVector3& P) const {
    if (!m_poly || !m_2dView) return -1;

    P = getClickPos(x, y);

    int closeIndex = -1;
    float closestSquareDist = 0;
    for (unsigned i = 0; i < m_poly->size(); ++i) {
        float squareDist = (*m_poly->getPoint(i) - P).norm2();
        if (closeIndex < 0 || squareDist < closestSquareDist) {
            closestSquareDist = squareDist;
            closeIndex = static_cast<int>(i);
        }
    }

    return closeIndex;
}

void qCanupo2DViewDialog::addOrSelectPoint(int x, int y) {
    if (!m_poly || !m_2dView) return;

    CCVector3 P;
    int closeIndex = getClosestVertex(x, y, P);

    const CCVector3* B =
            (closeIndex >= 0 ? m_poly->getPoint(closeIndex) : nullptr);

    double maxPickingDist =
            static_cast<double>(m_pickingRadius) * m_2dView->pixelSize();

    if (closeIndex >= 0) {
        assert(B);
        if ((P - *B).norm() <= maxPickingDist) {
            m_selectedPointIndex = closeIndex;
            return;
        }
    }

    // look if the click falls 'inside' a segment
    double nearestProjectionDist = maxPickingDist;
    CCVector3 nearestProj = P;
    int nearestSegIndex = -1;
    for (unsigned i = 0; i + 1 < m_poly->size(); ++i) {
        const CCVector3* A = m_poly->getPoint(i);
        const CCVector3* B2 = m_poly->getPoint(i + 1);

        CCVector3 AB = (*B2 - *A);
        CCVector3 AP = (P - *A);

        PointCoordinateType dot = AB.dot(AP);
        dot /= AB.norm2();
        if (dot > 0 && dot < PC_ONE) {
            CCVector3 AH = AB * dot;
            CCVector3 HP = AP - AH;
            double dist = HP.norm();

            if (dist < nearestProjectionDist) {
                nearestProjectionDist = dist;
                nearestProj = AH + *A;
                nearestSegIndex = static_cast<int>(i);
            }
        }
    }

    if (nearestSegIndex >= 0) {
        closeIndex = nearestSegIndex;
        P = nearestProj;
    } else {
        const CCVector3* A = m_poly->getPoint(0);
        const CCVector3* B2 = m_poly->getPoint(m_poly->size() - 1);
        if ((P - *A).norm2() < (P - *B2).norm2()) {
            closeIndex = -1;
        } else {
            closeIndex = static_cast<int>(m_poly->size() - 1);
        }
    }

    ccPointCloud* vertices =
            dynamic_cast<ccPointCloud*>(m_poly->getAssociatedCloud());
    if (!vertices) {
        assert(false);
        return;
    }
    unsigned newIndexInCloud = vertices->size();
    vertices->reserve(newIndexInCloud + 1);
    vertices->addPoint(P);

    m_poly->reserve(m_poly->size() + 1);
    m_poly->addPointIndex(newIndexInCloud);

    m_selectedPointIndex = closeIndex + 1;

    unsigned newIndexInPoly = static_cast<unsigned>(m_selectedPointIndex);
    while (newIndexInPoly < m_poly->size()) {
        unsigned previousIndexInCloud =
                m_poly->getPointGlobalIndex(newIndexInPoly);
        m_poly->setPointIndex(newIndexInPoly, newIndexInCloud);
        newIndexInCloud = previousIndexInCloud;
        ++newIndexInPoly;
    }

    if (m_2dView) m_2dView->update();
}

void qCanupo2DViewDialog::removePoint(int x, int y) {
    if (!m_poly || !m_2dView) return;

    unsigned polySize = m_poly->size();
    if (polySize < 3) return;

    CCVector3 P;
    int closeIndex = getClosestVertex(x, y, P);
    if (closeIndex < 0) return;

    double maxPickingDist =
            static_cast<double>(m_pickingRadius) * m_2dView->pixelSize();

    const CCVector3* B = m_poly->getPoint(closeIndex);
    if ((P - *B).norm() > maxPickingDist) {
        return;
    }

    for (unsigned i = static_cast<unsigned>(closeIndex); i < polySize - 1; ++i)
        m_poly->setPointIndex(i, m_poly->getPointGlobalIndex(i + 1));
    m_poly->resize(polySize - 1);

    if (m_2dView) m_2dView->update();
}

void qCanupo2DViewDialog::moveSelectedPoint(int x,
                                            int y,
                                            Qt::MouseButtons buttons) {
    if (buttons != Qt::LeftButton) return;
    if (m_selectedPointIndex < 0) return;
    if (!m_poly || !m_2dView) return;

    CCVector3 newP = getClickPos(x, y);

    assert(static_cast<int>(m_poly->size()) > m_selectedPointIndex);
    CCVector3* P =
            const_cast<CCVector3*>(m_poly->getPoint(m_selectedPointIndex));
    *P = newP;

    m_2dView->update();
}

void qCanupo2DViewDialog::deselectPoint() { m_selectedPointIndex = -1; }
