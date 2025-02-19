//##########################################################################
//#                                                                        #
//#                   CLOUDVIEWER  PLUGIN: qAnimation                      #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#             COPYRIGHT: Ryan Wicks, 2G Robotics Inc., 2015              #
//#                                                                        #
//##########################################################################

#include "qAnimationDlg.h"

// Local
#include "ViewInterpolate.h"

// CV_DB_LIB
#include <CVTools.h>
#include <ecv2DViewportObject.h>
#include <ecvDisplayTools.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// Qt
#include <QApplication>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSettings>
#include <QtGui>

// standard includes
#include <iomanip>
#include <vector>

#ifdef QFFMPEG_SUPPORT
// QTFFmpeg
#include <QVideoEncoder.h>
#endif

// System
#include <algorithm>
#if defined(CV_WINDOWS)
#include "windows.h"
#else
#include <unistd.h>
#endif

static const QString s_stepDurationKey("StepDurationSec");
static const QString s_stepEnabledKey("StepEnabled");

QStringList getImageList(const QString& path) {
    QDir dir(path);
    if (!dir.exists()) {
        return QStringList();
    }
    dir.setFilter(QDir::Files | QDir::NoSymLinks);

    QStringList filters;
    // we grab the list of supported image file formats (reading)
    QList<QByteArray> formats = QImageReader::supportedImageFormats();
    if (formats.empty()) {
        filters << "*.bmp"
                << "*.png"
                << "*.jpg";
    } else {
        // we convert this list into a proper "filters" string
        for (int i = 0; i < formats.size(); ++i) {
            QString filter = QString("*.%1").arg(formats[i].data());
            filters.append(filter);
        }
    }

    dir.setNameFilters(filters);
    QFileInfoList infoList = dir.entryInfoList();
    QStringList filesList;
    for (size_t i = 0; i < infoList.size(); i++) {
        filesList.append(infoList.at(i).absoluteFilePath());
    }
    return filesList;
}

qAnimationDlg::qAnimationDlg(QWidget* view3d, QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::AnimationDialog(), m_view3d(view3d) {
    setupUi(this);

    // restore previous settings
    QString defaultOutputFormat;
    {
        QSettings settings;
        settings.beginGroup("qAnimation");

        // last filename
        {
            QString defaultDir;
#ifdef _MSC_VER
            defaultDir = QApplication::applicationDirPath();
#else
            defaultDir = QDir::homePath();
#endif
            const QString defaultFileName(defaultDir + "/animation.mp4");
            QString lastFilename =
                    settings.value("filename", defaultFileName).toString();
            QString lastTexturePath =
                    settings.value("texturesPath", defaultDir).toString();
#ifndef QFFMPEG_SUPPORT
            lastFilename = QFileInfo(lastFilename).absolutePath();
#endif
            outputFileLineEdit->setText(lastFilename);
            inputTexturesPathLineEdit->setText(lastTexturePath);
            textureNumLabel->setText(
                    QString("Textures num: %1")
                            .arg(getImageList(lastTexturePath).size()));
        }

        // other parameters
        {
            bool startPreviewFromSelectedStep =
                    settings.value("previewFromSelected",
                                   previewFromSelectedCheckBox->isChecked())
                            .toBool();
            bool updateTextures =
                    settings.value("updateTextures",
                                   updateTexturesCheckBox->isChecked())
                            .toBool();
            bool loop =
                    settings.value("loop", loopCheckBox->isChecked()).toBool();
            int frameRate =
                    settings.value("frameRate", fpsSpinBox->value()).toInt();
            int superRes =
                    settings.value("superRes", superResolutionSpinBox->value())
                            .toInt();
            int renderingMode =
                    settings.value("renderingMode",
                                   renderingModeComboBox->currentIndex())
                            .toInt();
            int bitRate =
                    settings.value("bitRate", bitrateSpinBox->value()).toInt();
            bool autoStepDuration =
                    settings.value("autoStepDuration",
                                   autoStepDurationCheckBox->isChecked())
                            .toBool();
            bool smoothTrajectory =
                    settings.value("smoothTrajectory",
                                   smoothTrajectoryGroupBox->isChecked())
                            .toBool();
            double smoothRatio =
                    settings.value("smoothRatio",
                                   smoothRatioDoubleSpinBox->value())
                            .toDouble();
            defaultOutputFormat = settings.value("outputFormat").toString();

            previewFromSelectedCheckBox->setChecked(
                    startPreviewFromSelectedStep);
            updateTexturesCheckBox->setChecked(updateTextures);
            loopCheckBox->setChecked(loop);
            fpsSpinBox->setValue(frameRate);
            superResolutionSpinBox->setValue(superRes);
            renderingModeComboBox->setCurrentIndex(renderingMode);
            bitrateSpinBox->setValue(bitRate);
            autoStepDurationCheckBox->setChecked(
                    autoStepDuration);  // this might be modified when init will
                                        // be called!
            smoothTrajectoryGroupBox->setChecked(smoothTrajectory);
            smoothRatioDoubleSpinBox->setValue(smoothRatio);
        }

        settings.endGroup();
    }

    // populate the output format combo-box
    {
        outputFormatComboBox->addItem("Auto", QVariant(QString()));
#ifdef QFFMPEG_SUPPORT
        std::vector<QVideoEncoder::OutputFormat> formats;
        if (QVideoEncoder::GetSupportedOutputFormats(formats, true)) {
            int defaultIndex = 0;
            for (const QVideoEncoder::OutputFormat& f : formats) {
                QString title = f.longName;
                if (!f.extensions.isEmpty()) {
                    title += "[" + f.extensions + "]";
                    static const int s_maxTitleLength = 48;
                    if (title.size() > s_maxTitleLength)
                        title = title.left(s_maxTitleLength - 3) + "...";
                }
                outputFormatComboBox->addItem(title, QVariant(f.shortName));
                if (defaultIndex == 0 && !defaultOutputFormat.isEmpty() &&
                    defaultOutputFormat == f.shortName) {
                    defaultIndex = outputFormatComboBox->count() - 1;
                }
            }
            outputFormatComboBox->setCurrentIndex(defaultIndex);
        }
#endif
    }

    connect(autoStepDurationCheckBox, &QAbstractButton::toggled, this,
            &qAnimationDlg::onAutoStepsDurationToggled);
    connect(smoothTrajectoryGroupBox, &QGroupBox::toggled, this,
            &qAnimationDlg::onSmoothTrajectoryToggled);
    connect(smoothRatioDoubleSpinBox,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &qAnimationDlg::onSmoothRatioChanged);

    connect(fpsSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &qAnimationDlg::onFPSChanged);
    connect(totalTimeDoubleSpinBox,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &qAnimationDlg::onTotalTimeChanged);
    connect(stepTimeDoubleSpinBox,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &qAnimationDlg::onStepTimeChanged);
    connect(loopCheckBox, &QAbstractButton::toggled, this,
            &qAnimationDlg::onLoopToggled);

    connect(browseButton, &QAbstractButton::clicked, this,
            &qAnimationDlg::onBrowseButtonClicked);
    connect(browseTextureButton, &QAbstractButton::clicked, this,
            &qAnimationDlg::onBrowseTexturesButtonClicked);

    connect(previewButton, &QAbstractButton::clicked, this,
            &qAnimationDlg::preview);
    connect(renderButton, &QAbstractButton::clicked, this,
            &qAnimationDlg::renderAnimation);
    connect(exportFramesPushButton, &QAbstractButton::clicked, this,
            &qAnimationDlg::renderFrames);
    connect(buttonBox, &QDialogButtonBox::accepted, this,
            &qAnimationDlg::onAccept);
    connect(buttonBox, &QDialogButtonBox::rejected, this,
            &qAnimationDlg::onReject);
}

qAnimationDlg::~qAnimationDlg() {}

bool qAnimationDlg::smoothModeEnabled() const {
    return smoothTrajectoryGroupBox->isChecked() && !m_smoothVideoSteps.empty();
}

bool qAnimationDlg::updateTextures() const {
    return updateTexturesCheckBox->isChecked() && !m_mesh_list.empty();
}

bool qAnimationDlg::init(const std::vector<cc2DViewportObject*>& viewports,
                         const std::vector<ccMesh*>& meshes) {
    if (viewports.size() < 2 && meshes.empty()) {
        assert(false);
        return false;
    }

    try {
        m_videoSteps.resize(viewports.size());
        m_mesh_list.resize(meshes.size());
    } catch (const std::bad_alloc&) {
        // not enough memory
        return false;
    }

    ccBBox visibleObjectsBBox;
    ecvDisplayTools::GetVisibleObjectsBB(visibleObjectsBBox);

    for (size_t i = 0; i < viewports.size(); ++i) {
        cc2DViewportObject* vp = viewports[i];

        // check if the (1st) viewport has a duration in meta data (from a
        // previous run)
        double duration_sec = 2.0;
        if (vp->hasMetaData(s_stepDurationKey)) {
            duration_sec = vp->getMetaData(s_stepDurationKey).toDouble();
            // disable "auto step duration"
            autoStepDurationCheckBox->blockSignals(true);
            autoStepDurationCheckBox->setChecked(false);
            autoStepDurationCheckBox->blockSignals(false);
        }
        bool isChecked = true;
        if (vp->hasMetaData(s_stepEnabledKey)) {
            isChecked = vp->getMetaData(s_stepEnabledKey).toBool();
        }

        QString itemName = QString("step %1 (%2)")
                                   .arg(QString::number(i + 1), vp->getName());
        QListWidgetItem* item =
                new QListWidgetItem(itemName, stepSelectionList);
        item->setFlags(item->flags() |
                       Qt::ItemIsUserCheckable);  // set checkable flag
        item->setCheckState(isChecked
                                    ? Qt::Checked
                                    : Qt::Unchecked);  // initialize check state
        stepSelectionList->addItem(item);

        m_videoSteps[i].viewport = vp;
        m_videoSteps[i].viewportParams = vp->getParameters();
        m_videoSteps[i].duration_sec = duration_sec;
        m_videoSteps[i].indexInOriginalTrajectory = static_cast<int>(i);

        // compute the real camera center
        ccGLMatrixd viewMat = vp->getParameters().computeViewMatrix();
        m_videoSteps[i].cameraCenter =
                viewMat.inverse().getTranslationAsVec3D();
    }

    for (std::size_t i = 0; i < meshes.size(); ++i) {
        m_mesh_list[i] = meshes[i];
    }

    // manually trigger some actions if necessary

    updateCameraTrajectory();  // also takes care of the smooth version!

    connect(stepSelectionList, &QListWidget::currentRowChanged, this,
            &qAnimationDlg::onCurrentStepChanged);
    connect(stepSelectionList, &QListWidget::itemChanged, this,
            &qAnimationDlg::onItemChanged);

    stepSelectionList->setCurrentRow(0);  // select the first one by default
    onCurrentStepChanged(getCurrentStepIndex());

    return true;
}

void qAnimationDlg::onReject() {}

void qAnimationDlg::onAccept() {
    assert(stepSelectionList->count() >= m_videoSteps.size());
    for (size_t i = 0; i < m_videoSteps.size(); ++i) {
        cc2DViewportObject* vp = m_videoSteps[i].viewport;

        // save the step duration as meta data
        if (!autoStepDurationCheckBox->isChecked()) {
            vp->setMetaData(s_stepDurationKey, m_videoSteps[i].duration_sec);
        }
        // save whether the step is enabled or not as meta data
        vp->setMetaData(
                s_stepEnabledKey,
                (stepSelectionList->item(static_cast<int>(i))->checkState() ==
                 Qt::Checked));
    }

    // store settings
    {
        QSettings settings;
        settings.beginGroup("qAnimation");
        settings.setValue("previewFromSelected",
                          previewFromSelectedCheckBox->isChecked());
        settings.setValue("updateTextures",
                          updateTexturesCheckBox->isChecked());
        settings.setValue("loop", loopCheckBox->isChecked());
        settings.setValue("frameRate", fpsSpinBox->value());
        settings.setValue("renderingMode",
                          renderingModeComboBox->currentIndex());
        settings.setValue("superRes", superResolutionSpinBox->value());
        settings.setValue("bitRate", bitrateSpinBox->value());
        settings.setValue("autoStepDuration",
                          autoStepDurationCheckBox->isChecked());
        settings.setValue("smoothTrajectory",
                          smoothTrajectoryGroupBox->isChecked());
        settings.setValue("smoothRatio", smoothRatioDoubleSpinBox->value());
        settings.setValue("outputFormat",
                          outputFormatComboBox->currentData().toString());

        settings.endGroup();
    }
}

bool qAnimationDlg::updateCameraTrajectory() {
    if (m_videoSteps.empty()) return false;
    m_smoothVideoSteps.clear();
    for (Step& step : m_videoSteps) {
        step.indexInSmoothTrajectory = -1;
        step.length = 0;
    }

    if (m_videoSteps.size() < 2) {
        CVLog::Warning("Not enough animation steps");
        updateTotalDuration();
        return false;
    }

    // update the segment lengths
    size_t vp1Index = 0, vp2Index = 0;
    while (getNextSegment(vp1Index, vp2Index)) {
        assert(vp1Index < stepSelectionList->count());
        Step& step1 = m_videoSteps[vp1Index];

        step1.length = (m_videoSteps[vp2Index].cameraCenter -
                        m_videoSteps[vp1Index].cameraCenter)
                               .norm();

        if (vp2Index < vp1Index) {
            // loop mode
            break;
        }
        vp1Index = vp2Index;
    }

    bool result = true;
    if (smoothTrajectoryGroupBox->isChecked()) {
        result = updateSmoothCameraTrajectory();
    }

    if (autoStepDurationCheckBox->isChecked()) {
        onAutoStepsDurationToggled(true);
    } else {
        updateTotalDuration();
    }

    return result;
}

void qAnimationDlg::updateSmoothTrajectoryDurations() {
    if (m_videoSteps.empty()) return;
    bool smoothMode = smoothModeEnabled();
    if (!smoothMode) {
        return;
    }

    size_t vp1Index = 0, vp2Index = 0;
    while (getNextSegment(vp1Index, vp2Index)) {
        assert(vp1Index < stepSelectionList->count());
        Step& step1 = m_videoSteps[vp1Index];
        const Step& step2 = m_videoSteps[vp2Index];

        int i1Smooth = step1.indexInSmoothTrajectory;
        int i2Smooth = step2.indexInSmoothTrajectory;
        if (i1Smooth < 0 || i2Smooth < 0) {
            assert(false);
            continue;
        }
        if (i2Smooth < i1Smooth) {
            // loop mode
            i2Smooth += static_cast<int>(m_smoothVideoSteps.size());
        }

        double length = 0;
        for (int i = i1Smooth; i < i2Smooth; ++i) {
            const Step& s = m_smoothVideoSteps[static_cast<size_t>(i) %
                                               m_smoothVideoSteps.size()];
            length += s.length;
        }

        if (cloudViewer::GreaterThanEpsilon(length)) {
            for (int i = i1Smooth; i < i2Smooth; ++i) {
                Step& s = m_smoothVideoSteps[static_cast<size_t>(i) %
                                             m_smoothVideoSteps.size()];
                s.duration_sec = step1.duration_sec * (s.length / length);
            }
        }

        if (vp2Index < vp1Index) {
            // loop case
            break;
        }
        vp1Index = vp2Index;
    }
}

bool qAnimationDlg::smoothTrajectory(double ratio, unsigned iterationCount) {
    if (m_videoSteps.empty()) return false;

    if (iterationCount == 0) {
        assert(false);
        CVLog::Warning("[smoothTrajectory] Invalid input (iteration count)");
        return false;
    }

    if (ratio < 0.05 || ratio > 0.45) {
        assert(false);
        CVLog::Warning("[smoothTrajectory] invalid input (ratio)");
        return false;
    }

    size_t enabledStepCount = countEnabledSteps();
    if (enabledStepCount < 3) {
        CVLog::Warning("[smoothTrajectory] not enough segments");
        return false;
    }

    try {
        Trajectory previousTrajectory;
        if (!getCompressedTrajectory(previousTrajectory)) {
            CVLog::Error("Not enough memory");
            return false;
        }
        const Trajectory* currentIterationTrajectory = &previousTrajectory;

        bool openPoly = !loopCheckBox->isChecked();

        for (unsigned it = 0; it < iterationCount; ++it) {
            // reserve memory for the new steps
            size_t vertCount = currentIterationTrajectory->size();
            size_t segmentCount = (openPoly ? vertCount - 1 : vertCount);

            Trajectory newTrajectory;
            newTrajectory.reserve(segmentCount * 2);

            if (openPoly) {
                // we always keep the first step
                newTrajectory.push_back(currentIterationTrajectory->front());
            }

            for (size_t i = 0; i < segmentCount; ++i) {
                size_t iP = i;
                size_t iQ = ((iP + 1) % vertCount);

                const Step& sP = currentIterationTrajectory->at(iP);
                const Step& sQ = currentIterationTrajectory->at(iQ);

                ViewInterpolate interpolator(sP.viewportParams,
                                             sQ.viewportParams);

                if (!openPoly || i != 0) {
                    Step interpolatedStep;
                    interpolatedStep.cameraCenter =
                            (PC_ONE - ratio) * sP.cameraCenter +
                            ratio * sQ.cameraCenter;
                    interpolatedStep.duration_sec = sP.duration_sec * ratio;
                    interpolator.interpolate(interpolatedStep.viewportParams,
                                             ratio);
                    interpolatedStep.indexInOriginalTrajectory =
                            (it == 0 ? -1 : sP.indexInOriginalTrajectory);
                    newTrajectory.push_back(interpolatedStep);
                }

                if (!openPoly || i + 1 != segmentCount) {
                    Step interpolatedStep;
                    interpolatedStep.cameraCenter =
                            ratio * sP.cameraCenter +
                            (PC_ONE - ratio) * sQ.cameraCenter;
                    interpolatedStep.duration_sec =
                            sP.duration_sec * (PC_ONE - ratio);
                    interpolator.interpolate(interpolatedStep.viewportParams,
                                             PC_ONE - ratio);
                    interpolatedStep.indexInOriginalTrajectory =
                            (it == 0 ? sQ.indexInOriginalTrajectory : -1);
                    newTrajectory.push_back(interpolatedStep);
                }
            }

            if (openPoly) {
                // we always keep the last vertex
                newTrajectory.push_back(currentIterationTrajectory->back());
            }

            // last iteration?
            if (it + 1 == iterationCount) {
                m_smoothVideoSteps = newTrajectory;
            } else {
                previousTrajectory = newTrajectory;
                currentIterationTrajectory = &previousTrajectory;
            }
        }

        // update the segment lengths
        size_t smoothSegmentCount = m_smoothVideoSteps.size();
        if (openPoly) {
            --smoothSegmentCount;
        }
        for (size_t i = 0; i < smoothSegmentCount; ++i) {
            CCVector3d d =
                    m_smoothVideoSteps[(i + 1) % m_smoothVideoSteps.size()]
                            .cameraCenter -
                    m_smoothVideoSteps[i].cameraCenter;
            m_smoothVideoSteps[i].length = d.norm();
        }

        // update the loop-back indexes
        for (size_t i = 0; i < m_smoothVideoSteps.size(); ++i) {
            const Step& s = m_smoothVideoSteps[i];
            if (s.indexInOriginalTrajectory != -1) {
                assert(m_videoSteps[s.indexInOriginalTrajectory]
                               .indexInSmoothTrajectory < 0);
                m_videoSteps[s.indexInOriginalTrajectory]
                        .indexInSmoothTrajectory = static_cast<int>(i);
            }
        }

        // update the durations
        updateSmoothTrajectoryDurations();
    } catch (const std::bad_alloc&) {
        CVLog::Warning("[smoothTrajectory] not enough memory");
        m_smoothVideoSteps.clear();
        return false;
    }

    return true;
}

bool qAnimationDlg::updateSmoothCameraTrajectory() {
    // reset existing data
    m_smoothVideoSteps.clear();
    for (Step& step : m_videoSteps) {
        step.indexInSmoothTrajectory = -1;
    }

    if (!smoothTrajectoryGroupBox->isChecked()) {
        return true;
    }

    if (countEnabledSteps() < 3) {
        // nothing we can do for now
        return true;
    }

    const unsigned chaikinIterationCount = 5;
    const double chaikinRatio = smoothRatioDoubleSpinBox->value();

    if (!smoothTrajectory(chaikinRatio, chaikinIterationCount)) {
        CVLog::Error("Failed to generate the smooth trajectory");
        smoothTrajectoryGroupBox->blockSignals(true);
        smoothTrajectoryGroupBox->setChecked(false);
        smoothTrajectoryGroupBox->blockSignals(false);
        return false;
    }
    assert(!m_smoothVideoSteps.empty());

    return true;
}

void qAnimationDlg::onAutoStepsDurationToggled(bool state) {
    if (!state) {
        //'auto step duration' mode deactivated: nothing to do
        return;
    }

    if (m_videoSteps.empty()) return;

    Trajectory* referenceTrajectory = nullptr;
    bool smoothMode = smoothModeEnabled();
    if (smoothMode) {
        referenceTrajectory = &m_smoothVideoSteps;
    } else {
        referenceTrajectory = &m_videoSteps;
    }
    assert(referenceTrajectory);

    // total length
    double referenceLength = 0;
    for (const Step& s : *referenceTrajectory) {
        referenceLength += s.length;
    }

    double totalTime = 0.0;
    double totalTimeSmooth = 0.0;

    // now process each segment
    size_t vp1Index = 0, vp2Index = 0;
    while (getNextSegment(vp1Index, vp2Index)) {
        assert(vp1Index < stepSelectionList->count());
        Step& step1 = m_videoSteps[vp1Index];
        const Step& step2 = m_videoSteps[vp2Index];

        double length = 0;

        int i1Smooth = -1, i2Smooth = -1;
        if (smoothMode) {
            i1Smooth = step1.indexInSmoothTrajectory;
            i2Smooth = step2.indexInSmoothTrajectory;
            if (i1Smooth < 0 || i2Smooth < 0) {
                assert(false);
                continue;
            }
            if (i2Smooth < i1Smooth) {
                // loop mode
                i2Smooth += static_cast<int>(m_smoothVideoSteps.size());
            }

            for (int i = i1Smooth; i < i2Smooth; ++i) {
                const Step& s = m_smoothVideoSteps[static_cast<size_t>(i) %
                                                   m_smoothVideoSteps.size()];
                length += s.length;
            }
        } else {
            length = step1.length;
        }

        if (cloudViewer::LessThanEpsilon(length)) {
            step1.duration_sec = 0.0;
        } else {
            step1.duration_sec = totalTimeDoubleSpinBox->value() *
                                 (length / referenceLength);

            // update the segments time as well
            if (smoothMode) {
                for (int i = i1Smooth; i < i2Smooth; ++i) {
                    Step& s = m_smoothVideoSteps[static_cast<size_t>(i) %
                                                 m_smoothVideoSteps.size()];
                    s.duration_sec = step1.duration_sec * (s.length / length);
                    totalTimeSmooth += s.duration_sec;
                }
            }
        }

        totalTime += step1.duration_sec;

        if (vp2Index < vp1Index) {
            // loop case
            break;
        }
        vp1Index = vp2Index;
    }

    CVLog::PrintDebug(QString("Total time = %1 / Smooth time = %2")
                              .arg(totalTime)
                              .arg(totalTimeSmooth));

    onCurrentStepChanged(getCurrentStepIndex());
}

void qAnimationDlg::onSmoothTrajectoryToggled(bool state) {
    if (state) {
        updateSmoothCameraTrajectory();
    }

    onAutoStepsDurationToggled(autoStepDurationCheckBox->isChecked());
}

void qAnimationDlg::onSmoothRatioChanged(double ratio) {
    if (smoothTrajectoryGroupBox->isChecked()) {
        onSmoothTrajectoryToggled(true);
    }
}

ccPolyline* qAnimationDlg::getTrajectory() {
    // TODO
    return nullptr;
}

bool qAnimationDlg::exportTrajectoryOnExit() {
    return exportTrajectoryCheckBox->isChecked();
}

double qAnimationDlg::computeTotalTime() {
    double totalDuration_sec = 0;
    size_t vp1Index = 0, vp2Index = 0;
    while (getNextSegment(vp1Index, vp2Index)) {
        assert(vp1Index < stepSelectionList->count());
        totalDuration_sec += m_videoSteps[vp1Index].duration_sec;
        if (vp2Index < vp1Index) {
            // loop case
            break;
        }
        vp1Index = vp2Index;
    }

    return totalDuration_sec;
}

int qAnimationDlg::getCurrentStepIndex() {
    return stepSelectionList->currentRow();
}

void qAnimationDlg::applyViewport(
        const ecvViewportParameters& viewportParameters) {
    if (m_view3d) {
        ecvDisplayTools::SetViewportParameters(viewportParameters);
        ecvDisplayTools::UpdateScreen();
    }
}

void qAnimationDlg::onFPSChanged(int fps) {
    // nothing to do
}

void qAnimationDlg::onTotalTimeChanged(double newTime_sec) {
    if (m_videoSteps.empty()) return;
    if (autoStepDurationCheckBox->isChecked()) {
        onAutoStepsDurationToggled(true);
        return;
    }

    //'manual' mode
    double previousTime_sec = computeTotalTime();
    if (previousTime_sec != newTime_sec) {
        assert(previousTime_sec != 0);
        double scale = newTime_sec / previousTime_sec;

        bool smoothMode = smoothModeEnabled();
        double totalTime = 0.0;
        double totalTimeSmooth = 0.0;

        size_t vp1Index = 0, vp2Index = 0;
        while (getNextSegment(vp1Index, vp2Index)) {
            assert(vp1Index < stepSelectionList->count());
            Step& step1 = m_videoSteps[vp1Index];
            const Step& step2 = m_videoSteps[vp2Index];

            step1.duration_sec *= scale;
            totalTime += step1.duration_sec;

            if (smoothMode) {
                int i1Smooth = step1.indexInSmoothTrajectory;
                int i2Smooth = step2.indexInSmoothTrajectory;
                if (i1Smooth < 0 || i2Smooth < 0) {
                    assert(false);
                    continue;
                }
                if (i2Smooth < i1Smooth) {
                    // loop mode
                    i2Smooth += static_cast<int>(m_smoothVideoSteps.size());
                }

                double length = 0;
                for (int i = i1Smooth; i < i2Smooth; ++i) {
                    const Step& s =
                            m_smoothVideoSteps[static_cast<size_t>(i) %
                                               m_smoothVideoSteps.size()];
                    length += s.length;
                }

                if (cloudViewer::LessThanEpsilon(length)) {
                    // divide equally over all the segments
                    size_t count = static_cast<size_t>(i2Smooth - i1Smooth);
                    for (int i = i1Smooth; i < i2Smooth; ++i) {
                        Step& s = m_smoothVideoSteps[static_cast<size_t>(i) %
                                                     m_smoothVideoSteps.size()];
                        s.duration_sec = step1.duration_sec / count;
                        totalTimeSmooth += s.duration_sec;
                    }
                } else {
                    // divide over all the segments based on their respective
                    // length
                    for (int i = i1Smooth; i < i2Smooth; ++i) {
                        Step& s = m_smoothVideoSteps[static_cast<size_t>(i) %
                                                     m_smoothVideoSteps.size()];
                        s.duration_sec = step1.duration_sec * s.length / length;
                        totalTimeSmooth += s.duration_sec;
                    }
                }
            }

            if (vp2Index < vp1Index) {
                // loop case
                break;
            }
            vp1Index = vp2Index;
        }

        CVLog::PrintDebug(QString("Total time = %1 / Smooth time = %2")
                                  .arg(totalTime)
                                  .arg(totalTimeSmooth));

        // update current step
        updateCurrentStepDuration();
    }
}

void qAnimationDlg::onStepTimeChanged(double time_sec) {
    if (m_videoSteps.empty()) return;
    int currentStepIndex = getCurrentStepIndex();
    if (currentStepIndex >= 0) {
        m_videoSteps[getCurrentStepIndex()].duration_sec = time_sec;
    }

    // update total duration
    updateTotalDuration();
    // update current step
    updateCurrentStepDuration();
    // we have to update the whole smooth trajectory duration as well
    updateSmoothTrajectoryDurations();
}

void qAnimationDlg::onBrowseButtonClicked() {
#ifdef QFFMPEG_SUPPORT
    QString filename = QFileDialog::getSaveFileName(
            this, tr("Output animation file"), outputFileLineEdit->text());
#else
    QString filename = QFileDialog::getExistingDirectory(
            this, tr("Open Directory"), outputFileLineEdit->text(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
#endif

    if (filename.isEmpty()) {
        // cancelled by user
        return;
    }

    outputFileLineEdit->setText(filename);
}

void qAnimationDlg::onBrowseTexturesButtonClicked() {
    QString texturePath = QFileDialog::getExistingDirectory(
            this, tr("Input textures Directory"),
            inputTexturesPathLineEdit->text(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (texturePath.isEmpty()) {
        // cancelled by user
        return;
    }

    inputTexturesPathLineEdit->setText(texturePath);
    textureNumLabel->setText(
            QString("Textures num: %1").arg(getImageList(texturePath).size()));
}

size_t qAnimationDlg::countEnabledSteps() const {
    if (m_videoSteps.empty()) return 0;
    assert(stepSelectionList->count() == static_cast<int>(m_videoSteps.size()));

    size_t count = 0;
    for (int i = 0; i < stepSelectionList->count(); ++i) {
        if (stepSelectionList->item(i)->checkState() == Qt::Checked) ++count;
    }

    return count;
}

bool qAnimationDlg::getNextSegment(size_t& vp1Index, size_t& vp2Index) const {
    if (m_videoSteps.empty()) return false;
    assert(stepSelectionList->count() == static_cast<int>(m_videoSteps.size()));
    if (vp1Index >= m_videoSteps.size()) {
        assert(false);
        return false;
    }

    size_t inputVP1Index = vp1Index;
    while (stepSelectionList->item(static_cast<int>(vp1Index))->checkState() ==
           Qt::Unchecked) {
        ++vp1Index;
        if (vp1Index == m_videoSteps.size()) {
            if (loopCheckBox->isChecked()) {
                vp1Index = 0;
            } else {
                // no more valid start (vp1)
                return false;
            }
        }
        if (vp1Index == inputVP1Index) {
            return false;
        }
    }

    // look for the next enabled viewport
    for (vp2Index = vp1Index + 1; vp2Index <= m_videoSteps.size(); ++vp2Index) {
        if (vp1Index == vp2Index) {
            return false;
        }

        if (vp2Index == m_videoSteps.size()) {
            if (loopCheckBox->isChecked()) {
                vp2Index = 0;
            } else {
                // stop
                break;
            }
        }

        if (stepSelectionList->item(static_cast<int>(vp2Index))->checkState() ==
            Qt::Checked) {
            // we have found a valid couple (vp1, vp2)
            return true;
        }
    }

    // no more valid stop (vp2)
    return false;
}

int qAnimationDlg::countFrames(size_t startIndex /*=0*/) {
    // reset the interpolators and count the total number of frames
    int totalFrameCount = 0;
    {
        double fps = fpsSpinBox->value();

        size_t vp1Index = startIndex;
        size_t vp2Index = vp1Index + 1;

        while (getNextSegment(vp1Index, vp2Index)) {
            const Step& currentStep = m_videoSteps[vp1Index];
            int frameCount = static_cast<int>(fps * currentStep.duration_sec);
            totalFrameCount += frameCount;

            // take care of the 'loop' case
            if (vp2Index < vp1Index) {
                assert(loopCheckBox->isChecked());
                break;
            }
            vp1Index = vp2Index;
        }
    }

    return totalFrameCount;
}

bool qAnimationDlg::getCompressedTrajectory(
        Trajectory& compressedTrajectory) const {
    if (m_videoSteps.empty()) return true;

    compressedTrajectory.clear();

    size_t enabledStepCount = countEnabledSteps();
    try {
        compressedTrajectory.reserve(enabledStepCount);
    } catch (const std::bad_alloc&) {
        return false;
    }

    assert(stepSelectionList->count() == static_cast<int>(m_videoSteps.size()));
    for (size_t i = 0; i < m_videoSteps.size(); ++i) {
        if (stepSelectionList->item(static_cast<int>(i))->checkState() ==
            Qt::Checked) {
            compressedTrajectory.push_back(m_videoSteps[i]);
        }
    }

    return true;
}

void qAnimationDlg::preview() {
    setEnabled(false);

    bool openPoly = !loopCheckBox->isChecked();
    int fps = fpsSpinBox->value();
    double timeStep = 1.0 / fps;
    // theoretical waiting time per frame
    qint64 delay_ms = static_cast<qint64>(1000 / fps);

    QString inputTexturesPath = inputTexturesPathLineEdit->text();
    QStringList texture_files = getImageList(inputTexturesPath);

    bool update_textures = updateTextures() && !texture_files.empty();
    if (m_videoSteps.size() < 2 && update_textures) {  // only update textures
        // show progress dialog
        int total_count = texture_files.size();
        QProgressDialog progressDialog(QString("Frames: %1").arg(total_count),
                                       "Cancel", 0, total_count, this);
        progressDialog.setWindowTitle("Preview");
        progressDialog.show();
        progressDialog.setModal(true);
        progressDialog.setAutoClose(false);
        QApplication::processEvents();

        textureAnimationPreview(texture_files, progressDialog);
    } else if (m_videoSteps.size() >= 2) {  //  preview viewports update
        size_t vp1Index = previewFromSelectedCheckBox->isChecked()
                                  ? static_cast<size_t>(getCurrentStepIndex())
                                  : 0;
        Trajectory compressedTrajectory;
        const Trajectory* trajectory = nullptr;

        bool smoothMode = smoothModeEnabled();
        if (smoothMode) {
            trajectory = &m_smoothVideoSteps;
            assert(m_videoSteps[vp1Index].indexInSmoothTrajectory >= 0);
            vp1Index = static_cast<size_t>(
                    m_videoSteps[vp1Index].indexInSmoothTrajectory);
        } else {
            if (!getCompressedTrajectory(compressedTrajectory)) {
                CVLog::Error("Not enough memory");
                return;
            }
            trajectory = &compressedTrajectory;
        }
        assert(trajectory);

        size_t segmentCount = trajectory->size();
        if (openPoly && segmentCount != 0) {
            --segmentCount;
        }

        if (vp1Index >= segmentCount) {
            // can't start from the specified index
            vp1Index = 0;
        }

        double totalTime = 0;
        double startTime = 0;
        for (size_t i = 0; i < segmentCount; ++i) {
            totalTime += trajectory->at(i).duration_sec;
            if (i < vp1Index) {
                startTime += trajectory->at(i).duration_sec;
            }
        }

        // count the total number of frames
        double remainingTime = (openPoly ? totalTime - startTime : totalTime);
        int frameCount = static_cast<int>(fps * remainingTime);

        int total_count = frameCount;
        if (frameCount == 0 && update_textures) {
            total_count = texture_files.size();
        }

        // show progress dialog
        QProgressDialog progressDialog(QString("Frames: %1").arg(total_count),
                                       "Cancel", 0, total_count, this);
        progressDialog.setWindowTitle("Preview");
        progressDialog.show();
        progressDialog.setModal(true);
        progressDialog.setAutoClose(false);
        QApplication::processEvents();

        assert(stepSelectionList->count() >= m_videoSteps.size());

        double currentTime = startTime;
        double currentStepStartTime = startTime;

        if (frameCount == 0 && update_textures) {
            textureAnimationPreview(texture_files, progressDialog);
        } else if (frameCount > 0) {
            // we'll take the rendering time into account!
            QElapsedTimer timer;
            timer.start();
            for (int frameIndex = 0; frameIndex < frameCount;) {
                size_t vp2Index = vp1Index + 1;
                if (vp2Index == trajectory->size()) {
                    assert(!openPoly);
                    vp2Index = 0;
                }

                const Step& step1 = trajectory->at(vp1Index);
                double deltaTime = currentTime - currentStepStartTime;
                if (deltaTime <= step1.duration_sec) {
                    const Step& step2 = trajectory->at(vp2Index);
                    ViewInterpolate interpolator(step1.viewportParams,
                                                 step2.viewportParams);

                    ecvViewportParameters currentViewport;
                    interpolator.interpolate(currentViewport,
                                             deltaTime / step1.duration_sec);

                    timer.restart();
                    if (update_textures) {
                        QString texture_file =
                                texture_files[frameIndex %
                                              texture_files.size()];
                        if (QFileInfo::exists(texture_file)) {
                            for (auto& mesh : m_mesh_list) {
                                if (!mesh->updateTextures(CVTools::FromQString(
                                            texture_file))) {
                                    CVLog::Warning(
                                            "Update Textures failed, please "
                                            "toggle shown material first!");
                                }
                            }
                        } else {
                            CVLog::Warning(QString("Ignoring not existing "
                                                   "texture image: %1")
                                                   .arg(texture_file));
                        }
                    }

                    applyViewport(currentViewport);

                    // next frame
                    currentTime += timeStep;
                    ++frameIndex;

                    progressDialog.setValue(frameIndex);
                    QApplication::processEvents();
                    if (progressDialog.wasCanceled()) {
                        break;
                    }

                    qint64 dt_ms = timer.elapsed();

                    // remaining time
                    if (dt_ms < delay_ms) {
                        int wait_ms = static_cast<int>(delay_ms - dt_ms);
#if defined(CV_WINDOWS)
                        ::Sleep(wait_ms);
#else
                        usleep(wait_ms * 1000);
#endif
                    }
                } else {
                    // we'll try the next step
                    ++vp1Index;

                    if (vp1Index == segmentCount) {
                        if (openPoly) {
                            break;
                        }

                        // else restart from 0
                        vp1Index = 0;
                        currentStepStartTime = 0.0;
                        currentTime = std::max(0.0, currentTime - totalTime);
                    } else {
                        currentStepStartTime += step1.duration_sec;
                    }
                }
            }
        } else {
            CVLog::Warning(
                    "Please select update tetures and set textures path or "
                    "select at least two viewports!");
        }
    } else {
        CVLog::Warning(
                "Please select update tetures and set textures path or "
                "select at least two viewports!");
    }

    // reset view
    onCurrentStepChanged(getCurrentStepIndex());

    setEnabled(true);
}

void qAnimationDlg::textureAnimationPreview(const QStringList& texture_files,
                                            QProgressDialog& progressDialog) {
    bool openPoly = !loopCheckBox->isChecked();
    int fps = fpsSpinBox->value();
    double timeStep = 1.0 / fps;
    // theoretical waiting time per frame
    qint64 delay_ms = static_cast<qint64>(1000 / fps);
    std::size_t total_count = texture_files.size();
    // we'll take the rendering time into account!
    QElapsedTimer timer;
    timer.start();
    for (std::size_t frameIndex = 0; frameIndex < texture_files.size();) {
        timer.restart();
        QString texture_file = texture_files[frameIndex];
        if (QFileInfo::exists(texture_file)) {
            for (auto& mesh : m_mesh_list) {
                if (!mesh->updateTextures(CVTools::FromQString(texture_file))) {
                    CVLog::Warning(
                            "Update Textures failed, please "
                            "toggle shown material first!");
                }
            }
        } else {
            CVLog::Warning(QString("Ignoring not existing texture image: %1")
                                   .arg(texture_file));
        }
        ecvDisplayTools::UpdateScreen();

        // next frame
        ++frameIndex;

        progressDialog.setValue(frameIndex);
        QApplication::processEvents();
        if (progressDialog.wasCanceled()) {
            break;
        }

        qint64 dt_ms = timer.elapsed();

        // remaining time
        if (dt_ms < delay_ms) {
            int wait_ms = static_cast<int>(delay_ms - dt_ms);
#if defined(CV_WINDOWS)
            ::Sleep(wait_ms);
#else
            usleep(wait_ms * 1000);
#endif
        }

        if (!openPoly && frameIndex == total_count) {
            frameIndex = 0;
        }
    }
}

bool qAnimationDlg::textureAnimationRender(
        const QStringList& texture_files,
        QProgressDialog& progressDialog,
        bool asSeparateFrames
#ifdef QFFMPEG_SUPPORT
        ,
        QScopedPointer<QVideoEncoder>& encoder
#endif
) {

    bool success = true;
    bool openPoly = !loopCheckBox->isChecked();
    int fps = fpsSpinBox->value();
    double timeStep = 1.0 / fps;
    // theoretical waiting time per frame
    qint64 delay_ms = static_cast<qint64>(1000 / fps);
    std::size_t total_count = texture_files.size();

    // super resolution
    int superRes = superResolutionSpinBox->value();
    const int SUPER_RESOLUTION = 0;
    const int ZOOM = 1;
    int renderingMode = renderingModeComboBox->currentIndex();
    assert(renderingMode == SUPER_RESOLUTION || renderingMode == ZOOM);

    QString outputFilename = outputFileLineEdit->text();
    QDir outputDir(QFileInfo(outputFilename).absolutePath());

    // we'll take the rendering time into account!
    QElapsedTimer timer;
    timer.start();
    for (std::size_t frameIndex = 0; frameIndex < total_count;) {
        if (QFileInfo::exists(texture_files[frameIndex])) {
            for (auto& mesh : m_mesh_list) {
                if (!mesh->updateTextures(
                            CVTools::FromQString(texture_files[frameIndex]))) {
                    CVLog::Warning(
                            "Update Textures failed, please "
                            "toggle shown material first!");
                };
            }
            ecvDisplayTools::UpdateScreen();
        }

        // render to image
        QImage image = ecvDisplayTools::RenderToImage(superRes, false, true, 0);

        if (image.isNull()) {
            QMessageBox::critical(this, "Error", "Failed to grab the screen!");
            success = false;
            break;
        }

        if (renderingMode == SUPER_RESOLUTION && superRes > 1) {
            image = image.scaled(
                    image.width() / superRes, image.height() / superRes,
                    Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        }

        if (asSeparateFrames) {
            QString filename =
                    QString("frame_%1.png").arg(frameIndex, 6, 10, QChar('0'));
            QString fullPath = outputDir.filePath(filename);
            if (!image.save(fullPath)) {
                QMessageBox::critical(this, "Error",
                                      QString("Failed to save frame #%1")
                                              .arg(frameIndex + 1));
                success = false;
                break;
            }
        } else {
#ifdef QFFMPEG_SUPPORT
            QString errorString;
            if (!encoder->encodeImage(image, frameIndex, &errorString)) {
                QMessageBox::critical(this, "Error",
                                      QString("Failed to encode frame #%1: %2")
                                              .arg(frameIndex + 1)
                                              .arg(errorString));
                success = false;
                break;
            }
#endif
        }

        // next frame
        ++frameIndex;

        progressDialog.setValue(frameIndex);
        QApplication::processEvents();
        if (progressDialog.wasCanceled()) {
            QMessageBox::warning(this, "Warning",
                                 QString("Process has been cancelled"));
            success = false;
            break;
        }
    }

    return success;
}

void qAnimationDlg::render(bool asSeparateFrames) {
    if (!m_view3d) {
        assert(false);
        return;
    }

    QString outputFilename = outputFileLineEdit->text();
    QString inputTexturesPath = inputTexturesPathLineEdit->text();
    QStringList texture_files = getImageList(inputTexturesPath);

    bool update_textures = updateTextures() && !texture_files.empty();

    // save to persistent settings
    {
        QSettings settings;
        settings.beginGroup("qAnimation");
        settings.setValue("filename", outputFilename);
        settings.setValue("texturesPath", inputTexturesPath);
        settings.endGroup();
    }

    if (!update_textures && m_videoSteps.size() < 2) {
        CVLog::Warning(
                "Please select update tetures and set textures path or select "
                "at least two viewports!");
        return;
    }

    setEnabled(false);

    Trajectory compressedTrajectory;
    const Trajectory* trajectory = nullptr;

    bool smoothMode = smoothModeEnabled();
    if (smoothMode) {
        trajectory = &m_smoothVideoSteps;
    } else {
        if (!getCompressedTrajectory(compressedTrajectory)) {
            CVLog::Error("Not enough memory");
            return;
        }
        trajectory = &compressedTrajectory;
    }
    assert(trajectory);

    bool openPoly = !loopCheckBox->isChecked();
    size_t segmentCount = trajectory->size();
    if (openPoly && segmentCount != 0) {
        --segmentCount;
    }

    double totalTime = 0;
    for (size_t i = 0; i < segmentCount; ++i) {
        totalTime += trajectory->at(i).duration_sec;
    }

    // count the total number of frames
    int fps = fpsSpinBox->value();
    int frameCount = static_cast<int>(fps * totalTime);
    int total_count = frameCount;
    if (total_count == 0 && update_textures) {
        total_count = texture_files.size();
    }

    // super resolution
    int superRes = superResolutionSpinBox->value();
    const int SUPER_RESOLUTION = 0;
    const int ZOOM = 1;
    int renderingMode = renderingModeComboBox->currentIndex();
    assert(renderingMode == SUPER_RESOLUTION || renderingMode == ZOOM);

    // show progress dialog
    QProgressDialog progressDialog(QString("Frames: %1").arg(total_count),
                                   "Cancel", 0, total_count, this);
    progressDialog.setWindowTitle("Render");
    progressDialog.show();
    QApplication::processEvents();

#ifdef QFFMPEG_SUPPORT
    QScopedPointer<QVideoEncoder> encoder(nullptr);
    QSize originalViewSize;
    if (!asSeparateFrames) {
        // get original viewport size
        originalViewSize = ecvDisplayTools::GetScreenSize();

        // hack: as the encoder requires that the video dimensions are multiples
        // of 8, we resize the window a little bit...
        {
            // find the nearest multiples of 8
            QSize customSize = originalViewSize;
            if (originalViewSize.width() % 8 || originalViewSize.height() % 8) {
                if (originalViewSize.width() % 8)
                    customSize.setWidth((originalViewSize.width() / 8 + 1) * 8);
                if (originalViewSize.height() % 8)
                    customSize.setHeight((originalViewSize.height() / 8 + 1) *
                                         8);
                m_view3d->resize(customSize);
                QApplication::processEvents();
            }
        }

        int bitrate = bitrateSpinBox->value() * 1024;
        int gop = fps;
        int animScale = 1;
        if (renderingMode == ZOOM) {
            animScale = superRes;
        }

        encoder.reset(new QVideoEncoder(
                outputFilename, ecvDisplayTools::GlWidth() * animScale,
                ecvDisplayTools::GlHeight() * animScale, bitrate, gop,
                static_cast<unsigned>(fpsSpinBox->value())));
        QStringList errors;
        QString outputFormat = outputFormatComboBox->currentData().toString();
        bool success = encoder->open(outputFormat, errors);
        for (const QString& e : errors) {
            CVLog::Warning(e);
        }

        if (!success) {
            QMessageBox::critical(
                    this, "Error",
                    QString("Failed to open file for output: %1")
                            .arg(errors.back()));  // display the last error
                                                   // message
            setEnabled(true);
            return;
        }
    }
#else
    if (!asSeparateFrames) {
        QMessageBox::critical(
                this, "Error",
                QString("Animation mode is not supported (no FFMPEG support)"));
        return;
    }
#endif

    QDir outputDir(QFileInfo(outputFilename).absolutePath());

    bool success = true;

    if (frameCount == 0 && update_textures) {
        success = textureAnimationRender(texture_files, progressDialog,
                                         asSeparateFrames
#ifdef QFFMPEG_SUPPORT
                                         ,
                                         encoder
#endif
        );
    } else {
        double currentTime = 0.0;
        double currentStepStartTime = 0.0;
        double timeStep = 1.0 / fps;
        size_t vp1Index = 0;
        for (int frameIndex = 0; frameIndex < frameCount;) {
            size_t vp2Index = vp1Index + 1;
            if (vp2Index == trajectory->size()) {
                assert(!openPoly);
                vp2Index = 0;
            }

            const Step& step1 = trajectory->at(vp1Index);
            double deltaTime = currentTime - currentStepStartTime;
            if (deltaTime <= step1.duration_sec) {
                const Step& step2 = trajectory->at(vp2Index);
                ViewInterpolate interpolator(step1.viewportParams,
                                             step2.viewportParams);

                ecvViewportParameters currentViewport;
                interpolator.interpolate(currentViewport,
                                         deltaTime / step1.duration_sec);

                // update textures
                if (update_textures) {
                    QString texture_file =
                            texture_files[frameIndex % texture_files.size()];
                    if (QFileInfo::exists(texture_file)) {
                        for (auto& mesh : m_mesh_list) {
                            if (!mesh->updateTextures(
                                        CVTools::FromQString(texture_file))) {
                                CVLog::Warning(
                                        "Update Textures failed, please "
                                        "toggle shown material first!");
                            };
                        }
                    } else {
                        CVLog::Warning(QString("Ignoring not existing texture "
                                               "image: %1")
                                               .arg(texture_file));
                    }
                }

                applyViewport(currentViewport);

                // render to image
                QImage image = ecvDisplayTools::RenderToImage(superRes, false,
                                                              true, 0);

                if (image.isNull()) {
                    QMessageBox::critical(this, "Error",
                                          "Failed to grab the screen!");
                    success = false;
                    break;
                }

                if (renderingMode == SUPER_RESOLUTION && superRes > 1) {
                    image = image.scaled(
                            image.width() / superRes, image.height() / superRes,
                            Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
                }

                if (asSeparateFrames) {
                    QString filename =
                            QString("frame_%1.png")
                                    .arg(frameIndex, 6, 10, QChar('0'));
                    QString fullPath = outputDir.filePath(filename);
                    if (!image.save(fullPath)) {
                        QMessageBox::critical(
                                this, "Error",
                                QString("Failed to save frame #%1")
                                        .arg(frameIndex + 1));
                        success = false;
                        break;
                    }
                } else {
#ifdef QFFMPEG_SUPPORT
                    QString errorString;
                    if (!encoder->encodeImage(image, frameIndex,
                                              &errorString)) {
                        QMessageBox::critical(
                                this, "Error",
                                QString("Failed to encode frame #%1: %2")
                                        .arg(frameIndex + 1)
                                        .arg(errorString));
                        success = false;
                        break;
                    }
#endif
                }

                // next frame
                currentTime += timeStep;
                ++frameIndex;

                progressDialog.setValue(frameIndex);
                QApplication::processEvents();
                if (progressDialog.wasCanceled()) {
                    QMessageBox::warning(this, "Warning",
                                         QString("Process has been cancelled"));
                    success = false;
                    break;
                }
            } else {
                // we'll try the next step
                ++vp1Index;
                currentStepStartTime += step1.duration_sec;

                if (vp1Index == segmentCount) {
                    break;
                }
            }
        }
    }

#ifdef QFFMPEG_SUPPORT
    if (encoder) {
        encoder->close();

        // hack: restore original size
        m_view3d->resize(originalViewSize);
        QApplication::processEvents();
    }
#endif

    progressDialog.hide();
    QApplication::processEvents();

    if (success) {
        QMessageBox::information(this, "Job done",
                                 "The animation has been saved successfully");
    }

    setEnabled(true);
}

void qAnimationDlg::updateTotalDuration() {
    double totalDuration_sec = computeTotalTime();

    totalTimeDoubleSpinBox->blockSignals(true);
    totalTimeDoubleSpinBox->setValue(totalDuration_sec);
    totalTimeDoubleSpinBox->blockSignals(false);
}

void qAnimationDlg::updateCurrentStepDuration() {
    int index = getCurrentStepIndex();

    stepTimeDoubleSpinBox->blockSignals(true);
    stepTimeDoubleSpinBox->setValue(
            index < 0 ? 0.0 : m_videoSteps[index].duration_sec);
    stepTimeDoubleSpinBox->blockSignals(false);
}

void qAnimationDlg::onItemChanged(QListWidgetItem*) {
    onCurrentStepChanged(stepSelectionList->currentRow());

    updateCameraTrajectory();
}

void qAnimationDlg::onCurrentStepChanged(int index) {
    // update current step descriptor
    stepIndexLabel->setText(QString::number(index + 1));

    updateCurrentStepDuration();

    if (index >= 0) {
        // apply either the current or the
        applyViewport(
                (smoothModeEnabled()
                         ? m_smoothVideoSteps[m_videoSteps[index]
                                                      .indexInSmoothTrajectory]
                         : m_videoSteps[index])
                        .viewportParams);
    }

    // check that the step is enabled
    bool isEnabled =
            (index >= 0 &&
             stepSelectionList->item(index)->checkState() == Qt::Checked);
    bool isLoop = loopCheckBox->isChecked();
    currentStepGroupBox->setEnabled(
            isEnabled &&
            ((index >= 0 && index + 1 < m_videoSteps.size()) || isLoop));
}

void qAnimationDlg::onLoopToggled(bool enabled) {
    updateCameraTrajectory();

    onCurrentStepChanged(stepSelectionList->currentRow());
}