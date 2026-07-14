#include "qDA3.h"

#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvImage.h>
#include <ecvGLMatrix.h>

#include <QAction>
#include <QDir>
#include <QFileDialog>
#include <QMainWindow>
#include <QMessageBox>

#include <algorithm>
#include <cmath>

qDA3::qDA3(QObject* parent)
    : QObject(parent)
    , ccStdPluginInterface(":/CC/plugin/qDA3/info.json") {
    m_action = new QAction("DA3 Depth Estimation", this);
    m_action->setToolTip("Depth Anything V3 — monocular depth, pose, 3D reconstruction");
    m_action->setIcon(QIcon(":/CC/plugin/qDA3/images/qDA3.svg"));
    connect(m_action, &QAction::triggered, this, &qDA3::showDialog);
}

void qDA3::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
}

QList<QAction*> qDA3::getActions() {
    return { m_action };
}

void qDA3::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);

    QStringList names;
    for (ccHObject* obj : images) {
        if (obj && obj->isEnabled()) {
            names.append(obj->getName());
        }
    }
    m_dialog->setDbImages(names);
}

ccImage* qDA3::findDbImage(const QString& name) const {
    if (!m_app) return nullptr;
    ccHObject* root = m_app->dbRootObject();
    if (!root) return nullptr;

    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    for (ccHObject* obj : images) {
        if (obj && obj->getName() == name) {
            return dynamic_cast<ccImage*>(obj);
        }
    }
    return nullptr;
}

void qDA3::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new DA3Dialog(m_app->getMainWindow());
        connect(m_dialog, &DA3Dialog::runRequested, this, &qDA3::executeTask);
        connect(m_dialog, &DA3Dialog::cancelRequested, this, &qDA3::cancelTask);
        connect(m_dialog, &DA3Dialog::exportDepthRequested, this, &qDA3::exportDepthMap);
        connect(m_dialog, &DA3Dialog::exportAllDepthsRequested, this, &qDA3::exportAllDepthMaps);
        connect(m_dialog, &DA3Dialog::refreshDbImagesRequested, this, [this]() {
            refreshDbImages();
        });
    }
    refreshDbImages();
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qDA3::executeTask(const DA3Dialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) {
        QMessageBox::warning(m_dialog, "DA3", "A task is already running.");
        return;
    }

    DA3Dialog::Settings resolvedSettings = settings;

    if (!settings.dbImageName.isEmpty()) {
        ccImage* img = findDbImage(settings.dbImageName);
        if (img && !img->data().isNull()) {
            QString tmpDir = DA3Dialog::modelCacheDir() + "/../tmp";
            QDir().mkpath(tmpDir);
            QString tmpPath = tmpDir + "/" + settings.dbImageName + ".png";
            if (img->data().save(tmpPath)) {
                resolvedSettings.inputPaths = QStringList() << tmpPath;
                m_dialog->appendLog(
                        tr("[DA3] Using DB image: %1 (%2x%3)")
                                .arg(settings.dbImageName)
                                .arg(img->getW())
                                .arg(img->getH()));
            } else {
                m_dialog->appendLog(
                        tr("[Error] Failed to export DB image: %1")
                                .arg(settings.dbImageName));
                return;
            }
        } else {
            m_dialog->appendLog(
                    tr("[Error] DB image not found or empty: %1")
                            .arg(settings.dbImageName));
            return;
        }
    }

    if (resolvedSettings.modelPath.isEmpty() &&
        resolvedSettings.mode != DA3Dialog::Mode::Quantize) {
        m_dialog->appendLog("[Error] Please select a GGUF model file.");
        return;
    }

    bool needsInput = (resolvedSettings.mode != DA3Dialog::Mode::Quantize &&
                        resolvedSettings.mode != DA3Dialog::Mode::ModelInfo);
    if (needsInput && resolvedSettings.inputPaths.isEmpty()) {
        m_dialog->appendLog(
                "[Error] No input image selected. Use 'Browse...' to select "
                "an image file, or choose an image from 'DB Images' dropdown.");
        return;
    }

    m_currentSettings = resolvedSettings;
    m_hasDepthResult = false;
    m_allDepthResults.clear();
    m_worker = new DA3Worker(resolvedSettings, this);
    connect(m_worker, &DA3Worker::logMessage, m_dialog, &DA3Dialog::appendLog);
    connect(m_worker, &DA3Worker::progressUpdate, m_dialog, &DA3Dialog::setProgress);
    connect(m_worker, &DA3Worker::depthResultReady, this, &qDA3::onDepthResult);
    connect(m_worker, &DA3Worker::reconResultReady, this, &qDA3::onReconResult);
    connect(m_worker, &DA3Worker::modelInfoReady, this, &qDA3::onModelInfo);
    connect(m_worker, &DA3Worker::taskFinished, this, &qDA3::onTaskFinished);

    m_dialog->setRunning(true);
    m_dialog->appendLog("[DA3] Starting task...");
    m_worker->start();
}

void qDA3::cancelTask() {
    if (m_worker && m_worker->isRunning()) {
        m_worker->requestInterruption();
        m_dialog->appendLog("[DA3] Cancel requested...");
    }
}

static QImage depthToGrayscaleImage(const QVector<float>& depth, int W, int H,
                                     bool invert = true) {
    if (depth.isEmpty() || W <= 0 || H <= 0) return {};
    float dmin = *std::min_element(depth.begin(), depth.end());
    float dmax = *std::max_element(depth.begin(), depth.end());
    float drange = (dmax - dmin > 1e-6f) ? (dmax - dmin) : 1.0f;

    QImage img(W, H, QImage::Format_Grayscale8);
    for (int y = 0; y < H; ++y) {
        uchar* line = img.scanLine(y);
        for (int x = 0; x < W; ++x) {
            float t = (depth[y * W + x] - dmin) / drange;
            t = std::clamp(t, 0.0f, 1.0f);
            if (invert) t = 1.0f - t;
            line[x] = static_cast<uchar>(t * 255.0f);
        }
    }
    return img;
}

void qDA3::onDepthResult(const DA3DepthResult& result) {
    if (!m_app) return;

    m_lastDepthResult = result;
    m_allDepthResults.append(result);
    m_hasDepthResult = true;

    const int W = result.width;
    const int H = result.height;
    const int step = std::max(1, m_currentSettings.downsampleStep);
    const bool unproject = m_currentSettings.unproject3D && result.hasPose;

    const int sampledW = (W + step - 1) / step;
    const int sampledH = (H + step - 1) / step;
    const int N = sampledW * sampledH;

    auto* cloud = new ccPointCloud(
        QString("DA3_Depth_%1").arg(result.sourceName));

    if (!cloud->reserve(static_cast<unsigned>(N))) {
        m_dialog->appendLog("[Error] Failed to allocate point cloud.");
        delete cloud;
        return;
    }

    auto* depthSF = new ccScalarField("Depth");
    bool hasSF = depthSF->reserveSafe(static_cast<unsigned>(N));

    auto* confSF = new ccScalarField("Confidence");
    bool hasConf = !result.confidence.isEmpty() &&
                   confSF->reserveSafe(static_cast<unsigned>(N));

    float fx = 1.0f, fy = 1.0f, cx = 0.0f, cy = 0.0f;
    float R[9] = {1,0,0, 0,1,0, 0,0,1};
    float t[3] = {0,0,0};

    if (unproject) {
        fx = result.intrinsics[0];
        fy = result.intrinsics[4];
        cx = result.intrinsics[2];
        cy = result.intrinsics[5];

        for (int i = 0; i < 3; ++i) {
            R[i * 3 + 0] = result.extrinsics[i * 4 + 0];
            R[i * 3 + 1] = result.extrinsics[i * 4 + 1];
            R[i * 3 + 2] = result.extrinsics[i * 4 + 2];
            t[i]         = result.extrinsics[i * 4 + 3];
        }
    }

    float dmin = *std::min_element(result.depth.begin(), result.depth.end());
    float dmax = *std::max_element(result.depth.begin(), result.depth.end());
    float drange = (dmax - dmin > 1e-6f) ? (dmax - dmin) : 1.0f;

    for (int sy = 0; sy < sampledH; ++sy) {
        int y = sy * step;
        if (y >= H) y = H - 1;

        for (int sx = 0; sx < sampledW; ++sx) {
            int x = sx * step;
            if (x >= W) x = W - 1;

            float d = result.depth[y * W + x];

            if (unproject) {
                float xc = (static_cast<float>(x) - cx) * d / fx;
                float yc = (static_cast<float>(y) - cy) * d / fy;
                float zc = d;

                float xw = R[0]*xc + R[1]*yc + R[2]*zc + t[0];
                float yw = R[3]*xc + R[4]*yc + R[5]*zc + t[1];
                float zw = R[6]*xc + R[7]*yc + R[8]*zc + t[2];

                cloud->addPoint(CCVector3(
                    static_cast<PointCoordinateType>(xw),
                    static_cast<PointCoordinateType>(-yw),
                    static_cast<PointCoordinateType>(-zw)));
            } else {
                float z = (d - dmin) / drange;
                cloud->addPoint(CCVector3(
                    static_cast<PointCoordinateType>(x),
                    static_cast<PointCoordinateType>(H - 1 - y),
                    static_cast<PointCoordinateType>(z * 100.0f)));
            }

            if (hasSF)
                depthSF->addElement(static_cast<ScalarType>(d));

            if (hasConf)
                confSF->addElement(
                    static_cast<ScalarType>(result.confidence[y * W + x]));
        }
    }

    if (hasSF) {
        depthSF->computeMinAndMax();
        int sfIdx = cloud->addScalarField(depthSF);
        cloud->setCurrentDisplayedScalarField(sfIdx);
        cloud->showSF(true);
    } else {
        depthSF->release();
    }

    if (hasConf) {
        confSF->computeMinAndMax();
        cloud->addScalarField(confSF);
    } else {
        confSF->release();
    }

    if (result.hasPose) {
        cloud->setMetaData("DA3_fx", QVariant(static_cast<double>(fx)));
        cloud->setMetaData("DA3_fy", QVariant(static_cast<double>(fy)));
        cloud->setMetaData("DA3_cx", QVariant(static_cast<double>(cx)));
        cloud->setMetaData("DA3_cy", QVariant(static_cast<double>(cy)));
        cloud->setMetaData("DA3_unproject", QVariant(unproject));

        ccGLMatrix poseMatrix;
        float* mat = poseMatrix.data();
        for (int r = 0; r < 3; ++r) {
            mat[r]      = result.extrinsics[r * 4 + 0];
            mat[4 + r]  = result.extrinsics[r * 4 + 1];
            mat[8 + r]  = result.extrinsics[r * 4 + 2];
            mat[12 + r] = result.extrinsics[r * 4 + 3];
        }
        mat[3] = mat[7] = mat[11] = 0.0f;
        mat[15] = 1.0f;
        cloud->setGLTransformationHistory(poseMatrix);
    }

    cloud->setVisible(true);
    cloud->setEnabled(true);

    auto* depthImage = new ccImage(
        depthToGrayscaleImage(result.depth, W, H, m_currentSettings.invertDepth),
        QString("DA3_DepthMap_%1").arg(result.sourceName));
    depthImage->setVisible(true);
    depthImage->setEnabled(true);

    m_app->addToDB(cloud);
    m_app->addToDB(depthImage);

    QString modeStr = unproject ? "3D unprojected" : "2D grid";
    m_dialog->appendLog(
        QString("[DA3] Depth cloud '%1' + depth image added (%2x%3 step=%4, %5 pts, %6)")
            .arg(cloud->getName()).arg(W).arg(H).arg(step).arg(N).arg(modeStr));
}

void qDA3::onReconResult(const DA3ReconResult& result) {
    if (!m_app) return;

    const int N = result.count;
    auto* cloud = new ccPointCloud(
        QString("DA3_Recon_%1").arg(result.sourceName));

    if (!cloud->reserve(static_cast<unsigned>(N))) {
        m_dialog->appendLog("[Error] Failed to allocate point cloud.");
        delete cloud;
        return;
    }

    bool hasColors = (result.colors.size() == N * 3) &&
                      cloud->reserveTheRGBTable();

    for (int i = 0; i < N; ++i) {
        cloud->addPoint(CCVector3(
            static_cast<PointCoordinateType>(result.positions[i * 3 + 0]),
            static_cast<PointCoordinateType>(result.positions[i * 3 + 1]),
            static_cast<PointCoordinateType>(result.positions[i * 3 + 2])));

        if (hasColors) {
            auto r = static_cast<ColorCompType>(
                std::clamp(result.colors[i * 3 + 0] * 255.0f, 0.0f, 255.0f));
            auto g = static_cast<ColorCompType>(
                std::clamp(result.colors[i * 3 + 1] * 255.0f, 0.0f, 255.0f));
            auto b = static_cast<ColorCompType>(
                std::clamp(result.colors[i * 3 + 2] * 255.0f, 0.0f, 255.0f));
            cloud->addRGBColor(r, g, b);
        }
    }

    if (result.opacities.size() == N) {
        auto* opacitySF = new ccScalarField("Opacity");
        if (opacitySF->reserveSafe(static_cast<unsigned>(N))) {
            for (int i = 0; i < N; ++i) {
                opacitySF->addElement(static_cast<ScalarType>(result.opacities[i]));
            }
            opacitySF->computeMinAndMax();
            cloud->addScalarField(opacitySF);
        } else {
            opacitySF->release();
        }
    }

    if (result.scales.size() == N * 3) {
        auto* scaleSF = new ccScalarField("Scale");
        if (scaleSF->reserveSafe(static_cast<unsigned>(N))) {
            for (int i = 0; i < N; ++i) {
                float s = std::sqrt(
                    result.scales[i * 3 + 0] * result.scales[i * 3 + 0] +
                    result.scales[i * 3 + 1] * result.scales[i * 3 + 1] +
                    result.scales[i * 3 + 2] * result.scales[i * 3 + 2]);
                scaleSF->addElement(static_cast<ScalarType>(s));
            }
            scaleSF->computeMinAndMax();
            cloud->addScalarField(scaleSF);
        } else {
            scaleSF->release();
        }
    }

    if (hasColors) cloud->showColors(true);
    cloud->setVisible(true);
    cloud->setEnabled(true);
    m_app->addToDB(cloud);

    m_dialog->appendLog(QString("[DA3] Reconstruction '%1' added (%2 gaussians)")
                        .arg(cloud->getName()).arg(N));
}

void qDA3::onModelInfo(const QString& info) {
    m_dialog->appendLog("[DA3] Model info:\n" + info);
}

void qDA3::onTaskFinished(bool success) {
    Q_UNUSED(success);
    m_dialog->appendLog("[DA3] Task finished.");
    m_dialog->setRunning(false);
    m_dialog->enableExportButtons(m_hasDepthResult);
    if (m_worker) {
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    // Clean up temporary files created for DB image export
    const QString tmpDir = DA3Dialog::modelCacheDir() + "/../tmp";
    QDir tmp(tmpDir);
    if (tmp.exists()) {
        tmp.removeRecursively();
    }

    if (m_app) {
        m_app->updateUI();
        m_app->refreshAll();
    }
}

bool qDA3::saveDepthAsImage(const DA3DepthResult& result,
                             const QString& path) {
    const int W = result.width;
    const int H = result.height;
    if (W <= 0 || H <= 0 || result.depth.isEmpty()) return false;

    const auto& depth = result.depth;
    float dmin = *std::min_element(depth.begin(), depth.end());
    float dmax = *std::max_element(depth.begin(), depth.end());
    float drange = (dmax - dmin > 1e-6f) ? (dmax - dmin) : 1.0f;

    QImage depthImg(W, H, QImage::Format_Grayscale8);
    for (int y = 0; y < H; ++y) {
        uchar* line = depthImg.scanLine(y);
        for (int x = 0; x < W; ++x) {
            float d = depth[y * W + x];
            float norm = (d - dmin) / drange;
            line[x] = static_cast<uchar>(
                    std::clamp(norm * 255.0f, 0.0f, 255.0f));
        }
    }
    return depthImg.save(path);
}

void qDA3::exportDepthMap() {
    if (!m_hasDepthResult || m_lastDepthResult.depth.isEmpty()) {
        QMessageBox::information(m_dialog, "DA3",
                                 "No depth result available. Run depth "
                                 "estimation first.");
        return;
    }

    QString path = QFileDialog::getSaveFileName(
            m_dialog, "Save Depth Map", QString(),
            "PNG Image (*.png);;TIFF Image (*.tiff *.tif);;All Files (*)");
    if (path.isEmpty()) return;

    if (saveDepthAsImage(m_lastDepthResult, path)) {
        m_dialog->appendLog(
                QString("[DA3] Depth map saved: %1 (%2x%3)")
                        .arg(path)
                        .arg(m_lastDepthResult.width)
                        .arg(m_lastDepthResult.height));
    } else {
        m_dialog->appendLog(
                QString("[Error] Failed to save depth map: %1").arg(path));
    }
}

void qDA3::exportAllDepthMaps(const QString& outputDir) {
    if (m_allDepthResults.isEmpty()) {
        QMessageBox::information(m_dialog, "DA3",
                                 "No depth results available. Run depth "
                                 "estimation first.");
        return;
    }

    QDir().mkpath(outputDir);
    int saved = 0;
    for (int i = 0; i < m_allDepthResults.size(); ++i) {
        const auto& r = m_allDepthResults[i];
        QString name = r.sourceName.isEmpty()
                               ? QString("depth_%1").arg(i, 4, 10, QChar('0'))
                               : r.sourceName;
        QString path = outputDir + "/" + name + "_depth.png";
        if (saveDepthAsImage(r, path)) {
            saved++;
        } else {
            m_dialog->appendLog(
                    QString("[Error] Failed to save: %1").arg(path));
        }
    }
    m_dialog->appendLog(
            QString("[DA3] Exported %1/%2 depth maps to: %3")
                    .arg(saved)
                    .arg(m_allDepthResults.size())
                    .arg(outputDir));
}
