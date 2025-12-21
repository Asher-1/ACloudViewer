// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionAnnotation.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkActor.h>
#include <vtkCell.h>
#include <vtkCoordinate.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

// Qt
#include <QDateTime>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

//-----------------------------------------------------------------------------
cvSelectionAnnotationManager::cvSelectionAnnotationManager(QObject* parent)
    : QObject(parent), m_viewer(nullptr) {
    CVLog::PrintDebug("[cvSelectionAnnotationManager] Initialized");
}

//-----------------------------------------------------------------------------
cvSelectionAnnotationManager::~cvSelectionAnnotationManager() { clearAll(); }

//-----------------------------------------------------------------------------
void cvSelectionAnnotationManager::setVisualizer(PclUtils::PCLVis* viewer) {
    m_viewer = viewer;
}

//-----------------------------------------------------------------------------
QString cvSelectionAnnotationManager::addAnnotation(
        const cvSelectionData& selection,
        const QString& text,
        bool autoPosition) {
    if (selection.isEmpty()) {
        CVLog::Warning(
                "[cvSelectionAnnotationManager] Cannot annotate empty "
                "selection");
        return QString();
    }

    // Generate unique ID
    QString id =
            QString("annotation_%1").arg(QDateTime::currentMSecsSinceEpoch());

    // Create annotation
    cvAnnotation annotation;
    annotation.id = id;
    annotation.text = text;
    annotation.followSelection = autoPosition;

    if (autoPosition && m_viewer) {
        // Compute selection center from polyData (ParaView-style)
        // Reference: ParaView's vtkSMSelectionHelper::ComputeSelectionBounds
        double* center = computeSelectionCenter(selection, nullptr);
        
        if (center) {
            annotation.position[0] = center[0];
            annotation.position[1] = center[1];
            annotation.position[2] = center[2];
            
            CVLog::PrintDebug(QString("[cvSelectionAnnotationManager] Annotation "
                                      "position: [%1, %2, %3]")
                                      .arg(center[0]).arg(center[1]).arg(center[2]));
            delete[] center;
        } else {
            // Fallback to origin
            annotation.position[0] = 0.0;
            annotation.position[1] = 0.0;
            annotation.position[2] = 0.0;
            CVLog::PrintDebug("[cvSelectionAnnotationManager] Using origin for "
                              "annotation (no polyData available)");
        }
    }

    m_annotations[id] = annotation;
    createTextActor(annotation);

    emit annotationAdded(id);
    emit annotationsChanged();

    CVLog::Print(QString("[cvSelectionAnnotationManager] Added annotation: %1")
                         .arg(id));

    return id;
}

//-----------------------------------------------------------------------------
QString cvSelectionAnnotationManager::addAnnotationAt(const QString& text,
                                                      const double position[3],
                                                      const QString& id) {
    QString annotationId =
            id.isEmpty() ? QString("annotation_%1")
                                   .arg(QDateTime::currentMSecsSinceEpoch())
                         : id;

    cvAnnotation annotation(text, position, annotationId);

    m_annotations[annotationId] = annotation;
    createTextActor(annotation);

    emit annotationAdded(annotationId);
    emit annotationsChanged();

    CVLog::Print(QString("[cvSelectionAnnotationManager] Added annotation at "
                         "(%1, %2, %3): %4")
                         .arg(position[0])
                         .arg(position[1])
                         .arg(position[2])
                         .arg(annotationId));

    return annotationId;
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::removeAnnotation(const QString& id) {
    if (!m_annotations.contains(id)) {
        CVLog::Warning(QString("[cvSelectionAnnotationManager] Annotation not "
                               "found: %1")
                               .arg(id));
        return false;
    }

    removeTextActor(id);
    m_annotations.remove(id);

    emit annotationRemoved(id);
    emit annotationsChanged();

    CVLog::Print(
            QString("[cvSelectionAnnotationManager] Removed annotation: %1")
                    .arg(id));

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::updateAnnotationText(const QString& id,
                                                        const QString& text) {
    if (!m_annotations.contains(id)) {
        return false;
    }

    m_annotations[id].text = text;
    updateTextActor(id);

    emit annotationUpdated(id);
    emit annotationsChanged();

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::updateAnnotationPosition(
        const QString& id, const double position[3]) {
    if (!m_annotations.contains(id)) {
        return false;
    }

    m_annotations[id].position[0] = position[0];
    m_annotations[id].position[1] = position[1];
    m_annotations[id].position[2] = position[2];
    updateTextActor(id);

    emit annotationUpdated(id);

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::setAnnotationVisible(const QString& id,
                                                        bool visible) {
    if (!m_annotations.contains(id)) {
        return false;
    }

    m_annotations[id].visible = visible;
    updateTextActor(id);

    emit annotationUpdated(id);

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::setAnnotationColor(const QString& id,
                                                      const QColor& color) {
    if (!m_annotations.contains(id)) {
        return false;
    }

    m_annotations[id].color = color;
    updateTextActor(id);

    emit annotationUpdated(id);

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::setAnnotationFontSize(const QString& id,
                                                         int fontSize) {
    if (!m_annotations.contains(id)) {
        return false;
    }

    m_annotations[id].fontSize = fontSize;
    updateTextActor(id);

    emit annotationUpdated(id);

    return true;
}

//-----------------------------------------------------------------------------
cvAnnotation cvSelectionAnnotationManager::getAnnotation(
        const QString& id) const {
    return m_annotations.value(id, cvAnnotation());
}

//-----------------------------------------------------------------------------
QList<cvAnnotation> cvSelectionAnnotationManager::allAnnotations() const {
    return m_annotations.values();
}

//-----------------------------------------------------------------------------
QStringList cvSelectionAnnotationManager::annotationIds() const {
    return m_annotations.keys();
}

//-----------------------------------------------------------------------------
void cvSelectionAnnotationManager::clearAll() {
    // Remove all actors
    for (const QString& id : m_annotations.keys()) {
        removeTextActor(id);
    }

    m_annotations.clear();
    m_textActors.clear();

    emit annotationsChanged();

    CVLog::Print("[cvSelectionAnnotationManager] All annotations cleared");
}

//-----------------------------------------------------------------------------
void cvSelectionAnnotationManager::showAll() {
    for (const QString& id : m_annotations.keys()) {
        m_annotations[id].visible = true;
        updateTextActor(id);
    }

    emit annotationsChanged();
}

//-----------------------------------------------------------------------------
void cvSelectionAnnotationManager::hideAll() {
    for (const QString& id : m_annotations.keys()) {
        m_annotations[id].visible = false;
        updateTextActor(id);
    }

    emit annotationsChanged();
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::exportToFile(const QString& filename) const {
    if (filename.isEmpty()) {
        CVLog::Error("[cvSelectionAnnotationManager] Invalid filename");
        return false;
    }

    QJsonArray annotationsArray;

    for (const cvAnnotation& annotation : m_annotations.values()) {
        QJsonObject annotationObj;
        annotationObj["id"] = annotation.id;
        annotationObj["text"] = annotation.text;
        annotationObj["x"] = annotation.position[0];
        annotationObj["y"] = annotation.position[1];
        annotationObj["z"] = annotation.position[2];
        annotationObj["colorR"] = annotation.color.red();
        annotationObj["colorG"] = annotation.color.green();
        annotationObj["colorB"] = annotation.color.blue();
        annotationObj["fontSize"] = annotation.fontSize;
        annotationObj["visible"] = annotation.visible;
        annotationObj["followSelection"] = annotation.followSelection;
        annotationObj["timestamp"] = annotation.timestamp;

        annotationsArray.append(annotationObj);
    }

    QJsonObject root;
    root["version"] = 1;
    root["count"] = m_annotations.size();
    root["annotations"] = annotationsArray;

    QJsonDocument doc(root);

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        CVLog::Error(QString("[cvSelectionAnnotationManager] Failed to open "
                             "file: %1")
                             .arg(filename));
        return false;
    }

    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();

    CVLog::Print(QString("[cvSelectionAnnotationManager] Exported %1 "
                         "annotations to: %2")
                         .arg(m_annotations.size())
                         .arg(filename));

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::importFromFile(const QString& filename,
                                                  bool merge) {
    if (filename.isEmpty()) {
        CVLog::Error("[cvSelectionAnnotationManager] Invalid filename");
        return false;
    }

    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        CVLog::Error(QString("[cvSelectionAnnotationManager] Failed to open "
                             "file: %1")
                             .arg(filename));
        return false;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isObject()) {
        CVLog::Error("[cvSelectionAnnotationManager] Invalid JSON format");
        return false;
    }

    QJsonObject root = doc.object();

    if (!merge) {
        clearAll();
    }

    QJsonArray annotationsArray = root["annotations"].toArray();
    int imported = 0;

    for (const QJsonValue& value : annotationsArray) {
        QJsonObject annotationObj = value.toObject();

        QString id = annotationObj["id"].toString();
        QString text = annotationObj["text"].toString();
        double position[3] = {annotationObj["x"].toDouble(),
                              annotationObj["y"].toDouble(),
                              annotationObj["z"].toDouble()};

        // Skip if already exists and merging
        if (merge && m_annotations.contains(id)) {
            continue;
        }

        cvAnnotation annotation(text, position, id);
        annotation.color = QColor(annotationObj["colorR"].toInt(),
                                  annotationObj["colorG"].toInt(),
                                  annotationObj["colorB"].toInt());
        annotation.fontSize = annotationObj["fontSize"].toInt(12);
        annotation.visible = annotationObj["visible"].toBool(true);
        annotation.followSelection =
                annotationObj["followSelection"].toBool(false);
        annotation.timestamp =
                annotationObj["timestamp"].toVariant().toLongLong();

        m_annotations[id] = annotation;
        createTextActor(annotation);

        ++imported;
    }

    emit annotationsChanged();

    CVLog::Print(QString("[cvSelectionAnnotationManager] Imported %1 "
                         "annotations from: %2")
                         .arg(imported)
                         .arg(filename));

    return true;
}

//-----------------------------------------------------------------------------
// Private methods
//-----------------------------------------------------------------------------

void cvSelectionAnnotationManager::createTextActor(
        const cvAnnotation& annotation) {
    if (!m_viewer) {
        return;
    }

    vtkRenderer* renderer = m_viewer->getCurrentRenderer();
    if (!renderer) {
        return;
    }

    // Create text actor
    vtkTextActor* textActor = vtkTextActor::New();
    textActor->SetInput(annotation.text.toUtf8().constData());

    // Set text properties
    vtkTextProperty* textProp = textActor->GetTextProperty();
    textProp->SetFontSize(annotation.fontSize);
    textProp->SetColor(annotation.color.redF(), annotation.color.greenF(),
                       annotation.color.blueF());
    textProp->SetBold(true);
    textProp->SetShadow(true);
    textProp->SetJustificationToCentered();

    // Convert 3D world position to 2D display coordinates (ParaView-style)
    // Reference: ParaView's vtkSMTextWidgetRepresentationProxy
    double displayPos[2] = {100, 100};  // Default fallback
    if (worldToDisplay(annotation.position, displayPos)) {
        textActor->SetPosition(displayPos[0], displayPos[1]);
        CVLog::PrintDebug(QString("[cvSelectionAnnotationManager] Annotation "
                                  "positioned at display [%1, %2]")
                                  .arg(displayPos[0])
                                  .arg(displayPos[1]));
    } else {
        // Fallback: use default position
        textActor->SetPosition(100, 100);
        CVLog::PrintDebug("[cvSelectionAnnotationManager] Using default "
                          "position for annotation");
    }

    textActor->SetVisibility(annotation.visible);

    // Add to renderer
    renderer->AddActor2D(textActor);
    m_textActors[annotation.id] = textActor;

    // Render
    vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
    if (renderWindow) {
        renderWindow->Render();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionAnnotationManager::updateTextActor(const QString& id) {
    if (!m_textActors.contains(id) || !m_annotations.contains(id)) {
        return;
    }

    vtkTextActor* textActor = m_textActors[id];
    const cvAnnotation& annotation = m_annotations[id];

    // Update text
    textActor->SetInput(annotation.text.toUtf8().constData());

    // Update properties
    vtkTextProperty* textProp = textActor->GetTextProperty();
    textProp->SetFontSize(annotation.fontSize);
    textProp->SetColor(annotation.color.redF(), annotation.color.greenF(),
                       annotation.color.blueF());

    // Update visibility
    textActor->SetVisibility(annotation.visible);

    // Render
    if (m_viewer) {
        vtkRenderer* renderer = m_viewer->getCurrentRenderer();
        if (renderer) {
            vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
            if (renderWindow) {
                renderWindow->Render();
            }
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionAnnotationManager::removeTextActor(const QString& id) {
    if (!m_textActors.contains(id)) {
        return;
    }

    vtkTextActor* textActor = m_textActors[id];

    if (m_viewer) {
        vtkRenderer* renderer = m_viewer->getCurrentRenderer();
        if (renderer) {
            renderer->RemoveActor2D(textActor);

            vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
            if (renderWindow) {
                renderWindow->Render();
            }
        }
    }

    // Smart pointer handles cleanup automatically
    m_textActors.remove(id);
}

//-----------------------------------------------------------------------------
double* cvSelectionAnnotationManager::computeSelectionCenter(
        const cvSelectionData& selection, vtkPolyData* polyData) {
    static double center[3] = {0.0, 0.0, 0.0};

    if (!polyData || selection.isEmpty()) {
        return center;
    }

    QVector<qint64> ids = selection.ids();
    double sum[3] = {0.0, 0.0, 0.0};
    int count = 0;

    if (selection.fieldAssociation() == cvSelectionData::POINTS) {
        for (qint64 id : ids) {
            if (id >= 0 && id < polyData->GetNumberOfPoints()) {
                double pt[3];
                polyData->GetPoint(id, pt);
                sum[0] += pt[0];
                sum[1] += pt[1];
                sum[2] += pt[2];
                ++count;
            }
        }
    } else {
        for (qint64 id : ids) {
            if (id >= 0 && id < polyData->GetNumberOfCells()) {
                vtkCell* cell = polyData->GetCell(id);
                if (cell) {
                    vtkIdType npts = cell->GetNumberOfPoints();
                    for (vtkIdType i = 0; i < npts; ++i) {
                        double pt[3];
                        polyData->GetPoint(cell->GetPointId(i), pt);
                        sum[0] += pt[0];
                        sum[1] += pt[1];
                        sum[2] += pt[2];
                        ++count;
                    }
                }
            }
        }
    }

    if (count > 0) {
        center[0] = sum[0] / count;
        center[1] = sum[1] / count;
        center[2] = sum[2] / count;
    }

    return center;
}

//-----------------------------------------------------------------------------
bool cvSelectionAnnotationManager::worldToDisplay(const double worldPos[3],
                                                   double displayPos[2]) {
    if (!m_viewer) {
        return false;
    }

    vtkRenderer* renderer = m_viewer->getCurrentRenderer();
    if (!renderer) {
        return false;
    }

    // Use vtkCoordinate to convert from world to display coordinates
    // Reference: ParaView's vtkSMTextWidgetRepresentationProxy
    vtkSmartPointer<vtkCoordinate> coordinate =
            vtkSmartPointer<vtkCoordinate>::New();
    coordinate->SetCoordinateSystemToWorld();
    coordinate->SetValue(worldPos[0], worldPos[1], worldPos[2]);

    // Get display position
    int* displayCoords = coordinate->GetComputedDisplayValue(renderer);
    if (displayCoords) {
        displayPos[0] = static_cast<double>(displayCoords[0]);
        displayPos[1] = static_cast<double>(displayCoords[1]);
        return true;
    }

    return false;
}
