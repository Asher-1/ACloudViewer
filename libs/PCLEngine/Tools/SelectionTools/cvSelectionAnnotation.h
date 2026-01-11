// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
// Qt - must be included before qPCL.h for MOC to work correctly
#include <QtGui/QColor>
#include <QtCore/QDateTime>
#include <QtGui/QFont>
#include <QtCore/QList>
#include <QtCore/QMap>
#include <QtCore/QObject>
#include <QtCore/QString>
// clang-format on

#include "qPCL.h"

// LOCAL
#include "cvSelectionData.h"

// Forward declarations
class vtkActor2D;
class vtkPolyData;
class vtkTextActor;
namespace PclUtils {
class PCLVis;
}

// Include for LabelProperties struct (nested type requires full definition)
#include "cvSelectionLabelPropertiesDialog.h"

/**
 * @brief Annotation for a selection
 *
 * Stores text annotation, position, style, and visibility.
 */
struct QPCL_ENGINE_LIB_API cvAnnotation {
    QString text;                     ///< Annotation text
    QString id;                       ///< Unique ID
    double position[3];               ///< 3D position
    QColor color;                     ///< Text color
    int fontSize;                     ///< Font size in points
    QString fontFamily;               ///< Font family (e.g., "Arial")
    bool bold;                        ///< Bold flag
    bool italic;                      ///< Italic flag
    bool shadow;                      ///< Shadow flag
    double opacity;                   ///< Text opacity (0.0 to 1.0)
    QString horizontalJustification;  ///< "Left", "Center", "Right"
    QString verticalJustification;    ///< "Top", "Center", "Bottom"
    bool visible;                     ///< Visibility flag
    bool followSelection;             ///< Auto-update position with selection
    qint64 timestamp;                 ///< Creation timestamp

    cvAnnotation()
        : fontSize(12),
          fontFamily("Arial"),
          bold(false),
          italic(false),
          shadow(true),
          opacity(1.0),
          horizontalJustification("Left"),
          verticalJustification("Bottom"),
          visible(true),
          followSelection(false),
          timestamp(0) {
        position[0] = position[1] = position[2] = 0.0;
        color = Qt::yellow;
    }

    cvAnnotation(const QString& txt,
                 const double pos[3],
                 const QString& uid = QString())
        : text(txt),
          id(uid.isEmpty()
                     ? QString::number(QDateTime::currentMSecsSinceEpoch())
                     : uid),
          fontSize(12),
          fontFamily("Arial"),
          bold(false),
          italic(false),
          shadow(true),
          opacity(1.0),
          horizontalJustification("Left"),
          verticalJustification("Bottom"),
          visible(true),
          followSelection(false),
          timestamp(QDateTime::currentMSecsSinceEpoch()) {
        position[0] = pos[0];
        position[1] = pos[1];
        position[2] = pos[2];
        color = Qt::yellow;
    }
};

/**
 * @brief Selection annotation manager
 *
 * Manages text annotations for selections:
 * - Add/remove annotations
 * - Show/hide annotations
 * - Update annotation text and style
 * - Export/import annotations
 * - Auto-position at selection center
 *
 * Based on ParaView's annotation functionality.
 */
class QPCL_ENGINE_LIB_API cvSelectionAnnotationManager : public QObject {
    Q_OBJECT

public:
    explicit cvSelectionAnnotationManager(QObject* parent = nullptr);
    ~cvSelectionAnnotationManager() override;

    /**
     * @brief Set the visualizer
     */
    void setVisualizer(PclUtils::PCLVis* viewer);

    /**
     * @brief Add annotation for a selection
     * @param selection Selection data
     * @param text Annotation text
     * @param autoPosition If true, position at selection center
     * @return Annotation ID
     */
    QString addAnnotation(const cvSelectionData& selection,
                          const QString& text,
                          bool autoPosition = true);

    /**
     * @brief Add annotation at specific position
     * @param text Annotation text
     * @param position 3D position
     * @param id Optional custom ID
     * @return Annotation ID
     */
    QString addAnnotationAt(const QString& text,
                            const double position[3],
                            const QString& id = QString());

    /**
     * @brief Remove annotation
     * @param id Annotation ID
     * @return True if removed
     */
    bool removeAnnotation(const QString& id);

    /**
     * @brief Update annotation text
     * @param id Annotation ID
     * @param text New text
     * @return True if updated
     */
    bool updateAnnotationText(const QString& id, const QString& text);

    /**
     * @brief Update annotation position
     * @param id Annotation ID
     * @param position New position
     * @return True if updated
     */
    bool updateAnnotationPosition(const QString& id, const double position[3]);

    /**
     * @brief Set annotation visibility
     * @param id Annotation ID
     * @param visible Visibility flag
     * @return True if updated
     */
    bool setAnnotationVisible(const QString& id, bool visible);

    /**
     * @brief Set annotation color
     * @param id Annotation ID
     * @param color Text color
     * @return True if updated
     */
    bool setAnnotationColor(const QString& id, const QColor& color);

    /**
     * @brief Set annotation font size
     * @param id Annotation ID
     * @param fontSize Font size in points
     * @return True if updated
     */
    bool setAnnotationFontSize(const QString& id, int fontSize);

    /**
     * @brief Apply label properties to all annotations (for cell or point
     * labels)
     * @param props Label properties from dialog
     * @param isCellLabel If true, apply cell label properties; if false, apply
     * point label properties
     */
    void applyLabelProperties(
            const cvSelectionLabelPropertiesDialog::LabelProperties& props,
            bool isCellLabel);

    /**
     * @brief Set default label properties for new annotations
     * @param props Label properties from dialog
     * @param isCellLabel If true, use cell label properties; if false, use
     * point label properties
     */
    void setDefaultLabelProperties(
            const cvSelectionLabelPropertiesDialog::LabelProperties& props,
            bool isCellLabel);

    /**
     * @brief Get annotation
     * @param id Annotation ID
     * @return Annotation data
     */
    cvAnnotation getAnnotation(const QString& id) const;

    /**
     * @brief Get all annotations
     */
    QList<cvAnnotation> allAnnotations() const;

    /**
     * @brief Get annotation IDs
     */
    QStringList annotationIds() const;

    /**
     * @brief Get annotation count
     */
    int count() const { return m_annotations.size(); }

    /**
     * @brief Clear all annotations
     */
    void clearAll();

    /**
     * @brief Show all annotations
     */
    void showAll();

    /**
     * @brief Hide all annotations
     */
    void hideAll();

    /**
     * @brief Export annotations to JSON file
     * @param filename Output filename
     * @return True on success
     */
    bool exportToFile(const QString& filename) const;

    /**
     * @brief Import annotations from JSON file
     * @param filename Input filename
     * @param merge If true, merge; if false, replace
     * @return True on success
     */
    bool importFromFile(const QString& filename, bool merge = true);

signals:
    /**
     * @brief Emitted when annotations change
     */
    void annotationsChanged();

    /**
     * @brief Emitted when annotation is added
     */
    void annotationAdded(const QString& id);

    /**
     * @brief Emitted when annotation is removed
     */
    void annotationRemoved(const QString& id);

    /**
     * @brief Emitted when annotation is updated
     */
    void annotationUpdated(const QString& id);

private:
    void createTextActor(const cvAnnotation& annotation);
    void updateTextActor(const QString& id);
    void removeTextActor(const QString& id);
    double* computeSelectionCenter(const cvSelectionData& selection,
                                   vtkPolyData* polyData);

    /**
     * @brief Convert 3D world coordinates to 2D display coordinates
     * @param worldPos 3D position in world coordinates
     * @param displayPos Output 2D position in display coordinates [x, y]
     * @return True if conversion was successful
     */
    bool worldToDisplay(const double worldPos[3], double displayPos[2]);

    PclUtils::PCLVis* m_viewer;
    QMap<QString, cvAnnotation> m_annotations;
    QMap<QString, vtkSmartPointer<vtkTextActor>> m_textActors;

    // Default label properties for new annotations (applied from dialog)
    cvSelectionLabelPropertiesDialog::LabelProperties m_defaultCellLabelProps;
    cvSelectionLabelPropertiesDialog::LabelProperties m_defaultPointLabelProps;
};
