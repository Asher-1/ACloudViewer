// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QDialog>
#include <QList>

#include "RelateDetectors.h"
#include "RelateOptions.h"

class QComboBox;
class QDoubleSpinBox;
class QSpinBox;
class QCheckBox;
class QLineEdit;

/** Popup for the RelateAnything thresholds, edge caps and custom predicates.
 *  Mirrors the options of the standalone qRelateAnything plugin: an operating
 *  point (official video reel / official image demo / custom) fills every
 *  number, and touching a number flips the operating point to Custom. */
class RelateOptionsDialog : public QDialog {
    Q_OBJECT
public:
    /** `detectors` fills the object-detector combo
     * (RelateDetectors::catalog()); an empty list hides the choice and keeps
     * the official default. */
    explicit RelateOptionsDialog(
            const RelateOptions& initial,
            const QList<RelateDetectors::Info>& detectors = {},
            QWidget* parent = nullptr);
    RelateOptions options() const;

private:
    void applyProfile(RelateOptions::Profile profile);
    void markCustom();
    void load(const RelateOptions& options);

    QComboBox* m_profile = nullptr;
    QComboBox* m_vocabulary = nullptr;
    QComboBox* m_detector = nullptr;
    QDoubleSpinBox* m_relation = nullptr;
    QDoubleSpinBox* m_spatial = nullptr;
    QSpinBox* m_maxRelations = nullptr;
    QSpinBox* m_maxSpatial = nullptr;
    QCheckBox* m_decompose = nullptr;
    QDoubleSpinBox* m_detectConfidence = nullptr;
    QDoubleSpinBox* m_detectIou = nullptr;
    QDoubleSpinBox* m_maxBoxFraction = nullptr;
    QSpinBox* m_maxBoxes = nullptr;
    QLineEdit* m_predicates = nullptr;
    bool m_loading = false;
};
