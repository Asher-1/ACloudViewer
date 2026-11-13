// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RelateOptionsDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

#include "RelateAnythingPresets.h"

namespace {
QDoubleSpinBox* ratio(const QString& name,
                      double lo,
                      double hi,
                      double step,
                      const QString& tip,
                      QWidget* parent) {
    auto* box = new QDoubleSpinBox(parent);
    box->setObjectName(name);
    box->setRange(lo, hi);
    box->setSingleStep(step);
    box->setDecimals(2);
    box->setToolTip(tip);
    return box;
}
}  // namespace

RelateOptionsDialog::RelateOptionsDialog(
        const RelateOptions& initial,
        const QList<RelateDetectors::Info>& detectors,
        QWidget* parent)
    : QDialog(parent) {
    setObjectName(QStringLiteral("RelateOptionsDialog"));
    setWindowTitle(tr("RelateAnything options"));
    setMinimumWidth(460);
    auto* root = new QVBoxLayout(this);

    m_profile = new QComboBox(this);
    m_profile->setObjectName(QStringLiteral("RelateProfile"));
    m_profile->addItem(tr("Official video reel (default)"),
                       int(RelateOptions::Profile::VideoReel));
    m_profile->addItem(tr("Official image demo"),
                       int(RelateOptions::Profile::ImageDemo));
    m_profile->addItem(tr("Custom"), int(RelateOptions::Profile::Custom));
    m_profile->setToolTip(
            tr("Operating points of RelateAnything-GGML: the video reel "
               "(render_video.py) separates spatial and interaction "
               "relations with floors 0.45 / 0.42; the image demo "
               "(demo_webcam.py) uses one graph with floor 0.5. Editing any "
               "number switches this to Custom."));
    auto* top = new QFormLayout;
    top->addRow(tr("Operating point"), m_profile);
    root->addLayout(top);

    auto* relations = new QGroupBox(tr("Relations"), this);
    auto* rl = new QFormLayout(relations);
    m_relation = ratio(QStringLiteral("RelateRelationThreshold"), 0, 1, .05,
                       tr("Minimum score of an interaction relation."), this);
    m_spatial = ratio(QStringLiteral("RelateSpatialThreshold"), 0, 1, .05,
                      tr("Minimum score of a spatial relation (beside, "
                         "behind...)."),
                      this);
    m_maxRelations = new QSpinBox(this);
    m_maxRelations->setObjectName(QStringLiteral("RelateMaxRelations"));
    m_maxRelations->setRange(1, 128);
    m_maxRelations->setToolTip(tr("Most relations shown per image."));
    m_maxSpatial = new QSpinBox(this);
    m_maxSpatial->setObjectName(QStringLiteral("RelateMaxSpatial"));
    m_maxSpatial->setRange(0, 128);
    m_maxSpatial->setToolTip(
            tr("How many of the shown relations may be spatial. 0 = "
               "interactions only."));
    m_decompose =
            new QCheckBox(tr("Separate spatial and semantic graphs"), this);
    m_decompose->setObjectName(QStringLiteral("RelateDecompose"));
    m_decompose->setToolTip(
            tr("Rank the spatial (layout) and semantic (content) predicates "
               "in separate columns from the same forward pass."));
    rl->addRow(tr("Relation floor"), m_relation);
    rl->addRow(tr("Spatial floor"), m_spatial);
    rl->addRow(tr("Relations shown"), m_maxRelations);
    rl->addRow(tr("Spatial shown"), m_maxSpatial);
    rl->addRow(m_decompose);
    root->addWidget(relations);

    auto* objects = new QGroupBox(tr("Automatic objects"), this);
    auto* ol = new QFormLayout(objects);
    objects->setToolTip(
            tr("Used when fewer than two rectangles are selected: a detector "
               "proposes the objects and they are added as rectangles."));
    m_detectConfidence = ratio(QStringLiteral("RelateDetectConfidence"), .01, 1,
                               .05, tr("Minimum detector confidence."), this);
    m_detectIou = ratio(QStringLiteral("RelateDetectIou"), .05, 1, .05,
                        tr("Detector NMS IoU."), this);
    m_maxBoxFraction =
            ratio(QStringLiteral("RelateMaxBoxFraction"), .05, 1, .05,
                  tr("Drop detections covering more than this fraction of the "
                     "image (0.85 in the video reel). 1.0 keeps everything."),
                  this);
    m_maxBoxes = new QSpinBox(this);
    m_maxBoxes->setObjectName(QStringLiteral("RelateMaxBoxes"));
    m_maxBoxes->setRange(2, 60);
    m_maxBoxes->setToolTip(
            tr("Most objects handed to the relation head (24 in the video "
               "reel, 32 in the image demo)."));
    m_detector = new QComboBox(this);
    m_detector->setObjectName(QStringLiteral("RelateDetector"));
    for (const auto& d : detectors) m_detector->addItem(d.display, d.filename);
    m_detector->setToolTip(
            tr("Proposes the objects whose relations are predicted. "
               "Segmentation models (recommended) also give object masks, as "
               "in the official demo. Missing files are downloaded on the "
               "first run."));
    if (m_detector->count() > 0) ol->addRow(tr("Object detector"), m_detector);
    ol->addRow(tr("Detector confidence"), m_detectConfidence);
    ol->addRow(tr("Detector NMS IoU"), m_detectIou);
    ol->addRow(tr("Largest object (image fraction)"), m_maxBoxFraction);
    ol->addRow(tr("Maximum objects"), m_maxBoxes);
    root->addWidget(objects);

    auto* vocab = new QGroupBox(tr("Predicates"), this);
    auto* vl = new QFormLayout(vocab);
    m_vocabulary = new QComboBox(this);
    m_vocabulary->setObjectName(QStringLiteral("RelateVocabulary"));
    m_vocabulary->addItem(tr("Official video vocabulary (65)"), 0);
    m_vocabulary->addItem(tr("Official image demo vocabulary (35)"), 1);
    m_vocabulary->addItem(tr("Full model vocabulary"), 2);
    m_vocabulary->addItem(tr("Custom text below"), 3);
    m_predicates = new QLineEdit(this);
    m_predicates->setObjectName(QStringLiteral("RelatePredicates"));
    m_predicates->setPlaceholderText(tr("riding | holding | next to"));
    m_predicates->setClearButtonEnabled(true);
    m_predicates->setToolTip(
            tr("Relations the model may name, separated by |. They are "
               "encoded by the relation model's own text encoder, so no "
               "retraining is needed. Empty = the full model vocabulary."));
    vl->addRow(tr("Vocabulary"), m_vocabulary);
    vl->addRow(tr("Predicates"), m_predicates);
    root->addWidget(vocab);

    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttons->button(QDialogButtonBox::Ok)
            ->setObjectName(QStringLiteral("RelateOptionsOk"));
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    root->addWidget(buttons);

    connect(m_profile, QOverload<int>::of(&QComboBox::activated), this,
            [this](int) {
                applyProfile(RelateOptions::Profile(
                        m_profile->currentData().toInt()));
            });
    for (auto* spin : {m_relation, m_spatial, m_detectConfidence, m_detectIou,
                       m_maxBoxFraction})
        connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, [this]() { markCustom(); });
    for (auto* spin : {m_maxRelations, m_maxSpatial, m_maxBoxes})
        connect(spin, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [this]() { markCustom(); });
    connect(m_decompose, &QCheckBox::toggled, this, [this]() { markCustom(); });
    // A vocabulary preset writes the predicate text; typing switches the
    // preset to Custom, exactly like the standalone plugin.
    connect(m_vocabulary, QOverload<int>::of(&QComboBox::activated), this,
            [this](int) {
                const int preset = m_vocabulary->currentData().toInt();
                m_loading = true;
                if (preset == 0)
                    m_predicates->setText(RelatePresets::join(
                            RelatePresets::videoPredicates()));
                else if (preset == 1)
                    m_predicates->setText(RelatePresets::join(
                            RelatePresets::webcamPredicates()));
                else if (preset == 2)
                    m_predicates->clear();
                m_loading = false;
            });
    // Each detector family has its own official confidence.
    connect(m_detector, QOverload<int>::of(&QComboBox::activated), this,
            [this](int) {
                const auto profile = RelateOptions::Profile(
                        m_profile->currentData().toInt());
                if (profile == RelateOptions::Profile::Custom) return;
                m_loading = true;
                m_detectConfidence->setValue(RelateOptions::detectorConfidence(
                        m_detector->currentData().toString(), profile));
                m_loading = false;
            });
    connect(m_predicates, &QLineEdit::textEdited, this, [this]() {
        const int custom = m_vocabulary->findData(3);
        if (!m_loading && custom >= 0) m_vocabulary->setCurrentIndex(custom);
    });
    load(initial);
}

void RelateOptionsDialog::load(const RelateOptions& o) {
    m_loading = true;
    m_relation->setValue(o.relationThreshold);
    m_spatial->setValue(o.spatialThreshold);
    m_maxRelations->setValue(o.maxRelations);
    m_maxSpatial->setValue(o.maxSpatial);
    m_decompose->setChecked(o.decompose);
    m_detectConfidence->setValue(o.detectConfidence);
    m_detectIou->setValue(o.detectIou);
    m_maxBoxFraction->setValue(o.maxBoxFraction);
    m_maxBoxes->setValue(o.maxBoxes);
    m_predicates->setText(o.predicates);
    if (m_detector->count() > 0) {
        const int row = m_detector->findData(
                o.detector.isEmpty() ? RelateDetectors::defaultFilename()
                                     : o.detector);
        m_detector->setCurrentIndex(qMax(0, row));
    }
    const QString text = o.predicates.trimmed();
    int preset = 3;
    if (text.isEmpty())
        preset = 2;
    else if (text == RelatePresets::join(RelatePresets::videoPredicates()))
        preset = 0;
    else if (text == RelatePresets::join(RelatePresets::webcamPredicates()))
        preset = 1;
    m_vocabulary->setCurrentIndex(m_vocabulary->findData(preset));
    m_profile->setCurrentIndex(m_profile->findData(int(o.matchedProfile())));
    m_loading = false;
}

void RelateOptionsDialog::applyProfile(RelateOptions::Profile profile) {
    if (profile == RelateOptions::Profile::Custom) return;
    RelateOptions o = RelateOptions::forProfile(profile);
    o.predicates = m_predicates->text();  // the vocabulary is its own choice
    if (m_detector->count() > 0) {
        const QString file = m_detector->currentData().toString();
        if (file != RelateDetectors::defaultFilename()) o.detector = file;
        o.detectConfidence = RelateOptions::detectorConfidence(file, profile);
    }
    load(o);
    // load() resolves the profile from the numbers; keep the chosen one.
    m_profile->setCurrentIndex(m_profile->findData(int(profile)));
}

void RelateOptionsDialog::markCustom() {
    if (m_loading) return;
    const int custom = m_profile->findData(int(RelateOptions::Profile::Custom));
    if (custom >= 0) m_profile->setCurrentIndex(custom);
}

RelateOptions RelateOptionsDialog::options() const {
    RelateOptions o;
    o.relationThreshold = float(m_relation->value());
    o.spatialThreshold = float(m_spatial->value());
    o.maxRelations = m_maxRelations->value();
    o.maxSpatial = m_maxSpatial->value();
    o.decompose = m_decompose->isChecked();
    o.detectConfidence = float(m_detectConfidence->value());
    o.detectIou = float(m_detectIou->value());
    o.maxBoxFraction = float(m_maxBoxFraction->value());
    o.maxBoxes = m_maxBoxes->value();
    o.predicates = m_predicates->text().trimmed();
    if (m_detector->count() > 0 && m_detector->currentData().toString() !=
                                           RelateDetectors::defaultFilename())
        o.detector = m_detector->currentData().toString();
    o.profile = RelateOptions::Profile(m_profile->currentData().toInt());
    return o;
}
