// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// qOCR workspace dialog: a pure view in the shared AICore plugin shape.
// Bilingual (Chinese/English) PP-OCR model selection, device/threads runtime
// row, shared-cache downloads through the unified ecvModelDownloader pipeline
// (digest-verified), a checkable multi-image input list (one-click official
// test data — the real-document table pages for tables mode, the PaddleOCR
// demo images otherwise — with the mode's recommended images checked, plus
// any added files), per-image results, and the DB-export switch.
// All log output routes through the application console via
// aicore_inference_log (no private log widget); progress uses the shared
// download/inference progress section; the heavy run itself is executed by
// OCRWorker through the plugin (runRequested / cancelRequested).

#include <QDialog>
#include <QSet>
#include <QVector>

#include "OCRWorker.h"
#include "ecvTestDataRepository.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTabWidget;
class QTableWidget;
class ecvClickableImageLabel;
class ecvModelDownloader;
class QDragEnterEvent;
class QDragMoveEvent;
class QDropEvent;

/** Everything the worker needs for one run; resolved by the dialog view. */
struct OCRRunSettings {
    QString detPath;
    QString recPath;
    QString layoutPath;  // empty = layout stage off
    QString tablePath;   // empty = table stage off
    /** Checked input images, run in list order through one model load. */
    QStringList imagePaths;
    QString device = QStringLiteral("auto");
    int threads = 0;  // 0 = backend default
    /** Materialize the numbered overlay in the DB and active render view. */
    bool addToDb = true;
};

Q_DECLARE_METATYPE(OCRRunSettings)

struct aicore_ocr_preset_entry;

class OCRDialog final : public QDialog {
    Q_OBJECT
public:
    /** Measured per-stage sample recommendations for a recognition family
     *  ("zh", "en", "ja"): every entry was benchmarked with the product
     *  pipeline (bench_ocr / bench_table with layout regions, cuda) across
     *  the published recipes and quantizations; see README "Sample data".
     *  Relative to the owning bundle's extraction root. */
    static QStringList recommendedTestImages(bool table,
                                             bool layout,
                                             const QString& family);
    /** The bundle a mode+family fill lists: table mode reads the dedicated
     *  real-document pages for the zh/ja families, while the en family
     *  demos from the PaddleOCR bundle (its dictionaries cannot read the
     *  zh/ja pages; the English synthetic table-rec set lives there). */
    static ecvTestDataRepository::Dataset datasetForMode(bool table,
                                                         const QString& family);
    explicit OCRDialog(QWidget* parent = nullptr);
    ~OCRDialog() override;

    /// Plugin-facing view API (same surface as RMBGDialog).
    void appendLog(const QString& msg);
    void setRunning(bool running);
    void setTaskStage(const QString& stage, int percent);
    void setProgress(int current, int total);
    /// Append one image's outcome to the per-image result selector and show
    /// it (GUI thread only; called once per processed image).
    void showResult(const OCRRunResult& result);

    OCRRunSettings runSettings() const;

    /// Render one reconstructed table into a grid widget: one item per cell
    /// at its top-left slot, row/column spans applied (GUI thread only).
    static void fillTableGrid(QTableWidget* grid, const OCRTableResult& table);

    /// True for bundle images whose content is generated rather than a real
    /// document: the PaddleOCR test_files/table.jpg demo, the PaddleX
    /// table_rec/ GT crops (labelled synthetic tables behind the TEDS gate)
    /// and the AICore layout_doc_test.png probe page. Listed, never checked.
    static bool isSyntheticTestImage(const QString& relativePath);
    /// Every image under the dataset root, the mode's recommended images
    /// first (in recommendation order), then the rest sorted by relative
    /// path. Absolute paths.
    static QStringList testDataImages(const QString& root,
                                      bool table,
                                      bool layout,
                                      const QString& family);
    /// Append images to the input list (duplicates skipped); returns the
    /// number added.
    int addImages(const QStringList& paths, bool checked);
    /// Checked input images in list order.
    QStringList checkedImages() const;

signals:
    void runRequested(const OCRRunSettings& settings);
    void cancelRequested();
    /** The per-image result selector moved to \p imagePath (GUI thread).
     *  The plugin owns the DB identity of the exported overlays and syncs
     *  the DB tree selection with it. */
    void resultImageSelected(const QString& imagePath);

private slots:
    void onDownloadMissing();
    void onUseTestData();
    void onAddImages();
    void onRun();
    void onCancel();
    void onDownloadFinished(bool ok, const QString& destPath);
    /// ensureDatasetReady chain settlement (success or failure): exactly
    /// one call per accepted preparation request.
    void onTestDataReady(bool success, ecvTestDataRepository::Dataset kind);
    /// Load the image behind the current input-list row into the shared
    /// click-to-enlarge preview label.
    void updateInputPreview();

private:
    QString cacheDir() const;
    const aicore_ocr_preset_entry* currentPreset() const;
    /** Script family of the selected recipe, derived from its rec stem:
     *  "en", "ja", or "zh" (the ch/v5/v6 dicts cover Chinese + English, so
     *  "zh" is the auto-capable superset, not a restriction to Chinese). */
    QString recipeFamily() const;
    QString detFileName() const;
    QString recFileName() const;
    QString layoutFileName() const;
    QString tableFileName() const;
    bool modelPresent(const QString& filename) const;
    void startDownload(const QString& url,
                       const QString& destPath,
                       bool verifyDigest,
                       qint64 minBytes);
    void pumpQueue();
    void updateRunState();
    void fillTestData();
    /// Display label: the path relative to the Ppocr dataset root for
    /// test images, else the file name.
    QString imageLabel(const QString& path) const;
    /// Text-tab helpers of the commercial interaction pack: substring
    /// filter over the recognized rows, clipboard export, plain-text save.
    void applyTextFilter();
    void copyResults(bool selectedOnly);
    void saveTexts();
    void loadSettings();
    void saveSettings() const;
    void showRunResult(int index);
    void showTable(int index);
    void exportTable(bool html);
    void exportAllTables();

protected:
    void closeEvent(QCloseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;

    // ---- model selection -------------------------------------------------
    /** The recognition model: one combo over the catalog recipes with
     *  script-family labels. The former language combo is gone — its only
     *  other effect was filtering the sample-data fill, which now derives
     *  from the recipe family, so the contradictory state (language says X,
     *  model does Y) cannot be expressed. */
    QComboBox* m_preset = nullptr;  // catalog recipe (det/rec/cls pairing)
    QComboBox* m_quant = nullptr;   // F32 / F16 / Q8_0

    // ---- runtime row -------------------------------------------------------
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;

    // ---- input -------------------------------------------------------------
    QListWidget* m_imageList = nullptr;     // checkable; UserRole = path
    QPushButton* m_browseButton = nullptr;  // "Add images..."
    QPushButton* m_removeImagesButton = nullptr;
    QPushButton* m_checkAllButton = nullptr;
    QPushButton* m_checkNoneButton = nullptr;
    QPushButton* m_testDataButton = nullptr;
    /** Shared click-to-enlarge preview of the current input row
     *  (ecvClickableImageLabel, the same widget the other AICore dialogs
     *  embed). */
    ecvClickableImageLabel* m_inputPreview = nullptr;
    bool m_testDataInProgress = false;
    /** Dataset kind of the in-flight test-data preparation (captured at
     *  request time, so a mode toggle during a download cannot detach the
     *  datasetReady settlement from its flow). */
    ecvTestDataRepository::Dataset m_testDataKind =
            ecvTestDataRepository::Dataset::Ppocr;
    /** Absolute paths the last Use-test-data fill checked as its
     *  recommendations. A later fill retires them (unchecks) when they are
     *  no longer recommended, so stale suggestions from another mode or
     *  recognition family never accumulate into the checked set; genuine
     *  user checks are never touched. */
    QSet<QString> m_activeRecommendations;

    // ---- output ------------------------------------------------------------
    QCheckBox* m_layoutCheck = nullptr;
    QCheckBox* m_tableCheck = nullptr;
    QCheckBox* m_addDbCheck = nullptr;
    QComboBox* m_resultImage = nullptr;  // processed images of the run
    QPushButton* m_saveAllButton = nullptr;
    QLabel* m_resultSummary = nullptr;  // selected image's counts/timings
    /** Per-image outcomes of the last run without their source pixels (the
     *  DB overlay owns the image), so a long batch holds text + tables only. */
    QVector<OCRRunResult> m_runResults;
    QTabWidget* m_resultTabs = nullptr;
    QTableWidget* m_results = nullptr;
    QLineEdit* m_textFilter = nullptr;  // substring filter over the rows
    QPushButton* m_copyTextButton = nullptr;
    QPushButton* m_saveTextButton = nullptr;
    QString m_textSource;  // image path of the shown rows
    QComboBox* m_tableSelect = nullptr;
    QTableWidget* m_tableGrid = nullptr;
    QPushButton* m_saveHtmlButton = nullptr;
    QPushButton* m_saveCsvButton = nullptr;
    QVector<OCRTableResult> m_tables;  // last run, for view and export
    QString m_tablesSource;            // image path of m_tables

    // ---- actions / state ---------------------------------------------------
    QPushButton* m_downloadButton = nullptr;
    QPushButton* m_runButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QLabel* m_status = nullptr;
    QLabel* m_taskStatusLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    bool m_runInProgress = false;
    bool m_downloadInProgress = false;
    QString m_activeDownloadKey;
    QString m_lastTaskError;

    ecvModelDownloader* m_downloader = nullptr;
};
