#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QFile>
#include <QGroupBox>
#include <QHash>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTextEdit>

struct LightGlueBuiltinModel {
    QString displayName;
    QString filename;
    QString downloadUrl;
    int matcherType = 2;
};

class LightGlueDialog : public QDialog {
    Q_OBJECT

public:
    enum class Mode { Match, ModelInfo };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    struct Settings {
        Mode mode = Mode::Match;
        QString modelPath;
        QStringList inputPaths;
        int threads = 0;
        QString device = "auto";
        double minScore = 0.1;
        int matcherType = 2;
        bool addResultToDb = true;
    };

    explicit LightGlueDialog(QWidget* parent = nullptr);

    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setRunning(bool running);
    void setTaskStage(const QString& stage, int percent = -1);
    void enableExportButton(bool enabled);

    void setDbImages(const QList<DbImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);

    static QString modelCacheDir();

signals:
    void runRequested(const LightGlueDialog::Settings& settings);
    void cancelRequested();
    void exportMatchesRequested();
    void refreshDbImagesRequested();

private slots:
    void onBrowseFile();
    void onBrowseFolder();
    void onBrowseCustomModel();
    void onModelComboChanged(int index);
    void onDbListItemChanged(QListWidgetItem* item);
    void onClearImages();
    void onRemoveInputItem();
    void onMatchToggled(bool checked);
    void onModeChanged(int index);
    void onRun();
    void onCancel();
    void onExportMatches();

private:
    void setupUi();
    void populateModelCombo(const QString& keepFilename = QString());
    bool selectModelByFilename(const QString& filename);
    QString resolveModelPath() const;
    bool ensureModelAvailable();
    void startDownload(const LightGlueBuiltinModel& model);
    void cancelDownload();
    void updateImageStatus();
    void updateRunButtonState();
    void refreshThumbnailStrip();
    void addInputPaths(const QStringList& paths, bool replace);
    void removeInputPath(const QString& path);
    QImage previewForPath(const QString& path) const;
    bool isModelReady() const;
    bool isInputValid() const;

    static QVector<LightGlueBuiltinModel> builtinModels();
    static QString formatFileSize(qint64 bytes);

    QComboBox* m_modeCombo = nullptr;
    QGroupBox* m_ioGroup = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QPushButton* m_browseCustomModelBtn = nullptr;

    QStringList m_inputPaths;
    QStringList m_matchPaths;

    QScrollArea* m_thumbScroll = nullptr;
    QWidget* m_thumbContainer = nullptr;
    QLabel* m_imageStatusLabel = nullptr;

    QListWidget* m_dbImageList = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QDoubleSpinBox* m_minScore = nullptr;
    QCheckBox* m_addToDbCheck = nullptr;

    QLabel* m_downloadLabel = nullptr;
    QLabel* m_taskStatusLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QTextEdit* m_log = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QPushButton* m_exportBtn = nullptr;

    QNetworkAccessManager* m_netManager = nullptr;
    QNetworkReply* m_currentDownload = nullptr;
    QFile* m_downloadOutFile = nullptr;
    bool m_downloadInProgress = false;
    bool m_taskRunning = false;
    bool m_autoRunAfterDownload = false;
    QString m_downloadTargetFilename;
    QString m_downloadTmpPath;

    QHash<QString, QImage> m_dbPreviews;
};
