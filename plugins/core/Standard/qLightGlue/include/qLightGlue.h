#pragma once

#include <ecvHObject.h>
#include <ecvStdPluginInterface.h>

#include <QAction>

#include "LightGlueDialog.h"
#include "LightGlueWorker.h"

class ccImage;

class qLightGlue : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qLightGlue" FILE
                          "../info.json")

public:
    explicit qLightGlue(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const LightGlueDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const LightGlueRunResult& result);
    void onModelInfo(const QString& info);
    void onTaskFinished(bool success);
    void onExportMatches();

private:
    bool warmupInferenceBackend(const QString& device, QString* logMsg) const;
    bool resolveInputPaths(const QStringList& rawPaths, QStringList& outPaths,
                           QString* errorMsg) const;
    void refreshDbImages();
    ccImage* findDbImage(const QString& name) const;
    QStringList selectedDbImageNames() const;
    QImage loadImageForPath(const QString& path) const;
    void addVisualizationToDb(const LightGlueRunResult& result);

    QAction* m_action = nullptr;
    LightGlueDialog* m_dialog = nullptr;
    LightGlueWorker* m_worker = nullptr;
    LightGlueDialog::Settings m_currentSettings;
    LightGlueRunResult m_lastResult;
    QStringList m_originalInputPaths;
    ccHObject::Container m_selectedEntities;
};
