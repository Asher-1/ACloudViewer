#pragma once

#include <ecvStdPluginInterface.h>
#include <ecvHObject.h>

#include "DA3Dialog.h"
#include "DA3Worker.h"

class ccImage;

class qDA3 : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qDA3"
                       FILE "../info.json")

public:
    explicit qDA3(QObject* parent = nullptr);
    ~qDA3() override = default;

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const DA3Dialog::Settings& settings);
    void cancelTask();
    void onDepthResult(const DA3DepthResult& result);
    void onReconResult(const DA3ReconResult& result);
    void onModelInfo(const QString& info);
    void onTaskFinished(bool success);
    void exportDepthMap();
    void exportAllDepthMaps(const QString& outputDir);

private:
    void refreshDbImages();
    ccImage* findDbImage(const QString& name) const;
    static bool saveDepthAsImage(const DA3DepthResult& result,
                                  const QString& path);

    QAction* m_action = nullptr;
    DA3Dialog* m_dialog = nullptr;
    DA3Worker* m_worker = nullptr;
    DA3Dialog::Settings m_currentSettings;
    ccHObject::Container m_selectedEntities;

    DA3DepthResult m_lastDepthResult;
    QVector<DA3DepthResult> m_allDepthResults;
    bool m_hasDepthResult = false;
};
